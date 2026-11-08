#!/usr/bin/env python3
"""Local reverse proxy that moves an Anthropic `system` prompt into the first
user message.

Why this exists: some third-party DeepSeek-compatible gateways silently drop or
downgrade the top-level `system` field of an Anthropic Messages request. The
prompt text then never reaches the model, so no amount of upstream prompt
engineering can take effect. Embedding the same text at the head of the first
user message survives any gateway that forwards `messages`, because it is no
longer a privileged field.

Wire format: Anthropic Messages (`POST /v1/messages`), authentication via
`x-api-key`, responses are `text/event-stream` when `stream: true`. This script
forwards everything else unchanged, so it is transport-agnostic about features
it does not touch.

Design constraints:

* Standard library only. No `requests`, no `httpx`, no `aiohttp`.
* Streaming is never buffered. Response chunks are relayed with
  `read1()` and written through immediately using chunked transfer encoding, so
  token-by-token display keeps working.
* One rewrite only: `system` -> head of the first user message. Nothing else in
  the body is reordered, and `tools` / `messages` / sampling parameters are
  forwarded byte-for-byte after re-serialization.

This is a workaround for a gateway defect, not a feature of the protocol. If the
gateway starts honouring `system`, set `--keep-system` off (the default) and the
proxy simply relocates the text; run with `--probe` to inspect what arrives.

Usage:

    python system-shim.py --upstream http://host:7863/anthropic/ --listen 8787

Then point the client's base URL at the proxy root:

    baseURL: http://127.0.0.1:8787/

The proxy appends the request path to the upstream base path, so
`/v1/messages` becomes `<upstream>/v1/messages`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.parse
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Headers a proxy must not forward verbatim: they describe the connection the
# client has with us, not the one we open upstream.
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

# Headers we always recompute or deliberately replace.
REPLACED = {"host", "content-length", "accept-encoding"}

# Paths whose JSON body carries a `system` field worth relocating.
REWRITE_SUFFIXES = ("/v1/messages", "/v1/messages/count_tokens", "/messages")


def system_to_text(system) -> str:
    """Flatten an Anthropic `system` value into one string.

    The field is either a string or a list of content blocks. Blocks that are
    not text (an image, for instance) carry no instruction and are dropped
    rather than forwarded as garbage. ``cache_control`` markers are discarded on
    purpose: they describe prefix caching for a system block that no longer
    exists at system rank.
    """
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "\n\n".join(part for part in parts if part.strip())
    return ""


def _as_blocks(content):
    """Return message ``content`` as a content-block list.

    Anthropic accepts a bare string or a block list. The head block has to be
    inserted as a block, so a string form is upgraded here. A ``None`` or
    unrecognised value becomes an empty list rather than raising, because a
    malformed message is still better forwarded than dropped by the proxy.
    """
    if isinstance(content, list):
        return list(content)
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return []


def rewrite_payload(payload: dict, keep_system: bool = False):
    """Move ``system`` to the head of the first user message.

    Returns ``(new_payload, moved)``. ``moved`` is False when the request
    carried no usable system text, in which case the payload is returned
    untouched so unrelated fields keep their exact values.

    The rewrite removes ``system`` by default: keeping it would duplicate the
    instructions on any gateway that *does* honour the field, and duplicated
    instructions are worse than either placement alone.
    """
    if not isinstance(payload, dict):
        return payload, False

    text = system_to_text(payload.get("system"))
    if not text.strip():
        return payload, False

    if keep_system:
        new_payload = dict(payload)
    else:
        new_payload = {key: value for key, value in payload.items() if key != "system"}

    head = {"type": "text", "text": text}
    messages = new_payload.get("messages")

    if not isinstance(messages, list) or not messages:
        # A request with no messages has nowhere to prepend into. Synthesising
        # one keeps the instruction visible instead of silently dropping it.
        new_payload["messages"] = [{"role": "user", "content": [head]}]
        return new_payload, True

    messages = list(messages)
    first = messages[0]

    if not isinstance(first, dict) or first.get("role") != "user":
        # The system text must lead the conversation, and a leading assistant
        # turn would make the placement meaningless. Insert a fresh user turn.
        messages.insert(0, {"role": "user", "content": [head]})
    else:
        upgraded = dict(first)
        upgraded["content"] = [head] + _as_blocks(first.get("content"))
        messages[0] = upgraded

    new_payload["messages"] = messages
    return new_payload, True


def wants_rewrite(path: str) -> bool:
    """Whether ``path`` is a Messages call whose body needs the rewrite."""
    bare = urllib.parse.urlsplit(path).path
    return any(bare.endswith(suffix) for suffix in REWRITE_SUFFIXES)


class Recorder:
    """Optional on-disk capture of request bodies, for verifying a gateway."""

    def __init__(self, directory: str | None):
        self.directory = directory
        self._lock = threading.Lock()
        self._seq = 0
        if directory:
            os.makedirs(directory, exist_ok=True)

    def write(self, kind: str, payload) -> None:
        if not self.directory:
            return
        with self._lock:
            self._seq += 1
            seq = self._seq
        path = os.path.join(self.directory, "%04d-%s.json" % (seq, kind))
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except OSError as error:
            log("record failed: %s" % error)


_VERBOSE = False


def log(message: str) -> None:
    sys.stderr.write("[system-shim] %s\n" % message)
    sys.stderr.flush()


class ShimHandler(BaseHTTPRequestHandler):
    """Forward one request upstream, relaying the response without buffering."""

    protocol_version = "HTTP/1.1"
    server_version = "dsh-system-shim/1.0"

    # Injected by main() through the server instance.
    upstream = None
    recorder = None
    keep_system = False

    def log_message(self, fmt, *args):
        if _VERBOSE:
            log("%s - %s" % (self.address_string(), fmt % args))

    # -- request plumbing ---------------------------------------------------

    def _read_body(self) -> bytes:
        length = self.headers.get("Content-Length")
        if not length:
            return b""
        try:
            return self.rfile.read(int(length))
        except (ValueError, OSError):
            return b""

    def _forward_headers(self) -> dict:
        headers = {}
        for name, value in self.headers.items():
            lowered = name.lower()
            if lowered in HOP_BY_HOP or lowered in REPLACED:
                continue
            headers[name] = value
        # Ask for an unencoded body so chunks can be relayed without inflating
        # them first. Relaying gzip would still stream, but the client would see
        # compressed bytes it did not ask for.
        headers["Accept-Encoding"] = "identity"
        return headers

    def _upstream_target(self) -> str:
        parsed = urllib.parse.urlsplit(self.path)
        base_path = self.upstream.path  # already ends with the caller's prefix
        target = base_path + parsed.path.lstrip("/")
        if parsed.query:
            target += "?" + parsed.query
        return target

    # -- response plumbing --------------------------------------------------

    def _send_response(self, upstream_response) -> None:
        """Relay status and headers, then stream the body as chunked."""
        self.send_response_only(upstream_response.status, upstream_response.reason)
        for name, value in upstream_response.getheaders():
            lowered = name.lower()
            if lowered in HOP_BY_HOP or lowered == "content-length":
                continue
            self.send_header(name, value)
        # Always chunked: the body length is unknown while it streams, and a
        # Content-Length would force buffering the whole answer first.
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        total = 0
        try:
            while True:
                chunk = upstream_response.read1(65536)
                if not chunk:
                    break
                total += len(chunk)
                self.wfile.write(b"%x\r\n" % len(chunk))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                # Flush per chunk, not per response: this is what keeps token
                # streaming visible instead of arriving in one burst at the end.
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The client hung up mid-stream; the upstream read is abandoned.
            self.close_connection = True
            return
        finally:
            try:
                upstream_response.close()
            except Exception:
                pass

        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

        if _VERBOSE:
            log("%s %s -> %s, %d bytes relayed"
                % (self.command, self.path, upstream_response.status, total))

    def _error(self, status: int, message: str) -> None:
        body = json.dumps(
            {"type": "error", "error": {"type": "proxy_error", "message": message}},
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response_only(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    # -- verbs --------------------------------------------------------------

    def do_POST(self):
        self._proxy("POST")

    def do_GET(self):
        self._proxy("GET")

    def do_DELETE(self):
        self._proxy("DELETE")

    def _proxy(self, method: str) -> None:
        raw = self._read_body()
        body = raw
        rewrite_note = ""

        if method == "POST" and raw and wants_rewrite(self.path):
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                # Not JSON: forward untouched rather than guess.
                log("body is not JSON (%s); forwarding unchanged" % error)
            else:
                if isinstance(payload, dict):
                    self.recorder.write("raw", payload)
                new_payload, moved = rewrite_payload(payload, self.keep_system)
                if moved:
                    self.recorder.write("rewritten", new_payload)
                    body = json.dumps(new_payload, ensure_ascii=False).encode("utf-8")
                    rewrite_note = " (system -> first user message)"
                elif _VERBOSE:
                    rewrite_note = " (no system text to move)"

        headers = self._forward_headers()
        if body:
            headers["Content-Length"] = str(len(body))

        connection = None
        try:
            if self.upstream.scheme == "https":
                connection = HTTPSConnection(
                    self.upstream.hostname, self.upstream.port, timeout=600
                )
            else:
                connection = HTTPConnection(
                    self.upstream.hostname, self.upstream.port, timeout=600
                )
            connection.request(method, self._upstream_target(), body=body, headers=headers)
            response = connection.getresponse()
        except Exception as error:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
            log("upstream request failed: %s" % error)
            self._error(502, "upstream request failed: %s" % error)
            return

        try:
            self._send_response(response)
        finally:
            try:
                connection.close()
            except Exception:
                pass

        if _VERBOSE:
            log("%s %s%s" % (method, self.path, rewrite_note))


def build_server(upstream: str, listen_host: str, listen_port: int,
                 keep_system: bool, record: str | None) -> ThreadingHTTPServer:
    """Wire the handler's shared state and return an unstarted server."""
    parsed = urllib.parse.urlsplit(upstream)
    if parsed.scheme not in ("http", "https"):
        raise SystemExit("--upstream must be an http:// or https:// URL")
    if not parsed.hostname:
        raise SystemExit("--upstream is missing a host")
    if not parsed.path.endswith("/"):
        parsed = parsed._replace(path=parsed.path + "/")

    handler = type(
        "BoundShimHandler",
        (ShimHandler,),
        {
            "upstream": parsed,
            "recorder": Recorder(record),
            "keep_system": keep_system,
        },
    )
    return ThreadingHTTPServer((listen_host, listen_port), handler)


def main(argv=None) -> int:
    global _VERBOSE

    parser = argparse.ArgumentParser(
        description="Move an Anthropic system prompt into the first user message "
                    "for gateways that drop it.",
    )
    parser.add_argument("--upstream", required=True,
                        help="gateway base URL, e.g. http://host:7863/anthropic/")
    parser.add_argument("--listen", default="127.0.0.1",
                        help="listen address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8787,
                        help="listen port (default 8787)")
    parser.add_argument("--keep-system", action="store_true",
                        help="also leave the original system field in place")
    parser.add_argument("--record", metavar="DIR",
                        help="save each request body before and after rewriting")
    parser.add_argument("--verbose", action="store_true",
                        help="log every request and rewrite")
    # Kept separate from --verbose so a one-off inspection needs no server.
    parser.add_argument("--probe", action="store_true",
                        help="print how a sample body would be rewritten, then exit")
    args = parser.parse_args(argv)

    _VERBOSE = args.verbose

    if args.probe:
        sample = {
            "model": "example",
            "system": "## Rules\n\n- Answer in Chinese.",
            "messages": [{"role": "user", "content": "hi"}],
        }
        rewritten, moved = rewrite_payload(sample, args.keep_system)
        print(json.dumps({"moved": moved, "result": rewritten},
                         ensure_ascii=False, indent=2))
        return 0

    server = build_server(args.upstream, args.listen, args.port,
                          args.keep_system, args.record)
    host, port = server.server_address[:2]
    log("listening on http://%s:%d/ -> %s" % (host, port, args.upstream))
    log("streaming relay enabled; system field relocated into the first user message")
    if args.record:
        log("recording request bodies to %s" % args.record)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
