"""Integration tests: run a mock upstream and the real proxy in one process.

These cover what unit tests cannot: the wire behaviour. Two properties matter
and are easy to get wrong in a proxy:

1. The rewrite actually reaches the upstream, in the message body rather than in
   a `system` field.
2. The response is relayed incrementally. A proxy that returns the right bytes
   after buffering the whole answer looks correct in a diff and is unusable in
   the UI, so the assertion is on arrival *timing*, not on content.

Standard library only, same as the proxy. Servers run in threads so no
subprocess is spawned and nothing can be left running after the suite exits.

Run: python -m unittest test_integration_system_shim
"""

import http.client
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from importlib import import_module

shim = import_module("system_shim")


class MockUpstream:
    """An upstream that records what it received and streams SSE on demand.

    ``chunk_delay`` is the whole point: it makes buffering observable. If the
    proxy holds the body until the end, the client sees all chunks at
    ``len(chunks) * chunk_delay``; if it relays, the client sees them spread
    across that interval.
    """

    def __init__(self, chunks=5, chunk_delay=0.12):
        self.chunks = chunks
        self.chunk_delay = chunk_delay
        self.received = []
        self.received_headers = []
        self.paths = []
        self._server = None
        self._thread = None

    def __enter__(self):
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw.decode("utf-8"))
                except Exception:
                    body = {"__unparsed__": raw.decode("utf-8", "replace")}
                upstream.received.append(body)
                upstream.received_headers.append(dict(self.headers))
                upstream.paths.append(self.path)

                self.send_response_only(200, "OK")
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                try:
                    for index in range(upstream.chunks):
                        event = "event: content_block_delta\ndata: {\"i\":%d}\n\n" % index
                        payload = event.encode("utf-8")
                        self.wfile.write(b"%x\r\n" % len(payload))
                        self.wfile.write(payload)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        # A deliberate gap between events. The mock never
                        # introduces one in the response the client sees unless
                        # the proxy actually forwards as it reads.
                        if index < upstream.chunks - 1:
                            time.sleep(upstream.chunk_delay)
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                upstream.paths.append(self.path)
                body = b'{"data":[]}'
                self.send_response_only(200, "OK")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.port = self._server.server_address[1]
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class ProxyFixture:
    """Start the proxy against an upstream and guarantee shutdown."""

    def __init__(self, upstream_port, keep_system=False):
        self.upstream = "http://127.0.0.1:%d/anthropic/" % upstream_port
        self.keep_system = keep_system
        self._server = None
        self._thread = None

    def __enter__(self):
        self._server = shim.build_server(
            self.upstream, "127.0.0.1", 0, self.keep_system, None
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.port = self._server.server_address[1]
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def post_streaming(port, path, payload, timeout=30):
    """POST a JSON body and return (status, [(elapsed, line), ...]).

    Reading line by line records when each line arrived, which is how buffering
    is detected. ``http.client`` de-chunks transparently.
    """
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = json.dumps(payload).encode("utf-8")
    started = time.monotonic()
    connection.request(
        "POST", path, body=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )
    response = connection.getresponse()
    arrivals = []
    while True:
        line = response.readline()
        if not line:
            break
        arrivals.append((time.monotonic() - started, line))
    status = response.status
    connection.close()
    return status, arrivals


class ForwardingTests(unittest.TestCase):
    def test_system_is_moved_into_first_user_message(self):
        with MockUpstream() as upstream, ProxyFixture(upstream.port) as proxy:
            payload = {
                "model": "test-model",
                "system": "## 语言规则\n\n- 思考用中文。",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            }
            status, _ = post_streaming(proxy.port, "/v1/messages", payload)

        self.assertEqual(status, 200)
        self.assertEqual(len(upstream.received), 1)
        seen = upstream.received[0]

        # The privileged field must be gone: leaving it would duplicate the
        # instructions on a gateway that does honour it.
        self.assertNotIn("system", seen)

        blocks = seen["messages"][0]["content"]
        self.assertEqual(blocks[0]["type"], "text")
        self.assertIn("语言规则", blocks[0]["text"])
        self.assertEqual(blocks[1]["text"], "hello")

        # Untouched fields must survive verbatim.
        self.assertEqual(seen["model"], "test-model")
        self.assertEqual(seen["max_tokens"], 64)
        self.assertTrue(seen["stream"])

    def test_upstream_path_receives_the_prefix_and_suffix(self):
        with MockUpstream() as upstream, ProxyFixture(upstream.port) as proxy:
            post_streaming(proxy.port, "/v1/messages", {"messages": []})

        self.assertEqual(upstream.paths[0], "/anthropic/v1/messages")

    def test_query_string_is_preserved(self):
        with MockUpstream() as upstream, ProxyFixture(upstream.port) as proxy:
            post_streaming(proxy.port, "/v1/messages?beta=true", {"messages": []})

        self.assertEqual(upstream.paths[0], "/anthropic/v1/messages?beta=true")

    def test_api_key_header_is_forwarded(self):
        with MockUpstream() as upstream, ProxyFixture(upstream.port) as proxy:
            connection = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=20)
            body = json.dumps({"system": "s", "messages": []}).encode()
            connection.request(
                "POST", "/v1/messages", body=body,
                headers={"Content-Type": "application/json",
                         "Content-Length": str(len(body)),
                         "x-api-key": "secret-key-123",
                         "anthropic-version": "2023-06-01"},
            )
            connection.getresponse().read()
            connection.close()

        headers = {k.lower(): v for k, v in upstream.received_headers[0].items()}
        self.assertEqual(headers.get("x-api-key"), "secret-key-123")
        self.assertEqual(headers.get("anthropic-version"), "2023-06-01")

    def test_keep_system_retains_the_field_upstream(self):
        with MockUpstream() as upstream, ProxyFixture(upstream.port, keep_system=True) as proxy:
            post_streaming(proxy.port, "/v1/messages",
                           {"system": "RULES", "messages": [{"role": "user", "content": "x"}]})

        seen = upstream.received[0]
        self.assertEqual(seen["system"], "RULES")
        self.assertEqual(seen["messages"][0]["content"][0]["text"], "RULES")

    def test_non_messages_path_is_not_rewritten(self):
        with MockUpstream() as upstream, ProxyFixture(upstream.port) as proxy:
            post_streaming(proxy.port, "/v1/other", {"system": "RULES", "messages": []})

        # The field survives untouched on paths outside the rewrite scope.
        self.assertEqual(upstream.received[0]["system"], "RULES")

    def test_request_without_system_is_forwarded_unchanged(self):
        with MockUpstream() as upstream, ProxyFixture(upstream.port) as proxy:
            payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            post_streaming(proxy.port, "/v1/messages", payload)

        self.assertEqual(upstream.received[0], payload)


class StreamingTests(unittest.TestCase):
    def test_response_is_relayed_incrementally_not_buffered(self):
        """The decisive test: chunk arrival must be spread over time.

        A buffering proxy produces one burst at the very end. The assertion is
        on the gap between the first and last arrival, not on total content.
        """
        chunks, delay = 5, 0.12
        with MockUpstream(chunks=chunks, chunk_delay=delay) as upstream, \
                ProxyFixture(upstream.port) as proxy:
            status, arrivals = post_streaming(
                proxy.port, "/v1/messages",
                {"system": "RULES", "messages": [{"role": "user", "content": "x"}]},
            )

        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(arrivals), chunks)

        first = arrivals[0][0]
        last = arrivals[-1][0]
        spread = last - first
        expected = delay * (chunks - 1)

        # Buffering would give spread ~= 0 (everything at the end). Requiring
        # most of the intended interval tolerates scheduler jitter while still
        # failing loudly on the bug that matters.
        self.assertGreater(
            spread, expected * 0.5,
            "chunks arrived within %.3fs; expected ~%.3fs of spread between "
            "first and last, which means the proxy buffered the stream"
            % (spread, expected),
        )

    def test_first_chunk_arrives_well_before_the_last(self):
        """Streaming means the client can render before the answer completes."""
        chunks, delay = 4, 0.15
        with MockUpstream(chunks=chunks, chunk_delay=delay) as upstream, \
                ProxyFixture(upstream.port) as proxy:
            _, arrivals = post_streaming(
                proxy.port, "/v1/messages",
                {"system": "RULES", "messages": [{"role": "user", "content": "x"}]},
            )

        first_arrival = arrivals[0][0]
        total = delay * (chunks - 1)
        # The first event should land near the start, not after the whole stream.
        self.assertLess(
            first_arrival, total * 0.75,
            "first chunk arrived at %.3fs of a ~%.3fs stream; the proxy held it"
            % (first_arrival, total),
        )

    def test_all_event_payloads_survive_the_relay(self):
        with MockUpstream(chunks=6, chunk_delay=0.02) as upstream, \
                ProxyFixture(upstream.port) as proxy:
            _, arrivals = post_streaming(
                proxy.port, "/v1/messages",
                {"system": "RULES", "messages": [{"role": "user", "content": "x"}]},
            )

        text = b"".join(line for _, line in arrivals).decode("utf-8", "replace")
        for index in range(6):
            self.assertIn('"i":%d' % index, text)

    def test_multibyte_utf8_is_not_split_corrupt(self):
        """Chinese text in a delta must survive chunk boundaries intact.

        The relay writes raw byte chunks, so a multi-byte character can straddle
        two writes; the client's decoder must still see valid text. This checks
        the proxy adds no mangling of its own.
        """
        with MockUpstream(chunks=3, chunk_delay=0.02) as upstream, \
                ProxyFixture(upstream.port) as proxy:
            _, arrivals = post_streaming(
                proxy.port, "/v1/messages",
                {"system": "中文系统提示", "messages": [{"role": "user", "content": "你好"}]},
            )

        text = b"".join(line for _, line in arrivals).decode("utf-8", "replace")
        self.assertIn("content_block_delta", text)
        # And the upstream received the Chinese system text intact.
        self.assertIn("中文系统提示", upstream.received[0]["messages"][0]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
