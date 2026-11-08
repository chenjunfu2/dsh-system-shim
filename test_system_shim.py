"""Tests for the pure rewrite functions.

Run: python -m unittest discover -s . -p "test_*.py"
"""

import json
import unittest

from importlib import import_module

shim = import_module("system_shim")


class SystemToTextTests(unittest.TestCase):
    def test_string_passes_through(self):
        self.assertEqual(shim.system_to_text("hello"), "hello")

    def test_none_and_missing(self):
        self.assertEqual(shim.system_to_text(None), "")
        self.assertEqual(shim.system_to_text(undefined := object()), "")

    def test_block_list_joins_text_blocks(self):
        system = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        self.assertEqual(shim.system_to_text(system), "first\n\nsecond")

    def test_non_text_blocks_are_dropped(self):
        system = [
            {"type": "text", "text": "keep"},
            {"type": "image", "source": {"type": "base64", "data": "..."}},
            {"type": "text", "text": "   "},
        ]
        self.assertEqual(shim.system_to_text(system), "keep")

    def test_cache_control_is_ignored(self):
        system = [{"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}}]
        self.assertEqual(shim.system_to_text(system), "cached")


class RewriteTests(unittest.TestCase):
    def test_moves_string_system_into_head_of_first_user(self):
        payload = {
            "model": "m",
            "system": "RULES",
            "messages": [{"role": "user", "content": "hi"}],
        }
        out, moved = shim.rewrite_payload(payload)
        self.assertTrue(moved)
        self.assertNotIn("system", out)
        blocks = out["messages"][0]["content"]
        self.assertEqual(blocks[0], {"type": "text", "text": "RULES"})
        self.assertEqual(blocks[1], {"type": "text", "text": "hi"})

    def test_upgrades_string_content_to_blocks(self):
        payload = {
            "system": "RULES",
            "messages": [{"role": "user", "content": "hi"}],
        }
        out, _ = shim.rewrite_payload(payload)
        self.assertIsInstance(out["messages"][0]["content"], list)

    def test_preserves_existing_block_list_order(self):
        original = [
            {"type": "text", "text": "question"},
            {"type": "text", "text": "more"},
        ]
        payload = {"system": "RULES", "messages": [{"role": "user", "content": original}]}
        out, _ = shim.rewrite_payload(payload)
        blocks = out["messages"][0]["content"]
        self.assertEqual(blocks[0]["text"], "RULES")
        self.assertEqual(blocks[1]["text"], "question")
        self.assertEqual(blocks[2]["text"], "more")
        # The caller's list must not be mutated in place.
        self.assertEqual(len(original), 2)

    def test_inserts_user_turn_when_first_is_assistant(self):
        payload = {
            "system": "RULES",
            "messages": [{"role": "assistant", "content": "prior"}],
        }
        out, _ = shim.rewrite_payload(payload)
        self.assertEqual(out["messages"][0]["role"], "user")
        self.assertEqual(out["messages"][0]["content"][0]["text"], "RULES")
        self.assertEqual(out["messages"][1]["role"], "assistant")

    def test_creates_message_when_messages_missing(self):
        out, moved = shim.rewrite_payload({"system": "RULES"})
        self.assertTrue(moved)
        self.assertEqual(out["messages"][0]["role"], "user")
        self.assertEqual(out["messages"][0]["content"][0]["text"], "RULES")

    def test_empty_system_leaves_payload_untouched(self):
        payload = {"system": "   ", "messages": [{"role": "user", "content": "hi"}]}
        out, moved = shim.rewrite_payload(payload)
        self.assertFalse(moved)
        self.assertIs(out, payload)
        # system must still be there: nothing was relocated, so nothing is removed.
        self.assertIn("system", out)

    def test_absent_system_leaves_payload_untouched(self):
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        out, moved = shim.rewrite_payload(payload)
        self.assertFalse(moved)
        self.assertIs(out, payload)

    def test_keep_system_retains_original_field(self):
        payload = {"system": "RULES", "messages": [{"role": "user", "content": "hi"}]}
        out, _ = shim.rewrite_payload(payload, keep_system=True)
        self.assertEqual(out["system"], "RULES")
        self.assertEqual(out["messages"][0]["content"][0]["text"], "RULES")

    def test_other_top_level_fields_survive(self):
        payload = {
            "model": "m",
            "system": "RULES",
            "system_probe": None,
            "max_tokens": 1024,
            "temperature": 0.3,
            "stream": True,
            "tools": [{"name": "t", "input_schema": {"type": "object"}}],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "messages": [{"role": "user", "content": "hi"}],
        }
        out, _ = shim.rewrite_payload(payload)
        for key in ("model", "max_tokens", "temperature", "stream", "tools", "thinking"):
            self.assertEqual(out[key], payload[key], key)
        self.assertEqual(out["system_probe"], None)

    def test_second_and_later_messages_untouched(self):
        payload = {
            "system": "RULES",
            "messages": [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ],
        }
        out, _ = shim.rewrite_payload(payload)
        self.assertEqual(out["messages"][1], payload["messages"][1])
        self.assertEqual(out["messages"][2], payload["messages"][2])

    def test_non_dict_payload_is_returned_unchanged(self):
        for value in ([], "text", 42, None):
            out, moved = shim.rewrite_payload(value)
            self.assertFalse(moved)
            self.assertEqual(out, value)

    def test_block_list_system_is_flattened_into_one_block(self):
        payload = {
            "system": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
            "messages": [{"role": "user", "content": "hi"}],
        }
        out, _ = shim.rewrite_payload(payload)
        head = out["messages"][0]["content"][0]
        self.assertEqual(head["text"], "A\n\nB")
        # One relocated block, not two: the head must stay a single prefix.
        self.assertEqual(len(out["messages"][0]["content"]), 2)


class PathTests(unittest.TestCase):
    def test_rewrites_messages_paths(self):
        self.assertTrue(shim.wants_rewrite("/v1/messages"))
        self.assertTrue(shim.wants_rewrite("/anthropic/v1/messages"))
        self.assertTrue(shim.wants_rewrite("/v1/messages?beta=true"))
        self.assertTrue(shim.wants_rewrite("/v1/messages/count_tokens"))

    def test_ignores_unrelated_paths(self):
        self.assertFalse(shim.wants_rewrite("/v1/models"))
        self.assertFalse(shim.wants_rewrite("/health"))


if __name__ == "__main__":
    unittest.main()
