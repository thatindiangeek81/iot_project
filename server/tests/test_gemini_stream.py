import time
import unittest
from unittest import mock

import _path  # noqa: F401
import requests

import config
import gemini
from fakes import FakeResponse, FakeSession, delta, sse

MSG = [{"role": "user", "content": "hi"}]


def call_chunk(**call):
    return {"choices": [{"delta": {"tool_calls": [call]}}]}


class SseParsing(unittest.TestCase):
    def test_events_survive_being_cut_anywhere(self):
        raw = sse(delta("naïve café 🙂 ok"), delta(" more"), {"choices": [{"delta": {}, "finish_reason": "stop"}]})
        want = list(gemini._iter_sse([raw]))
        self.assertEqual(len(want), 3)
        for cut in range(1, len(raw)):
            self.assertEqual(list(gemini._iter_sse([raw[:cut], raw[cut:]])), want, f"cut at {cut}")

    def test_comments_crlf_and_junk_are_ignored(self):
        raw = (b": keep-alive\r\n\r\nevent: message\r\ndata: {\"a\": 1}\r\n\r\ndata: not json\r\n\r\n"
               b"data: {\"b\": 2}\r\n\r\ndata: [DONE]\r\n\r\ndata: {\"never\": 3}\r\n")
        self.assertEqual(list(gemini._iter_sse([raw])), [{"a": 1}, {"b": 2}])

    def test_last_event_without_a_final_newline(self):
        self.assertEqual(list(gemini._iter_sse([b'data: {"a": 1}'])), [{"a": 1}])


class Assembling(unittest.TestCase):
    def build(self, *chunks):
        acc = gemini._Acc()
        text = [acc.add(c) for c in chunks]
        return acc.message(), text

    def test_text(self):
        msg, text = self.build(delta("Hello"), delta(" there."))
        self.assertEqual(msg, {"role": "assistant", "content": "Hello there."})
        self.assertEqual(text, ["Hello", " there."])

    def test_a_tool_call_in_pieces_keeps_gemini_signature(self):
        msg, _ = self.build(
            call_chunk(index=0, id="c1", type="function", function={"name": "get_weather", "arguments": '{"pla'},
                       extra_content={"google": {"thought_signature": "SIG=="}}),
            call_chunk(index=0, function={"arguments": 'ce": "Rajkot"}'}))
        call = msg["tool_calls"][0]
        self.assertEqual(call["function"], {"name": "get_weather", "arguments": '{"place": "Rajkot"}'})
        self.assertEqual(call["id"], "c1")
        self.assertEqual(call["extra_content"], {"google": {"thought_signature": "SIG=="}})

    def test_parallel_calls_all_numbered_zero_are_still_two_calls(self):
        msg, _ = self.build(
            call_chunk(index=0, id="a", function={"name": "get_weather", "arguments": '{"place": "A"}'}),
            call_chunk(index=0, id="b", function={"name": "web_search", "arguments": '{"query": "x"}'}))
        self.assertEqual([c["function"]["name"] for c in msg["tool_calls"]], ["get_weather", "web_search"])

    def test_whole_calls_without_index_or_id(self):
        msg, _ = self.build(call_chunk(function={"name": "a", "arguments": "{}"}),
                            call_chunk(function={"name": "b", "arguments": '{"x": 1}'}))
        self.assertEqual([c["function"]["name"] for c in msg["tool_calls"]], ["a", "b"])

    def test_call_without_arguments_gets_an_empty_object(self):
        msg, _ = self.build(call_chunk(index=0, id="c", function={"name": "get_weather"}))
        self.assertEqual(msg["tool_calls"][0]["function"]["arguments"], "{}")

    def test_name_repeated_in_a_continuation_does_not_split_the_call(self):
        msg, _ = self.build(call_chunk(index=0, id="c", function={"name": "w", "arguments": '{"pl'}),
                            call_chunk(index=0, id="c", function={"name": "w", "arguments": 'ace": "x"}'}))
        self.assertEqual(len(msg["tool_calls"]), 1)
        self.assertEqual(msg["tool_calls"][0]["function"]["arguments"], '{"place": "x"}')

    def test_an_error_event_raises(self):
        with self.assertRaises(gemini.GeminiError):
            gemini._Acc().add({"error": {"message": "quota"}})


class ChatStream(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, gemini, "_reasoning_ok", gemini._reasoning_ok)
        for name, value in (("GEMINI_HEDGE_S", 3.0), ("GEMINI_TIMEOUT", 5.0), ("GEMINI_API_KEY", "k"),
                            ("GEMINI_BASE", "http://gemini.test/v1beta")):
            p = mock.patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)

    def run_stream(self, first, fresh=None, **kw):
        """Run chat_stream with `first` as the session in use and `fresh` as the one a retry opens."""
        info = {}
        p1 = mock.patch.object(gemini, "_session", first)
        p2 = mock.patch.object(gemini, "_new_session", lambda: fresh)
        with p1, p2:
            events = list(gemini.chat_stream(MSG, info=info, **kw))
        return events, info

    def test_streams_words_then_the_whole_message(self):
        session = FakeSession(FakeResponse([sse(delta("Hi"), delta(" there."))]))
        events, info = self.run_stream(session)
        self.assertEqual(events, [("text", "Hi"), ("text", " there."),
                                  ("end", {"role": "assistant", "content": "Hi there."})])
        self.assertFalse(info["hedged"])
        self.assertIn("ttft_ms", info)
        sent = session.calls[0]
        self.assertTrue(sent["json"]["stream"])
        self.assertTrue(sent["url"].endswith("/openai/chat/completions"))

    def test_hedge_asks_again_when_nothing_arrives_and_keeps_the_faster_connection(self):
        stuck_resp = FakeResponse(hang=True)
        stuck = FakeSession(stuck_resp)
        fast = FakeSession(FakeResponse([sse(delta("Hello."))]))
        events, info = self.run_stream(stuck, fast, hedge_s=0.05)
        self.assertEqual(events[-1], ("end", {"role": "assistant", "content": "Hello."}))
        self.assertTrue(info["hedged"])
        self.assertTrue(stuck_resp.closed.wait(2), "the stuck request should have been cancelled")
        self.assertEqual(len(fast.calls), 1)

    def test_the_fresh_connection_replaces_the_stale_one_for_next_time(self):
        stuck, fast = FakeSession(FakeResponse(hang=True)), FakeSession(FakeResponse([sse(delta("Hello."))]))
        with mock.patch.object(gemini, "_session", stuck), mock.patch.object(gemini, "_new_session", lambda: fast):
            list(gemini.chat_stream(MSG, hedge_s=0.05))
            self.assertIs(gemini._session, fast)
        for _ in range(40):
            if stuck.closed:
                break
            time.sleep(0.05)
        self.assertTrue(stuck.closed, "the stale session should be closed")

    def test_a_failed_connection_is_retried_at_once(self):
        broken = FakeSession(requests.ConnectionError("reset"))
        fast = FakeSession(FakeResponse([sse(delta("Back."))]))
        events, info = self.run_stream(broken, fast, hedge_s=30)
        self.assertEqual(events[-1][1]["content"], "Back.")
        self.assertTrue(info["hedged"])

    def test_a_bad_request_is_not_retried(self):
        bad = FakeSession(FakeResponse(status=400, text='{"error": "bad"}'))
        fresh = FakeSession()
        with self.assertRaises(gemini.GeminiError) as ctx:
            self.run_stream(bad, fresh)
        self.assertIn("400", str(ctx.exception))
        self.assertEqual(fresh.calls, [])

    def test_a_rate_limit_is_not_retried(self):
        limited = FakeSession(FakeResponse(status=429))
        fresh = FakeSession()
        with self.assertRaises(gemini.GeminiError) as ctx:
            self.run_stream(limited, fresh)
        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(fresh.calls, [])

    def test_a_model_that_refuses_reasoning_effort_is_asked_again_without_it(self):
        session = FakeSession(FakeResponse(status=400, text="Unknown name reasoning_effort"),
                              FakeResponse([sse(delta("Fine."))]))
        events, _ = self.run_stream(session, reasoning="low")
        self.assertEqual(events[-1][1]["content"], "Fine.")
        self.assertIn("reasoning_effort", session.calls[0]["json"])
        self.assertNotIn("reasoning_effort", session.calls[1]["json"])
        self.assertFalse(gemini._reasoning_ok)

    def test_a_server_that_ignores_stream_still_works(self):
        reply = {"choices": [{"message": {"role": "assistant", "content": "Plain.",
                                          "tool_calls": [{"id": "x", "type": "function",
                                                          "function": {"name": "get_weather", "arguments": "{}"}}]}}]}
        session = FakeSession(FakeResponse(ctype="application/json", json_data=reply))
        events, _ = self.run_stream(session)
        self.assertEqual(events[0], ("text", "Plain."))
        self.assertEqual(events[-1][1]["tool_calls"][0]["function"]["name"], "get_weather")

    def test_an_empty_stream_is_an_error(self):
        session = FakeSession(FakeResponse([sse(delta(role="assistant"))]))
        with self.assertRaises(gemini.GeminiError):
            self.run_stream(session)

    def test_closing_early_cancels_the_request(self):
        resp = FakeResponse([sse(delta("One."), delta(" Two."), delta(" Three."))], gap=0.02)
        with mock.patch.object(gemini, "_session", FakeSession(resp)):
            gen = gemini.chat_stream(MSG)
            self.assertEqual(next(gen), ("text", "One."))
            gen.close()
        self.assertTrue(resp.closed.wait(2))

    def test_a_stream_that_stalls_after_starting_raises(self):
        class Stalls(FakeResponse):
            def iter_content(self, chunk_size=None):
                yield sse(delta("Start"))[:-len(b"data: [DONE]\n\n")]
                raise requests.ConnectionError("dropped")

        with self.assertRaises(gemini.GeminiError):
            self.run_stream(FakeSession(Stalls()))


if __name__ == "__main__":
    unittest.main()
