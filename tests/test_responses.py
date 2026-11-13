"""Tests for the OpenAI Responses protocol (the third dialect, alongside OpenAI and Anthropic)."""

import json

import pytest

from tokenwatt_verify.client import (Endpoint, Reply, _consume, _consume_json, _request, ask,
                                     normalize_api)
from tokenwatt_verify.conform.protocol import Protocol, norm_usage_responses


# ------------------------------------------------------------------ api names and inference
def test_api_aliases():
    assert normalize_api("responses") == "responses"
    assert normalize_api("Response") == "responses"
    assert normalize_api("openai-responses") == "responses"
    assert normalize_api("resp") == "responses"
    assert normalize_api("chat") == "openai"
    assert normalize_api("claude") == "anthropic"


def test_api_is_inferred_from_the_url():
    assert normalize_api("", "https://api.openai.com/v1") == "openai"
    assert normalize_api("", "https://api.anthropic.com/v1") == "anthropic"
    assert normalize_api("", "https://api.example.com/v1/responses") == "responses"


def test_unknown_api_is_refused():
    with pytest.raises(SystemExit):
        normalize_api("grpc", "https://api.example.com/v1")


def test_endpoint_from_env_reads_the_responses_dialect(monkeypatch):
    monkeypatch.setenv("EP_URL", "https://api.example.com/v1/")
    monkeypatch.setenv("EP_MODEL", "gpt-5")
    monkeypatch.setenv("EP_KEY", "k")
    monkeypatch.setenv("EP_API", "responses")
    ep = Endpoint.parse("env:EP")
    assert (ep.url, ep.model, ep.api) == ("https://api.example.com/v1", "gpt-5", "responses")


def test_endpoint_from_file_reads_the_responses_dialect(tmp_path):
    f = tmp_path / "ep.txt"
    f.write_text("URL=http://localhost:8000/v1\nMODEL=gpt-5\nAPI=responses\n")
    ep = Endpoint.parse(str(f))
    assert ep.api == "responses" and ep.key == ""


# ------------------------------------------------------------------ request shape
def test_responses_request_shape():
    ep = Endpoint("t", "https://api.example.com/v1", "sk-1", "gpt-5", "responses")
    req = _request(ep, "be precise", "hello", 256, 0.0)
    body = json.loads(req.data)
    assert req.full_url == "https://api.example.com/v1/responses"
    assert body["model"] == "gpt-5"
    assert body["input"] == "hello"
    assert body["instructions"] == "be precise"
    assert body["max_output_tokens"] == 256
    assert body["stream"] is True
    assert body["temperature"] == 0.0
    assert req.get_header("Authorization") == "Bearer sk-1"


def test_responses_request_omits_an_empty_system_prompt():
    ep = Endpoint("t", "https://api.example.com/v1", "", "gpt-5", "responses")
    body = json.loads(_request(ep, "", "hi", 64, None).data)
    assert "instructions" not in body
    assert "temperature" not in body


def test_responses_base_without_v1_gets_one():
    ep = Endpoint("t", "https://api.example.com", "", "gpt-5", "responses")
    assert _request(ep, "s", "u", 8, None).full_url == "https://api.example.com/v1/responses"


def test_responses_url_is_not_doubled():
    ep = Endpoint("t", "https://api.example.com/v1/responses", "", "gpt-5", "responses")
    assert _request(ep, "s", "u", 8, None).full_url == "https://api.example.com/v1/responses"


# ------------------------------------------------------------------ stream parsing
def _stream(events):
    r = Reply()
    for typ, data in events:
        piece = _consume("responses", {"type": typ, **data}, r)
        if piece:
            r.text += piece
    return r


def test_responses_stream_is_read():
    r = _stream([
        ("response.created", {"response": {"id": "resp_1", "model": "gpt-5"}}),
        ("response.output_text.delta", {"delta": "Hel"}),
        ("response.output_text.delta", {"delta": "lo"}),
        ("response.completed", {"response": {"usage": {"input_tokens": 11, "output_tokens": 3}}}),
    ])
    assert r.text == "Hello"
    assert r.input_tokens == 11 and r.output_tokens == 3
    assert r.model_claimed == "gpt-5"
    assert r.finish == "stop"


def test_responses_stream_reads_a_refusal():
    r = _stream([
        ("response.completed", {"response": {"status": "completed", "output": [
            {"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]}}),
    ])
    assert r.finish == "refusal"


def test_responses_stream_incomplete_is_a_length_cut():
    r = _stream([
        ("response.incomplete", {"response": {"incomplete_details": {"reason": "max_output_tokens"}}}),
    ])
    assert r.finish == "max_tokens"


def test_responses_stream_failure_is_reported():
    r = _stream([
        ("response.failed", {"response": {"error": {"message": "boom", "type": "server_error"}}}),
    ])
    assert "boom" in r.error


def test_responses_stream_error_event_is_reported():
    r = _stream([("error", {"message": "bad request", "code": "invalid_request"})])
    assert "bad request" in r.error


# ------------------------------------------------------------------ non-streaming body
def test_responses_json_body_is_read():
    r = Reply()
    _consume_json("responses", {
        "id": "resp_2", "object": "response", "model": "gpt-5", "status": "completed",
        "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi there"}]}],
        "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
    }, r)
    assert r.text == "hi there"
    assert (r.input_tokens, r.output_tokens) == (7, 2)
    assert r.model_claimed == "gpt-5"
    assert r.finish == "stop"


def test_responses_json_error_object_is_reported():
    r = Reply()
    _consume_json("responses", {"error": {"message": "no such model", "type": "invalid_request_error"}}, r)
    assert "no such model" in r.error


def test_openai_json_body_is_read_too():
    r = Reply()
    _consume_json("openai", {"model": "gpt-4o", "choices": [{"message": {"content": "hey"},
                                                            "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 5, "completion_tokens": 1}}, r)
    assert r.text == "hey" and r.finish == "stop" and r.input_tokens == 5


def test_anthropic_json_body_is_read_too():
    r = Reply()
    _consume_json("anthropic", {"model": "claude", "stop_reason": "end_turn",
                                 "content": [{"type": "text", "text": "hi"}],
                                 "usage": {"input_tokens": 4, "output_tokens": 2}}, r)
    assert r.text == "hi" and r.finish == "end_turn" and r.input_tokens == 4


# ------------------------------------------------------------------ conform: protocol
def test_protocol_path_and_request_body():
    p = Protocol("responses", "gpt-5")
    assert p.path == "/responses"
    body = p.request([p.user("hi")], system="be brief", max_tokens=32, stream=True, temperature=0.0)
    assert body["input"] == [{"role": "user", "content": "hi"}]
    assert body["instructions"] == "be brief"
    assert body["max_output_tokens"] == 32
    assert body["stream"] is True
    assert "max_tokens" not in body and "messages" not in body


def test_protocol_tool_shapes_differ_per_dialect():
    assert Protocol("responses", "m").tool("f", "d", {}) == {"type": "function", "name": "f",
                                                            "description": "d", "parameters": {}}
    assert Protocol("openai", "m").tool("f", "d", {})["function"]["name"] == "f"
    assert Protocol("anthropic", "m").tool("f", "d", {})["input_schema"] == {}
    assert Protocol("responses", "m").tool_choice("auto") == "auto"
    assert Protocol("responses", "m").tool_choice("forced", "f") == {"type": "function", "name": "f"}


def test_protocol_rejects_an_unknown_dialect():
    with pytest.raises(ValueError):
        Protocol("grpc", "m")


def test_protocol_parses_a_responses_body():
    p = Protocol("responses", "gpt-5")
    res = p.parse(_resp({
        "id": "resp_3", "object": "response", "model": "gpt-5", "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}],
        "usage": {"input_tokens": 9, "output_tokens": 2}}))
    assert res.ok and res.text == "answer"
    assert res.usage["prompt_total"] == 9 and res.usage["output"] == 2
    assert res.shape_problems == []


def test_protocol_parses_a_responses_stream():
    p = Protocol("responses", "gpt-5")
    res = p.parse(_resp(None, events=[
        {"t": 0.1, "event": None, "data": {"type": "response.created",
                                           "response": {"id": "resp_4", "model": "gpt-5"}}},
        {"t": 0.2, "event": None, "data": {"type": "response.output_text.delta", "delta": "par"}},
        {"t": 0.3, "event": None, "data": {"type": "response.output_text.delta", "delta": "tial"}},
        {"t": 0.4, "event": None, "data": {"type": "response.completed",
                                           "response": {"usage": {"input_tokens": 12, "output_tokens": 3}}}},
    ]))
    assert res.ok and res.text == "partial"
    assert res.final_event and res.usage_in_stream
    assert res.usage["prompt_total"] == 12
    assert res.token_events == 2


def test_protocol_assembles_a_streamed_tool_call():
    p = Protocol("responses", "gpt-5")
    res = p.parse(_resp(None, events=[
        {"t": 0.1, "event": None, "data": {"type": "response.output_item.added", "output_index": 0,
                                           "item": {"type": "function_call", "call_id": "call_1", "name": "get_weather"}}},
        {"t": 0.2, "event": None, "data": {"type": "response.function_call_arguments.delta",
                                           "output_index": 0, "delta": '{"city":'}},
        {"t": 0.3, "event": None, "data": {"type": "response.function_call_arguments.delta",
                                           "output_index": 0, "delta": '"Lisbon"}'}},
        {"t": 0.4, "event": None, "data": {"type": "response.completed",
                                           "response": {"usage": {"input_tokens": 5, "output_tokens": 1}}}},
    ]))
    assert res.tool_calls and res.tool_calls[0]["name"] == "get_weather"
    assert res.tool_calls[0]["input"] == {"city": "Lisbon"}
    assert res.tool_calls[0]["id"] == "call_1"


def test_usage_normalisation_separates_cached_tokens():
    u = norm_usage_responses({"input_tokens": 100, "output_tokens": 10,
                              "input_tokens_details": {"cached_tokens": 40}})
    assert u["prompt_total"] == 100 and u["cache_read"] == 40 and u["input_uncached"] == 60


def _resp(body, events=None):
    from tokenwatt_verify.conform.transport import Response
    r = Response(status=200, json=body, events=events or [])
    return r


# ------------------------------------------------------------------ a local Responses server
class _Handler(__import__("http.server", fromlist=["BaseHTTPRequestHandler"]).BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        inp = body.get("input")
        text = "ANSWER: 29830 − 1"          # a non-ASCII character, as a real model often writes
        payload = {
            "id": "resp_x", "object": "response", "model": body.get("model", "gpt-5"), "status": "completed",
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 4, "output_tokens": 3},
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def test_result_files_survive_a_non_ascii_answer(tmp_path):
    """A Windows console defaults to a legacy code page; the results must not be written with it.

    Before the fix the JSONL was written with the locale codec and a '−' in the answer crashed
    the tool with UnicodeEncodeError half-way through a run."""
    import threading
    from http.server import ThreadingHTTPServer

    from tokenwatt_verify.cli import run

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    try:
        class A:
            target = f"env:TSV"
            reference = f"env:TSV"
            probes = None
            kinds = None
            out = str(tmp_path)
            rounds = 1
            every = None
            hours = 24
            temperature = 0.0
            json = None
        import os
        os.environ["TSV_URL"] = url
        os.environ["TSV_MODEL"] = "gpt-5"
        os.environ["TSV_API"] = "responses"
        try:
            run(A())
        finally:
            for k in ("TSV_URL", "TSV_MODEL", "TSV_API"):
                os.environ.pop(k, None)
    finally:
        srv.shutdown()
    files = list(tmp_path.glob("*.jsonl"))
    assert files, "no result file was written"
    body = files[0].read_text(encoding="utf-8")
    assert "−" in body, "the non-ASCII answer was lost"
