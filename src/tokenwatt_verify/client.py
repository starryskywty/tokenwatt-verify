"""Minimal streaming client for OpenAI-compatible, Anthropic Messages and OpenAI Responses endpoints, with timing.

Standard library only. Every call returns the text, the usage the endpoint reported, the model name it
claimed, and time to first token / generation speed, which are themselves fingerprints.

Three wire dialects are spoken:

  openai      POST {base}/chat/completions   — messages, usage.prompt_tokens / completion_tokens
  anthropic   POST {base}/v1/messages        — content blocks, usage.input_tokens / output_tokens
  responses   POST {base}/v1/responses       — input/instructions, usage.input_tokens / output_tokens,
                                               SSE events response.output_text.delta / response.completed
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path


# The three dialects, plus the names people actually type for them.
API_ALIASES = {
    "openai": "openai", "openai-chat": "openai", "chat": "openai", "chat_completions": "openai",
    "anthropic": "anthropic", "messages": "anthropic", "claude": "anthropic",
    "responses": "responses", "response": "responses", "resp": "responses",
    "openai-responses": "responses", "openai_responses": "responses",
}


def normalize_api(name: str | None, url: str = "") -> str:
    """Map a user-supplied API name to a dialect; infer it from the URL when it isn't given."""
    n = (name or "").strip().lower()
    if n:
        if n not in API_ALIASES:
            raise SystemExit(f"unknown API {name!r}: use openai, anthropic or responses")
        return API_ALIASES[n]
    u = (url or "").lower().rstrip("/")
    if "anthropic" in u:
        return "anthropic"
    if u.endswith("/responses"):
        return "responses"
    return "openai"


@dataclass
class Endpoint:
    name: str
    url: str            # base URL, e.g. https://openrouter.ai/api/v1 or https://host/v1 (anthropic: base without /messages)
    key: str = field(repr=False)
    model: str
    api: str = "openai"  # openai | anthropic | responses

    @classmethod
    def from_file(cls, name: str, path: Path) -> "Endpoint":
        """Read `URL: …`, `KEY: …`, `MODEL: …`, optional `API: openai|anthropic|responses` lines."""
        c = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(\w+)\s*[:=]\s*(.*)", line)
            if m:
                c[m.group(1).upper()] = m.group(2).strip()
        url = c["URL"].rstrip("/")
        api = normalize_api(c.get("API", ""), url)
        return cls(name, url, c.get("KEY", ""), c["MODEL"], api)   # KEY optional: local servers

    @classmethod
    def from_env(cls, prefix: str) -> "Endpoint":
        """Read <PREFIX>_URL, <PREFIX>_MODEL and optional <PREFIX>_KEY, <PREFIX>_API from the environment.

        The key may be left unset for a local server without authentication (vLLM, SGLang, Ollama, llama.cpp).
        <PREFIX>_API takes openai, anthropic or responses (aliases accepted); without it the dialect is
        inferred from the URL."""
        import os
        p = prefix.upper()
        try:
            url, model = os.environ[f"{p}_URL"], os.environ[f"{p}_MODEL"]
        except KeyError as e:
            raise SystemExit(f"missing environment variable {e.args[0]}") from None
        key = os.environ.get(f"{p}_KEY", "")
        url = url.rstrip("/")
        api = normalize_api(os.environ.get(f"{p}_API", ""), url)
        return cls(prefix.lower(), url, key, model, api)

    @classmethod
    def openrouter(cls, model: str) -> "Endpoint":
        """A reference through OpenRouter (key in OPENROUTER_API_KEY), e.g. anthropic/claude-sonnet-4.6."""
        import os
        key = os.environ.get("OPENROUTER_API_KEY") or ""
        if not key:
            raise SystemExit("set OPENROUTER_API_KEY to use an openrouter: reference")
        return cls(f"openrouter:{model}", "https://openrouter.ai/api/v1", key, model, "openai")

    @classmethod
    def parse(cls, spec: str) -> "Endpoint":
        """`openrouter:<model>`, `env:<PREFIX>`, or a path to a file with URL/KEY/MODEL/API lines."""
        if spec.startswith("openrouter:"):
            return cls.openrouter(spec.split(":", 1)[1])
        if spec.startswith("env:"):
            return cls.from_env(spec.split(":", 1)[1])
        p = Path(spec).expanduser()
        if not p.is_file():
            raise SystemExit(f"no endpoint file {p} (use openrouter:<model>, env:<PREFIX> or a file)")
        return cls.from_file(p.stem, p)


@dataclass
class Reply:
    text: str = ""
    model_claimed: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_s: float | None = None
    total_s: float = 0.0
    chunks: int = 0
    error: str = ""
    finish: str = ""          # stop / end_turn / max_tokens / content_filter / refusal …

    @property
    def gen_tok_s(self) -> float | None:
        if self.output_tokens and self.ttft_s is not None and self.total_s > self.ttft_s:
            return self.output_tokens / (self.total_s - self.ttft_s)
        return None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["gen_tok_s"] = self.gen_tok_s
        return d


def _request(ep: Endpoint, system: str, user: str, max_tokens: int, temperature: float | None):
    if ep.api == "anthropic":
        url = ep.url + ("/messages" if ep.url.endswith("/v1") else "/v1/messages")
        body = {"model": ep.model, "max_tokens": max_tokens, "stream": True,
                "system": system, "messages": [{"role": "user", "content": user}]}
        headers = {"anthropic-version": "2023-06-01"}
        if ep.key:
            headers.update({"x-api-key": ep.key, "Authorization": f"Bearer {ep.key}"})
    elif ep.api == "responses":
        url = ep.url if ep.url.endswith("/responses") else \
            ep.url + ("/responses" if ep.url.endswith("/v1") else "/v1/responses")
        body = {"model": ep.model, "input": user, "max_output_tokens": max_tokens, "stream": True}
        if system:
            body["instructions"] = system
        headers = {"Authorization": f"Bearer {ep.key}"} if ep.key else {}
    else:
        url = ep.url + "/chat/completions"
        body = {"model": ep.model, "max_tokens": max_tokens, "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        headers = {"Authorization": f"Bearer {ep.key}"} if ep.key else {}
    if temperature is not None:
        body["temperature"] = temperature
    headers.update({"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 (compatible; tokenwatt-verify)"})
    return urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")


def _ask_once(ep: Endpoint, system: str, user: str, max_tokens: int, temperature: float | None,
              timeout: float) -> Reply:
    r = Reply()
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(_request(ep, system, user, max_tokens, temperature), timeout=timeout) as resp:
            if "text/event-stream" in (resp.headers.get("Content-Type") or ""):
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        ev = json.loads(data)
                    except ValueError:
                        continue
                    piece = _consume(ep.api, ev, r)
                    if piece:
                        if r.ttft_s is None:
                            r.ttft_s = time.perf_counter() - t0
                        r.text += piece
                        r.chunks += 1
            else:
                # A server that ignored stream=true answers with one JSON body.
                raw = resp.read().decode("utf-8", "replace")
                try:
                    j = json.loads(raw)
                except ValueError:
                    r.error = f"not an event stream and not JSON: {raw[:200]}"
                else:
                    _consume_json(ep.api, j, r)
                    r.ttft_s = time.perf_counter() - t0
                    r.chunks = 1
    except urllib.error.HTTPError as e:
        r.error = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
    except Exception as e:  # noqa: BLE001
        r.error = f"{type(e).__name__}: {e}"[:300]
    r.total_s = time.perf_counter() - t0
    return r


def ask(ep: Endpoint, system: str, user: str, max_tokens: int = 1024, temperature: float | None = 0.0,
        timeout: float = 300) -> Reply:
    r = _ask_once(ep, system, user, max_tokens, temperature, timeout)
    if temperature is not None and r.error.startswith("HTTP 400") and "temperature" in r.error.lower():
        # Some models (the reasoning ones) reject a temperature outright; retry once without it.
        retry = _ask_once(ep, system, user, max_tokens, None, timeout)
        if not retry.error:
            return retry
    return r


def _consume(api: str, ev: dict, r: Reply) -> str:
    if api == "anthropic":
        t = ev.get("type")
        if t == "message_start":
            m = ev.get("message") or {}
            r.model_claimed = m.get("model", "")
            r.input_tokens = (m.get("usage") or {}).get("input_tokens")
        elif t == "content_block_delta":
            d = ev.get("delta") or {}
            return d.get("text", "") if d.get("type") == "text_delta" else ""
        elif t == "message_delta":
            r.output_tokens = (ev.get("usage") or {}).get("output_tokens", r.output_tokens)
            r.finish = (ev.get("delta") or {}).get("stop_reason") or r.finish
        elif t == "error":
            r.error = json.dumps(ev.get("error"))[:300]
        return ""
    if api == "responses":
        return _consume_responses(ev, r)
    r.model_claimed = ev.get("model") or r.model_claimed
    if ev.get("usage"):
        r.input_tokens = ev["usage"].get("prompt_tokens", r.input_tokens)
        r.output_tokens = ev["usage"].get("completion_tokens", r.output_tokens)
    ch = ev.get("choices") or []
    if ch:
        r.finish = ch[0].get("finish_reason") or r.finish
        return (ch[0].get("delta") or {}).get("content") or ""
    return ""


# OpenAI Responses: input / instructions, usage.input_tokens / output_tokens, and a stream of
# response.* events whose text arrives as response.output_text.delta.
_INCOMPLETE_FINISH = {"max_output_tokens": "max_tokens", "content_filter": "content_filter"}


def _responses_usage(obj: dict, r: Reply) -> None:
    u = obj.get("usage") or {}
    if u.get("input_tokens") is not None:
        r.input_tokens = u["input_tokens"]
    if u.get("output_tokens") is not None:
        r.output_tokens = u["output_tokens"]


def _responses_refusal(obj: dict, r: Reply) -> None:
    for item in obj.get("output") or []:
        if not isinstance(item, dict):
            continue
        for c in item.get("content") or []:
            if isinstance(c, dict) and c.get("type") == "refusal":
                r.finish = "refusal"


def _consume_responses(ev: dict, r: Reply) -> str:
    t = ev.get("type") or ""
    if t == "response.output_text.delta":
        return ev.get("delta") or ""
    if t == "response.output_text.done":
        return "" if r.text else (ev.get("text") or "")
    if t in ("response.created", "response.in_progress", "response.completed", "response.incomplete",
             "response.failed", "response.output_item.done"):
        obj = ev.get("response")
        if not isinstance(obj, dict):
            obj = ev.get("item") if isinstance(ev.get("item"), dict) else {}
        r.model_claimed = obj.get("model") or r.model_claimed
        _responses_usage(obj, r)
        _responses_refusal(obj, r)
        if t == "response.completed":
            r.finish = r.finish or "stop"
        elif t == "response.incomplete":
            reason = (obj.get("incomplete_details") or {}).get("reason")
            r.finish = _INCOMPLETE_FINISH.get(reason, reason or "incomplete")
        elif t == "response.failed":
            err = obj.get("error") or {}
            r.error = (json.dumps(err) if err else "response.failed")[:300]
        return ""
    if t == "error":
        r.error = (ev.get("message") or json.dumps(ev))[:300]
    return ""


def _consume_json(api: str, j, r: Reply) -> None:
    """A single JSON body — the server ignored stream=true, or the client asked for no stream."""
    if not isinstance(j, dict):
        r.error = f"unexpected body: {str(j)[:200]}"
        return
    if j.get("error") is not None and not (j.get("output") or j.get("choices") or j.get("content")):
        r.error = json.dumps(j.get("error"))[:300]
        return
    if api == "anthropic":
        r.model_claimed = j.get("model") or r.model_claimed
        for b in j.get("content") or []:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    r.text += b.get("text") or ""
                elif b.get("type") == "refusal":
                    r.finish = "refusal"
        r.finish = j.get("stop_reason") or r.finish
        u = j.get("usage") or {}
        r.input_tokens = u.get("input_tokens", r.input_tokens)
        r.output_tokens = u.get("output_tokens", r.output_tokens)
        return
    if api == "responses":
        r.model_claimed = j.get("model") or r.model_claimed
        for item in j.get("output") or []:
            if not isinstance(item, dict):
                continue
            for c in item.get("content") or []:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "output_text":
                    r.text += c.get("text") or ""
                elif c.get("type") == "refusal":
                    r.finish = "refusal"
        _responses_usage(j, r)
        status = j.get("status")
        if status == "completed":
            r.finish = r.finish or "stop"
        elif status == "incomplete":
            reason = (j.get("incomplete_details") or {}).get("reason")
            r.finish = _INCOMPLETE_FINISH.get(reason, reason or "incomplete")
        elif status == "failed":
            r.error = (json.dumps(j.get("error") or {}) or "response.failed")[:300]
        return
    r.model_claimed = j.get("model") or r.model_claimed
    u = j.get("usage") or {}
    r.input_tokens = u.get("prompt_tokens", r.input_tokens)
    r.output_tokens = u.get("completion_tokens", r.output_tokens)
    ch = (j.get("choices") or [{}])[0] if isinstance(j.get("choices"), list) and j.get("choices") else {}
    msg = ch.get("message") or {}
    c = msg.get("content")
    if isinstance(c, list):
        c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
    r.text = c or ""
    r.finish = ch.get("finish_reason") or r.finish
