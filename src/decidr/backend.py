"""How `Client` talks to a model provider.

`OllamaBackend` is the only one built in and needs nothing beyond the
standard library -- it's what every example and test in this project uses.
`LiteLLMBackend` is optional (`pip install decidr[litellm]`) and routes the
same calls through LiteLLM (https://github.com/BerriAI/litellm), which
understands 100+ providers -- OpenAI, Bedrock, hosted vLLM, and Ollama
itself (though not for logprobs -- see below) -- behind one call, so decidr
can run against a hosted model without decidr itself depending on LiteLLM
by default. Anthropic (Claude) is a provider LiteLLM reaches but decidr
still can't use through it: Claude's API has no `logprobs` field on any
route, checked directly against Anthropic's own docs, not assumed -- see
docs/PROVIDERS.md.

Both implement the same one-method contract: given the messages for one
race, return the model's reply and its logprobs at that position, in one
normalized shape (see `Backend.chat`'s docstring). Every other part of this
package -- prefix matching, the id hierarchy, calibration -- works purely in
terms of that shape and never knows which backend produced it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class DecisionError(ValueError):
    """Bad row, or a backend that could not answer it. Defined here (not in
    core.py) because backends need to raise it and core.py imports backends,
    not the other way round; core.py re-exports this same class rather than
    defining a second one, so `except DecisionError` anywhere in this
    package catches every source of it."""


class Backend:
    """Base class for talking to a model provider. Subclass and implement
    `chat`; `Client` calls nothing else on a backend."""

    def chat(self, model: str, messages: list[dict]) -> dict:
        """Send `messages`, predicting exactly one token (temperature 0),
        and return:

            {"content": str | None,        # the model's own reply, if any
             "logprobs": [{"token": str, "logprob": float,
                            "top_logprobs": [{"token": str, "logprob": float}, ...]}]}

        `logprobs` has zero entries if the provider returned no logprob
        information for this call at all (not the same as an empty
        `top_logprobs` list inside a real entry) -- core.py treats a truly
        empty list as "the server didn't support this," and surfaces a clear
        error rather than guessing at a decision with no numbers behind it.
        """
        raise NotImplementedError


class OllamaBackend(Backend):
    """Talks to one Ollama server's `/api/chat`. decidr's only
    dependency-free path -- `urllib` from the standard library, nothing else.
    """

    # Ollama's own cap on this field, checked directly against its source
    # rather than assumed (server/routes.go enforces `> 20` at four call
    # sites, before either of its own backends ever see the request).
    TOP_LOGPROBS = 20

    def __init__(self, host: str = "http://127.0.0.1:11434", timeout: float = 120.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def chat(self, model: str, messages: list[dict]) -> dict:
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            # A reasoning preamble would put thinking tokens in the answer
            # slot, so the very next token stops being the decision.
            "think": False,
            "options": {"num_predict": 1, "temperature": 0},
            "logprobs": True,
            "top_logprobs": self.TOP_LOGPROBS,
        }
        resp = self._post("/api/chat", body)
        return {
            "content": (resp.get("message") or {}).get("content"),
            "logprobs": resp.get("logprobs") or [],  # already this exact shape
        }

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except json.JSONDecodeError:
                pass
            raise DecisionError(f"{path} failed ({e.code}): {detail}") from e
        except urllib.error.URLError as e:
            raise DecisionError(f"cannot reach Ollama at {self.host}: {e.reason}") from e


class LiteLLMBackend(Backend):
    """Routes through LiteLLM instead of talking to Ollama directly, so
    `Client` can run against any provider LiteLLM supports that actually
    returns `logprobs`: OpenAI, Bedrock, a hosted vLLM endpoint, or Ollama
    itself under a different name. Not installed by default --
    `pip install decidr[litellm]`.

    Anthropic (Claude) is reachable through LiteLLM but not usable here:
    Claude's API has no `logprobs` field on any route (native Messages API
    or Anthropic's own OpenAI-compatible endpoint), so this backend can't
    get anything to score from it regardless of how the call is routed --
    see docs/PROVIDERS.md.

    LiteLLM's own top_logprobs ceiling varies by provider; this backend
    doesn't try to raise or detect it, since decidr's per-level branching
    (`MAX_BRANCHES_PER_LEVEL` in core.py) already keeps each individual race
    small regardless of what a provider allows.

    Any keyword LiteLLM's own `completion()` accepts (`api_key`, `api_base`,
    a routed model name like `"gpt-4o-mini"` or `"bedrock/meta.llama3-1-8b..."`)
    can be passed here and is forwarded on every call.
    """

    def __init__(self, **litellm_kwargs):
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "LiteLLMBackend needs the optional litellm package: pip install decidr[litellm]"
            ) from e
        self._litellm = litellm
        self._kwargs = litellm_kwargs

    def chat(self, model: str, messages: list[dict]) -> dict:
        resp = self._litellm.completion(
            model=model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            **self._kwargs,
        )
        choice = resp.choices[0]
        content = getattr(choice.message, "content", None)
        entries: list[dict] = []
        lp = getattr(choice, "logprobs", None)
        content_list = getattr(lp, "content", None) if lp else None
        if content_list:
            first = content_list[0]
            entries = [{
                "token": first.token,
                "logprob": first.logprob,
                "top_logprobs": [
                    {"token": t.token, "logprob": t.logprob} for t in (first.top_logprobs or [])
                ],
            }]
        return {"content": content, "logprobs": entries}
