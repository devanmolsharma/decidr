"""How `Client` talks to a model provider. See docs/SPEC.md §3 for the
normative `Backend` contract (shared with the TypeScript port).

`OpenAIBackend`, built on the official `openai` PyPI package, is the
single backend -- and it's enough. It works against any
OpenAI-compatible `/v1/chat/completions` endpoint: OpenAI itself,
Ollama's own OpenAI-compatible endpoint (verified live to return real,
correct logprobs), and other OpenAI-compatible hosts confirmed to
forward logprobs correctly (see docs/PROVIDERS.md). There is
deliberately no separate Ollama-specific backend or multi-provider
abstraction layer -- SPEC.md §3.2 has the full researched reasoning for
why a general-purpose multi-provider client (LiteLLM, OpenRouter, and
others) has a silent or structural logprobs gap for at least one major
provider, and MUST NOT be relied on for the core path.
"""

from __future__ import annotations


class DecisionError(ValueError):
    """Bad row, or a backend that could not answer it. Defined here (not
    in core.py) because backends need to raise it and core.py imports
    backends, not the other way round."""


class Backend:
    """Base class for talking to a model provider. Subclass and
    implement `chat`; `warmup`/`discover_tokens_batch` have default
    implementations built purely on `chat` (see below) that a subclass
    may override with something cheaper if its provider offers one."""

    def chat(self, model: str, messages: list[dict], max_tokens: int = 1) -> dict:
        """Send `messages`, predicting `max_tokens` tokens (usually just
        1) at temperature 0, and return:

            {"content": str | None,
             "logprobs": [{"token": str, "logprob": float,
                            "top_logprobs": [{"token": str, "logprob": float}, ...]}]}

        `logprobs` has zero entries if the provider returned no logprob
        information for this call at all -- treated as a hard error by
        `Client`, never guessed at.
        """
        raise NotImplementedError

    def warmup(self, model: str) -> None:
        """Pre-establish the connection before a latency-sensitive
        `decide()` call. Default: one throwaway `chat` call."""
        self.chat(model, [{"role": "user", "content": "."}])

    def discover_tokens_batch(self, model: str, words: list[str]) -> dict[str, list[str]]:
        """Discover the real token boundaries of several option ids in
        one request. See docs/SPEC.md §8 for the normative algorithm."""
        unique = list(dict.fromkeys(words))
        result: dict[str, list[str]] = {}
        if not unique:
            return result

        prompt = f"List these {len(unique)} words, one per line, exactly as given, nothing else:\n" + "\n".join(unique)
        max_tokens = sum(len(w) for w in unique) + len(unique) * 4 + 8
        resp = self.chat(model, [{"role": "user", "content": prompt}], max_tokens)
        entries = resp.get("logprobs") or []

        word_index = 0
        consumed = ""
        tokens: list[str] = []
        for entry in entries:
            if word_index >= len(unique):
                break
            target = unique[word_index]
            token = entry["token"]
            remaining = target[len(consumed):]
            if remaining and token and remaining.startswith(token):
                tokens.append(token)
                consumed += token
                if consumed == target:
                    result[target] = tokens
                    word_index += 1
                    consumed = ""
                    tokens = []
                continue
            if tokens:
                word_index += 1
                consumed = ""
                tokens = []
        return result


def _to_openai_content(content: object) -> object:
    """OpenAI's `/v1/chat/completions` takes an array of typed parts for
    multimodal content -- `{"type": "text", "text": ...}` and
    `{"type": "image_url", "image_url": {"url": ...}}`, where `url` is a
    normal http(s) link or a `data:` URI. No video/audio input on this
    endpoint -- a block of either type raises."""
    if isinstance(content, str):
        return content
    parts: list[dict] = []
    for block in content:
        block_type = block["type"]
        if block_type == "text":
            parts.append({"type": "text", "text": block["text"]})
        elif block_type == "image":
            url = block.get("url")
            if not url:
                data = block.get("data")
                if not data:
                    raise DecisionError('an "image" content block needs "url" or "data"')
                mime = block.get("mimeType", "image/png")
                url = f"data:{mime};base64,{data}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            raise DecisionError(
                f'cannot send a "{block_type}" content block -- the chat completions endpoint '
                f"has no {block_type} input"
            )
    return parts


def _to_openai_message(message: dict) -> dict:
    return {**message, "content": _to_openai_content(message["content"])}


class OpenAIBackend(Backend):
    """Talks to any OpenAI-compatible `/v1/chat/completions` endpoint --
    OpenAI itself, Ollama's compat endpoint, or a self-hosted/third-party
    host speaking the same wire format. Built on the official `openai`
    PyPI package (see this module's docstring for why, and why there's
    no separate hand-rolled HTTP path or Ollama-specific backend).

    **Only some models support `logprobs`.** Most consistently absent
    from reasoning-focused models (OpenAI's o-series and similar
    elsewhere), present on standard chat models. Some otherwise
    OpenAI-compatible hosts reject `logprobs` outright (Groq returns an
    explicit 400) -- see docs/PROVIDERS.md for the current checked list.
    """

    TOP_LOGPROBS = 20

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
    ):
        try:
            import openai
        except ImportError as e:
            raise ImportError("OpenAIBackend needs the openai package: pip install decidr") from e
        # Ollama's OpenAI-compatible endpoint ignores the key entirely but
        # the SDK requires a non-empty string.
        self._client = openai.OpenAI(api_key=api_key or "ollama", base_url=base_url, timeout=timeout)
        self._openai = openai
        # Some providers (Ollama's OpenAI-compatible endpoint, notably)
        # run a reasoning/thinking preamble by default, which consumes
        # the single requested token on thinking output instead of the
        # real answer -- silently breaking this mechanism with no error.
        # Sending reasoning_effort="none" disables that on providers
        # that support it. Real OpenAI HARD REJECTS that same field for
        # models with no reasoning mode to disable (400, not a silent
        # ignore). There's no reliable way to know in advance which
        # behavior a given (model, provider) pairing has, so this is
        # detected once per backend instance and remembered.
        self._sends_reasoning_effort: bool | None = None

    def chat(self, model: str, messages: list[dict], max_tokens: int = 1) -> dict:
        body = {
            "model": model,
            "messages": [_to_openai_message(m) for m in messages],
            "max_tokens": max_tokens,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": self.TOP_LOGPROBS,
        }
        try:
            if self._sends_reasoning_effort is not False:
                try:
                    response = self._client.chat.completions.create(**body, reasoning_effort="none")
                    self._sends_reasoning_effort = True
                except Exception as e:
                    if self._is_unrecognized_argument_error(e, "reasoning_effort"):
                        self._sends_reasoning_effort = False
                        response = self._client.chat.completions.create(**body)
                    else:
                        raise
            else:
                response = self._client.chat.completions.create(**body)
        except Exception as e:
            raise DecisionError(f"chat completion failed: {e}") from e

        choice = response.choices[0] if response.choices else None
        entries: list[dict] = []
        lp = getattr(choice, "logprobs", None) if choice else None
        content_list = getattr(lp, "content", None) if lp else None
        if content_list:
            entries = [
                {
                    "token": entry.token,
                    "logprob": entry.logprob,
                    "top_logprobs": [{"token": t.token, "logprob": t.logprob} for t in (entry.top_logprobs or [])],
                }
                for entry in content_list
            ]
        return {
            "content": getattr(choice.message, "content", None) if choice else None,
            "logprobs": entries,
        }

    def _is_unrecognized_argument_error(self, error: Exception, param_name: str) -> bool:
        """Whether `error` is specifically the provider's "unrecognized
        request argument" rejection for `param_name` -- a 400 with that
        exact message shape. Used to detect (once) that this provider
        doesn't support `reasoning_effort`, distinct from any other
        request failure, which should still propagate."""
        status = getattr(error, "status_code", None) or getattr(error, "status", None)
        if status != 400:
            return False
        return param_name in str(error)
