import pytest

from decidr import Backend, DecisionError, OpenAIBackend


class _FakeTopLogprob:
    def __init__(self, token, logprob):
        self.token, self.logprob = token, logprob


class _FakeLogprobContent:
    def __init__(self, token, logprob, alts):
        self.token, self.logprob = token, logprob
        self.top_logprobs = [_FakeTopLogprob(t, lp) for t, lp in alts.items()]


class _FakeLogprobs:
    def __init__(self, entries):
        self.content = entries


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content, entries):
        self.message = _FakeMessage(content)
        self.logprobs = _FakeLogprobs(entries)


class _FakeResponse:
    def __init__(self, choices):
        self.choices = choices


class _FakeAPIError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


def _make_backend(monkeypatch, create_fn):
    backend = OpenAIBackend(api_key="sk-fake")
    monkeypatch.setattr(backend._client.chat.completions, "create", create_fn)
    return backend


def test_openai_backend_normalizes_the_response_shape(monkeypatch):
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        entry = _FakeLogprobContent("billing", -0.1, {"billing": -0.1})
        return _FakeResponse([_FakeChoice("billing", [entry])])

    backend = _make_backend(monkeypatch, fake_create)
    result = backend.chat("gpt-4o-mini", [{"role": "user", "content": "hi"}])

    assert captured["model"] == "gpt-4o-mini"
    assert captured["max_tokens"] == 1
    assert captured["temperature"] == 0
    assert captured["top_logprobs"] == OpenAIBackend.TOP_LOGPROBS == 20
    assert result == {
        "content": "billing",
        "logprobs": [{"token": "billing", "logprob": -0.1, "top_logprobs": [{"token": "billing", "logprob": -0.1}]}],
    }


def test_openai_backend_handles_a_response_with_no_logprobs(monkeypatch):
    def fake_create(**kwargs):
        return _FakeResponse([_FakeChoice("hi", [])])

    backend = _make_backend(monkeypatch, fake_create)
    result = backend.chat("gpt-4o-mini", [{"role": "user", "content": "hi"}])
    assert result == {"content": "hi", "logprobs": []}


def test_openai_backend_wraps_errors(monkeypatch):
    def fake_create(**kwargs):
        raise RuntimeError("boom")

    backend = _make_backend(monkeypatch, fake_create)
    with pytest.raises(DecisionError, match="chat completion failed"):
        backend.chat("gpt-4o-mini", [{"role": "user", "content": "hi"}])


def test_openai_backend_detects_reasoning_effort_rejection_once(monkeypatch):
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        if "reasoning_effort" in kwargs:
            raise _FakeAPIError("Unrecognized request argument supplied: reasoning_effort", status_code=400)
        entry = _FakeLogprobContent("billing", -0.1, {"billing": -0.1})
        return _FakeResponse([_FakeChoice("billing", [entry])])

    backend = _make_backend(monkeypatch, fake_create)
    backend.chat("gpt-4o-mini", [{"role": "user", "content": "hi"}])
    assert "reasoning_effort" in calls[0]
    assert "reasoning_effort" not in calls[1]

    # second call: remembered not to send it again, only one request this time
    calls.clear()
    backend.chat("gpt-4o-mini", [{"role": "user", "content": "hi"}])
    assert len(calls) == 1
    assert "reasoning_effort" not in calls[0]


def test_openai_backend_propagates_unrelated_errors_even_with_reasoning_effort(monkeypatch):
    def fake_create(**kwargs):
        raise _FakeAPIError("rate limited", status_code=429)

    backend = _make_backend(monkeypatch, fake_create)
    with pytest.raises(DecisionError, match="chat completion failed"):
        backend.chat("gpt-4o-mini", [{"role": "user", "content": "hi"}])


def test_backend_default_discover_tokens_batch_walks_logprob_entries():
    class FakeBackend(Backend):
        def chat(self, model, messages, max_tokens=1):
            return {
                "content": None,
                "logprobs": [
                    {"token": "access", "logprob": -0.1, "top_logprobs": []},
                    {"token": "_denied", "logprob": -0.2, "top_logprobs": []},
                    {"token": "billing", "logprob": -0.3, "top_logprobs": []},
                ],
            }

    result = FakeBackend().discover_tokens_batch("m", ["access_denied", "billing"])
    assert result == {"access_denied": ["access", "_denied"], "billing": ["billing"]}


def test_backend_default_discover_tokens_batch_empty_input():
    class FakeBackend(Backend):
        def chat(self, model, messages, max_tokens=1):
            raise AssertionError("should not be called for empty input")

    assert FakeBackend().discover_tokens_batch("m", []) == {}


def test_backend_default_warmup_sends_one_throwaway_chat():
    calls = []

    class FakeBackend(Backend):
        def chat(self, model, messages, max_tokens=1):
            calls.append((model, messages, max_tokens))
            return {"content": ".", "logprobs": []}

    FakeBackend().warmup("m")
    assert len(calls) == 1
    assert calls[0][0] == "m"
