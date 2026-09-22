import json
from urllib.error import HTTPError, URLError

import pytest

from decidr import DecisionError, LiteLLMBackend, OllamaBackend


class _FakeHTTPResponse:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_ollama_backend_normalizes_the_response_shape(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data)
        return _FakeHTTPResponse({
            "message": {"content": "billing"},
            "logprobs": [{"token": "billing", "logprob": -0.1,
                          "top_logprobs": [{"token": "billing", "logprob": -0.1}]}],
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    backend = OllamaBackend(host="http://localhost:11434")
    result = backend.chat("qwen3.5:4b", [{"role": "user", "content": "hi"}])

    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["body"]["model"] == "qwen3.5:4b"
    assert captured["body"]["top_logprobs"] == OllamaBackend.TOP_LOGPROBS == 20
    assert captured["body"]["options"] == {"num_predict": 1, "temperature": 0}
    # normalized shape: content pulled up from message.content, logprobs passed through as-is
    assert result == {
        "content": "billing",
        "logprobs": [{"token": "billing", "logprob": -0.1,
                      "top_logprobs": [{"token": "billing", "logprob": -0.1}]}],
    }


def test_ollama_backend_handles_a_response_with_no_logprobs():
    # A model/server that doesn't return logprob info at all should come
    # back as an empty list, not a missing key that callers have to guard.
    class _NoLogprobsBackend(OllamaBackend):
        def _post(self, path, body):
            return {"message": {"content": "hi"}}

    result = _NoLogprobsBackend().chat("test", [])
    assert result == {"content": "hi", "logprobs": []}


def test_ollama_backend_wraps_http_errors(monkeypatch):
    def fake_urlopen(req, timeout):
        raise HTTPError(req.full_url, 400, "Bad Request", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    backend = OllamaBackend()
    with pytest.raises(DecisionError, match=r"failed \(400\)"):
        backend.chat("test", [])


def test_ollama_backend_wraps_connection_errors(monkeypatch):
    def fake_urlopen(req, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    backend = OllamaBackend(host="http://localhost:11434")
    with pytest.raises(DecisionError, match="cannot reach Ollama"):
        backend.chat("test", [])


def test_litellm_backend_raises_a_clear_error_when_litellm_is_not_installed(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "litellm":
            raise ImportError("no module named litellm")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"pip install decidr\[litellm\]"):
        LiteLLMBackend()


def test_litellm_backend_normalizes_the_response_shape(monkeypatch):
    pytest.importorskip("litellm")

    class FakeTopLogprob:
        def __init__(self, token, logprob):
            self.token, self.logprob = token, logprob

    class FakeLogprobContent:
        def __init__(self):
            self.token, self.logprob = "billing", -0.1
            self.top_logprobs = [FakeTopLogprob("billing", -0.1)]

    class FakeLogprobs:
        def __init__(self):
            self.content = [FakeLogprobContent()]

    class FakeMessage:
        content = "billing"

    class FakeChoice:
        message = FakeMessage()
        logprobs = FakeLogprobs()

    class FakeResponse:
        choices = [FakeChoice()]

    import litellm
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(litellm, "completion", fake_completion)
    backend = LiteLLMBackend(api_key="sk-fake")
    result = backend.chat("gpt-4o-mini", [{"role": "user", "content": "hi"}])

    assert captured["model"] == "gpt-4o-mini"
    assert captured["max_tokens"] == 1
    assert captured["temperature"] == 0
    assert captured["api_key"] == "sk-fake"  # forwarded kwargs reach the real call
    assert result == {
        "content": "billing",
        "logprobs": [{"token": "billing", "logprob": -0.1,
                      "top_logprobs": [{"token": "billing", "logprob": -0.1}]}],
    }
