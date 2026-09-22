import json

import pytest

from decidr import Client, DecisionError, OllamaBackend, build_prefix_messages, validate_row
from decidr.backend import _to_ollama_message, _to_openai_message


class _FakeHTTPResponse:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _image_row(state):
    return {
        "id": "r1",
        "state": state,
        "question": "what is it?",
        "options": [
            {"id": "cat", "description": "a cat"},
            {"id": "dog", "description": "a dog"},
        ],
    }


def test_validate_row_accepts_content_block_state_with_url_image():
    validate_row(_image_row([
        {"type": "text", "text": "look at this:"},
        {"type": "image", "url": "https://example.com/cat.png"},
    ]))


def test_validate_row_accepts_content_block_state_with_inline_data_image():
    validate_row(_image_row([{"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"}]))


def test_validate_row_rejects_image_block_with_neither_url_nor_data():
    with pytest.raises(DecisionError):
        validate_row(_image_row([{"type": "image"}]))


def test_validate_row_rejects_image_block_with_both_url_and_data():
    with pytest.raises(DecisionError):
        validate_row(_image_row([{"type": "image", "url": "https://x/y.png", "data": "aGk="}]))


def test_validate_row_rejects_unknown_block_type():
    with pytest.raises(DecisionError):
        validate_row(_image_row([{"type": "bogus"}]))


def test_build_prefix_messages_keeps_content_block_state_as_a_list():
    messages = build_prefix_messages(_image_row([{"type": "image", "url": "https://example.com/cat.png"}]))
    user = messages[1]
    assert isinstance(user["content"], list)
    assert user["content"][0]["type"] == "image"
    assert user["content"][1]["type"] == "text"
    assert "Answer with exactly one of: cat, dog." in user["content"][1]["text"]


def test_to_ollama_message_moves_image_data_into_images_field():
    message = {
        "role": "user",
        "content": [{"type": "text", "text": "what is this"}, {"type": "image", "data": "aGVsbG8="}],
    }
    out = _to_ollama_message(message)
    assert out["content"] == "what is this"
    assert out["images"] == ["aGVsbG8="]


def test_to_ollama_message_raises_on_url_only_image():
    message = {"role": "user", "content": [{"type": "image", "url": "https://x/y.png"}]}
    with pytest.raises(DecisionError):
        _to_ollama_message(message)


def test_to_ollama_message_raises_on_video_block():
    message = {"role": "user", "content": [{"type": "video", "data": "aGVsbG8="}]}
    with pytest.raises(DecisionError):
        _to_ollama_message(message)


def test_to_openai_message_builds_typed_content_array():
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
        ],
    }
    out = _to_openai_message(message)
    assert out["content"] == [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
    ]


def test_to_openai_message_passes_through_plain_url():
    message = {"role": "user", "content": [{"type": "image", "url": "https://x/y.png"}]}
    out = _to_openai_message(message)
    assert out["content"] == [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]


def test_to_openai_message_raises_on_video_block():
    message = {"role": "user", "content": [{"type": "video", "url": "https://x/y.mp4"}]}
    with pytest.raises(DecisionError):
        _to_openai_message(message)


def test_ollama_backend_sends_images_field_end_to_end(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["body"] = json.loads(req.data)
        return _FakeHTTPResponse({
            "message": {"content": "cat"},
            "logprobs": [{"token": "cat", "logprob": -0.1,
                          "top_logprobs": [{"token": "cat", "logprob": -0.1}, {"token": "dog", "logprob": -2.0}]}],
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = Client(model="qwen3.5:4b", backend=OllamaBackend(host="http://localhost:11434"))
    decision = client.decide(_image_row([{"type": "image", "data": "aGVsbG8="}]))

    assert decision.choice == "cat"
    sent_message = captured["body"]["messages"][-1]
    assert sent_message["images"] == ["aGVsbG8="]
