import pytest

from decidr import Client, DecisionError, build_prefix_messages, validate_row
from decidr.backend import Backend, _to_openai_message


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


def test_to_openai_message_raises_on_image_with_neither_url_nor_data():
    message = {"role": "user", "content": [{"type": "image"}]}
    with pytest.raises(DecisionError):
        _to_openai_message(message)


def test_multimodal_row_end_to_end_through_a_fake_backend():
    # A Backend receives the row's content blocks in decidr's own
    # generic shape (build_prefix_messages's output) -- converting that
    # to a specific provider's wire format (e.g. OpenAI's "image_url")
    # is each backend's own job, tested separately in
    # test_to_openai_message_* above, not something a generic Backend
    # implementation ever sees.
    class FakeBackend(Backend):
        def chat(self, model, messages, max_tokens=1):
            sent_content = messages[-1]["content"]
            assert isinstance(sent_content, list)
            assert any(part["type"] == "image" for part in sent_content)
            return {
                "content": "cat",
                "logprobs": [{"token": "cat", "logprob": -0.1, "top_logprobs": [{"token": "cat", "logprob": -0.1}, {"token": "dog", "logprob": -2.0}]}],
            }

    client = Client(model="gpt-4o-mini", backend=FakeBackend(), cache=False)
    decision = client.decide(_image_row([{"type": "image", "data": "aGVsbG8="}]))
    assert decision.choice == "cat"
