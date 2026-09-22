import json
import math

import pytest

from decidr import Backend, Client, DecisionError, OllamaBackend, softmax, validate_row

ROW = {
    "id": "route-1",
    "state": "Customer cannot access an account after a password reset.",
    "question": "Which queue should handle this request?",
    "options": [
        {"id": "access", "description": "Account access support."},
        {"id": "billing", "description": "Billing support."},
    ],
}


def test_validate_accepts_a_good_row():
    validate_row(ROW)


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda r: r.pop("question"), "missing fields"),
        (lambda r: r.update(question=""), "nonempty strings"),
        (lambda r: r.update(state=""), "state must be"),
        (lambda r: r.update(options=r["options"][:1]), "at least 2"),
        (lambda r: r.update(options=[{"id": "a"}, {"id": "b"}]), "id and description"),
        (lambda r: r.update(options=[{"id": "x", "description": "d"}] * 2), "unique"),
        (lambda r: r.update(options=[{"id": "Access", "description": "d"}, r["options"][1]]), "required format"),
        (lambda r: r.update(options=[{"id": "x" * 41, "description": "d"}, r["options"][1]]), "too long"),
        (lambda r: r.update(options=[{"id": "access", "description": "d"}, {"id": "access_denied", "description": "d"}]),
         "is a prefix of"),
    ],
)
def test_validate_rejects_bad_rows(mutate, expected):
    row = json.loads(json.dumps(ROW))
    mutate(row)
    with pytest.raises(DecisionError, match=expected):
        validate_row(row)


def test_validate_skips_format_checks_when_asked():
    # Internal escape hatch used only where ids aren't real option ids (see
    # validate_row's docstring) -- exercised directly here since nothing in
    # the current codebase actually calls it this way anymore.
    row = json.loads(json.dumps(ROW))
    row["options"][0]["id"] = "x" * 41
    validate_row(row, check_id_format=False)  # should not raise


def test_softmax_normalizes_over_supplied_options_only():
    probs = softmax([-0.1, -2.0, -8.0])
    assert math.isclose(sum(probs), 1.0)
    assert probs[0] > probs[1] > probs[2]


def test_softmax_handles_degenerate_input():
    assert softmax([]) == []
    equal = softmax([-1.0, -1.0])
    assert math.isclose(equal[0], 0.5)


def test_temperature_flattens_the_distribution():
    sharp = softmax([-0.1, -3.0], temperature=1.0)
    flat = softmax([-0.1, -3.0], temperature=5.0)
    assert flat[0] < sharp[0]  # higher temperature is less confident


class FakeBackend(Backend):
    """Backend stubbed to a fixed response, so decision parsing is tested
    without a live server or any real provider. Records the last call for
    inspection. Speaks the same normalized `{"content", "logprobs"}` shape
    every real backend does -- OllamaBackend's own response-shape mapping is
    tested separately in test_backend.py, not re-tested through this fake."""

    def __init__(self, response):
        self._response = response
        self.last_model = None
        self.last_messages = None

    def chat(self, model, messages):
        self.last_model = model
        self.last_messages = messages
        return self._response


def test_client_defaults_to_an_ollama_backend():
    # No backend given -> a real OllamaBackend is constructed, pointed at
    # the given host. OllamaBackend's own request/response handling is
    # tested directly in test_backend.py, not re-tested through Client here.
    client = Client(model="test", host="http://example.invalid:1234")
    assert isinstance(client.backend, OllamaBackend)
    assert client.backend.host == "http://example.invalid:1234"


def test_client_accepts_an_injected_backend():
    backend = FakeBackend({"content": "access", "logprobs": [
        {"token": "access", "logprob": -0.01, "top_logprobs": [
            {"token": "access", "logprob": -0.01}, {"token": "billing", "logprob": -2.0},
        ]},
    ]})
    client = Client(model="test", backend=backend)
    client.decide(ROW)
    assert backend.last_model == "test"
