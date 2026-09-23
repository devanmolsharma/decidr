import json
import math

import pytest

from decidr import (
    Backend,
    Client,
    Decision,
    DecisionError,
    MIN_ID_LENGTH,
    confidence,
    is_reliable,
    softmax,
    validate_row,
)

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
        (lambda r: r.update(options=r["options"][:1]), "at least 2"),
        (lambda r: r.update(options=[{"id": "a"}, {"id": "b"}]), "id and description"),
        (lambda r: r.update(options=[{"id": "x", "description": "d"}] * 2), "unique"),
        (lambda r: r.update(options=[{"id": "Access", "description": "d"}, r["options"][1]]), "required format"),
        (lambda r: r.update(options=[{"id": "x" * 41, "description": "d"}, r["options"][1]]), "too long"),
        (lambda r: r.update(options=[{"id": "a", "description": "d"}, r["options"][1]]), "shorter than"),
        (
            lambda r: r.update(options=[{"id": "access", "description": "d"}, {"id": "access_denied", "description": "d"}]),
            "is a prefix of",
        ),
    ],
)
def test_validate_rejects_bad_rows(mutate, expected):
    row = json.loads(json.dumps(ROW))
    mutate(row)
    with pytest.raises(DecisionError, match=expected):
        validate_row(row)


def test_validate_min_id_length_boundary():
    row = json.loads(json.dumps(ROW))
    row["options"][0]["id"] = "a" * MIN_ID_LENGTH
    validate_row(row)  # exactly at the floor: fine


def test_validate_skips_format_checks_when_asked():
    row = json.loads(json.dumps(ROW))
    row["options"][0]["id"] = "x" * 41
    validate_row(row, check_id_format=False)


def test_validate_allows_empty_dict_or_list_state():
    # An empty object/array is spec-valid state, not a special case to reject.
    row = json.loads(json.dumps(ROW))
    row["state"] = {}
    validate_row(row)
    row["state"] = []
    validate_row(row)


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
    assert flat[0] < sharp[0]


def test_confidence_reads_the_chosen_options_probability():
    d = Decision(id="r", choice="a", probabilities={"a": 0.7, "b": 0.3}, logprobs={}, mode="prefix")
    assert confidence(d) == 0.7


def test_is_reliable_false_when_unscored_present():
    d = Decision(id="r", choice="a", probabilities={"a": 1.0}, logprobs={}, mode="prefix", unscored=["b"])
    assert not is_reliable(d)


def test_is_reliable_unaffected_by_eliminated_or_stopped_early():
    d = Decision(
        id="r", choice="a", probabilities={"a": 1.0}, logprobs={}, mode="prefix",
        eliminated=["b"], stopped_early=["a"],
    )
    assert is_reliable(d)


class FakeBackend(Backend):
    """Speaks the normalized {"content", "logprobs"} shape every real
    backend does -- OpenAIBackend's own response-shape mapping is tested
    separately in test_backend.py."""

    def __init__(self, response):
        self._response = response
        self.last_model = None
        self.last_messages = None
        self.calls = 0

    def chat(self, model, messages, max_tokens=1):
        self.last_model = model
        self.last_messages = messages
        self.calls += 1
        return self._response


def test_client_accepts_an_injected_backend():
    backend = FakeBackend(
        {
            "content": "access",
            "logprobs": [
                {
                    "token": "access",
                    "logprob": -0.01,
                    "top_logprobs": [{"token": "access", "logprob": -0.01}, {"token": "billing", "logprob": -2.0}],
                }
            ],
        }
    )
    client = Client(model="test", backend=backend, cache=False)
    client.decide(ROW)
    assert backend.last_model == "test"


def test_client_score_weighted_position():
    backend = FakeBackend(
        {
            "content": "crit",
            "logprobs": [
                {
                    "token": "crit",
                    "logprob": -0.01,
                    "top_logprobs": [{"token": "crit", "logprob": -0.01}, {"token": "mod", "logprob": -6.0}, {"token": "cosm", "logprob": -8.0}],
                }
            ],
        }
    )
    client = Client(model="test", backend=backend, cache=False)
    result = client.score(
        {
            "id": "s1",
            "state": "the bug crashes the app",
            "question": "how severe?",
            "levels": [
                {"id": "cosm", "description": "cosmetic"},
                {"id": "mod", "description": "moderate"},
                {"id": "crit", "description": "critical"},
            ],
        }
    )
    assert result["score"] > 1.9
    assert result["decision"].choice == "crit"


def test_client_score_rejects_fewer_than_two_levels():
    client = Client(model="test", backend=FakeBackend({"content": None, "logprobs": []}), cache=False)
    with pytest.raises(DecisionError, match="levels"):
        client.score({"id": "s1", "state": "s", "question": "q", "levels": [{"id": "cosm", "description": "d"}]})


def test_client_truth_reads_the_true_probability():
    backend = FakeBackend(
        {
            "content": "true",
            "logprobs": [
                {"token": "true", "logprob": -0.01, "top_logprobs": [{"token": "true", "logprob": -0.01}, {"token": "false", "logprob": -8.0}]}
            ],
        }
    )
    client = Client(model="test", backend=backend, cache=False)
    result = client.truth({"id": "t1", "state": "a hotdog has bread and a filling", "question": "is a hotdog a sandwich?"})
    assert result["truth"] > 0.99
    assert result["decision"].choice == "true"


def test_client_is_a_context_manager_that_shuts_down_its_pool():
    backend = FakeBackend({"content": "access", "logprobs": [{"token": "access", "logprob": -0.01, "top_logprobs": []}]})
    with Client(model="test", backend=backend, cache=False) as client:
        client.decide(ROW)
    # closing twice (via __exit__ then an explicit call) must not raise
    client.close()
