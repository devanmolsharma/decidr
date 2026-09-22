import math
import random

import pytest

from decidr.calibrate import evaluate_out_of_fold, expected_calibration_error, fit_temperature
from decidr.core import Decision


def make_decision(logprobs: dict[str, float], id_: str = "x") -> Decision:
    ids = list(logprobs)
    probs = {i: math.exp(logprobs[i]) for i in ids}
    total = sum(probs.values())
    probs = {i: p / total for i, p in probs.items()}
    return Decision(id=id_, choice=max(probs, key=probs.get), probabilities=probs,
                     logprobs=logprobs, mode="exact")


def overconfident_dataset(n: int, true_t: float, seed: int = 0) -> list[tuple[Decision, str]]:
    """Synthetic decisions where the *true* generating temperature is known, so
    fit_temperature's recovered T can be checked against ground truth instead
    of just checked for "doesn't crash".

    Ground truth: at true separation `gap`, the correct answer really is right
    with probability softmax([gap, 0])[0] -- the label is *sampled* from that
    probability, not always true_gap-favored, or the "model" would be a perfect
    predictor and T=0 (maximum sharpening) would legitimately minimize NLL.

    The model's *reported* logprobs then sharpen that true gap by true_t (per
    the module's convention, calibrated_prob = softmax(logits / T), so a model
    overconfident by factor true_t reports gaps already multiplied by true_t;
    dividing by the fitted T should undo exactly that and recover the true gap).
    true_t > 1 simulates overconfidence, true_t < 1 underconfidence.
    """
    rng = random.Random(seed)
    pairs = []
    for i in range(n):
        true_gap = rng.uniform(0.3, 1.5)
        p_a_correct = 1.0 / (1.0 + math.exp(-true_gap))  # softmax([gap, 0])[0]
        correct = "A" if rng.random() < p_a_correct else "B"
        reported_gap = true_gap * true_t
        # The reported logprobs always favor A (that's the model's belief);
        # whether A is actually correct was already decided above, so the
        # model is sometimes confidently wrong -- exactly what miscalibration
        # looks like in practice.
        logprobs = {"A": reported_gap, "B": 0.0}
        pairs.append((make_decision(logprobs, id_=f"row{i}"), correct))
    return pairs


def test_fit_recovers_a_known_overconfidence_factor():
    pairs = overconfident_dataset(n=400, true_t=2.5)
    result = fit_temperature(pairs)
    # Grid search in 0.1 steps against a noisy synthetic set won't hit 2.5
    # exactly, but should land in its neighborhood.
    assert 2.0 <= result.temperature <= 3.0
    assert result.ece_after < result.ece_before


def test_fit_on_already_calibrated_data_stays_near_one():
    pairs = overconfident_dataset(n=400, true_t=1.0)
    result = fit_temperature(pairs)
    assert 0.7 <= result.temperature <= 1.5


def test_temperature_never_changes_the_choice():
    # This is the guarantee the module exists to protect, so it is tested
    # directly rather than only relied on via the internal assertion.
    pairs = overconfident_dataset(n=100, true_t=3.0)
    result = fit_temperature(pairs)
    for decision, _ in pairs:
        from decidr.calibrate import _rescaled_probs
        raw_choice = decision.choice
        scaled = _rescaled_probs(decision.logprobs, result.temperature)
        assert max(scaled, key=scaled.get) == raw_choice


def test_fit_temperature_requires_enough_data():
    pairs = overconfident_dataset(n=5, true_t=1.0)
    with pytest.raises(ValueError, match="at least 10"):
        fit_temperature(pairs)


def test_ece_is_zero_for_a_perfectly_calibrated_toy_case():
    # 100 decisions all reporting exactly 70% confidence, 70 of them correct:
    # a textbook calibrated set, ECE should be ~0.
    pairs = []
    for i in range(100):
        logprob_gap = math.log(0.7 / 0.3)  # softmax([gap, 0]) == [0.7, 0.3]
        d = make_decision({"A": logprob_gap, "B": 0.0}, id_=f"row{i}")
        answer = "A" if i < 70 else "B"
        pairs.append((d, answer))
    ece = expected_calibration_error(pairs, temperature=1.0)
    assert ece < 0.05


def test_out_of_fold_reports_higher_ece_than_naive_in_sample_fit():
    # Out-of-fold evaluation exists specifically because in-sample fitting can
    # look better than it is. On noisy synthetic data the OOF ECE should not
    # be dramatically better than the in-sample one -- if it were, that would
    # itself indicate a bug (leakage) rather than a good result.
    pairs = overconfident_dataset(n=200, true_t=2.0, seed=1)
    in_sample = fit_temperature(pairs)
    oof = evaluate_out_of_fold(pairs, folds=5, seed=1)
    assert oof.ece_after >= in_sample.ece_after - 0.05  # OOF should not look implausibly better


def test_out_of_fold_requires_enough_rows_per_fold():
    pairs = overconfident_dataset(n=20, true_t=1.0)
    with pytest.raises(ValueError, match="at least 50"):
        evaluate_out_of_fold(pairs, folds=5)
