"""Fit one temperature scalar to a workload's labeled decisions, and measure
whether it actually helped.

This is arithmetic over numbers you already have, not another model call.
`Decision.logprobs` already holds every scored option's raw logprob; this
module just asks "what T makes softmax(logprobs / T) least surprised by the
correct answers, on average" and reports before/after calibration error so
that answer is checked rather than assumed.

Guarantee that matters: dividing every option's logprob by the same T is a
monotone rescaling, so it can never change which option has the highest
probability. Fitting temperature only touches confidence, never choice --
calibrate() asserts this rather than trusting it.

Usage:

    from decidr import Client
    from decidr.calibrate import fit_temperature, evaluate

    client = Client(model="qwen3.5:4b")
    labeled = [(client.decide(row), row["correct_id"]) for row in labeled_rows]

    result = fit_temperature(labeled)
    print(result.temperature, result.ece_before, result.ece_after)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .core import Decision, softmax

_DEFAULT_GRID = [round(0.1 * i, 2) for i in range(1, 101)]  # 0.1 .. 10.0


@dataclass
class CalibrationResult:
    temperature: float
    n: int
    ece_before: float   # expected calibration error at T=1 (raw softmax)
    ece_after: float     # expected calibration error at the fitted T
    accuracy: float      # unaffected by T; reported so a reader isn't left computing it


def _rescaled_probs(logprobs: dict[str, float], temperature: float) -> dict[str, float]:
    ids = list(logprobs)
    probs = softmax([logprobs[i] for i in ids], temperature)
    return dict(zip(ids, probs))


def _nll(pairs: list[tuple[Decision, str]], temperature: float) -> float:
    total = 0.0
    for decision, correct_id in pairs:
        probs = _rescaled_probs(decision.logprobs, temperature)
        p = max(probs.get(correct_id, 1e-9), 1e-9)
        total += -math.log(p)
    return total / len(pairs)


def expected_calibration_error(pairs: list[tuple[Decision, str]], temperature: float, bins: int = 10) -> float:
    """Mean, over `bins` equal-width confidence buckets, of |confidence - accuracy|,
    weighted by how many predictions fall in each bucket. Standard ECE."""
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for decision, correct_id in pairs:
        probs = _rescaled_probs(decision.logprobs, temperature)
        choice = max(probs, key=probs.get)
        confidence = probs[choice]
        correct = choice == correct_id
        idx = min(int(confidence * bins), bins - 1)
        buckets[idx].append((confidence, correct))

    n = len(pairs)
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        avg_acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        ece += (len(bucket) / n) * abs(avg_conf - avg_acc)
    return ece


def fit_temperature(pairs: list[tuple[Decision, str]], grid: list[float] | None = None) -> CalibrationResult:
    """Grid search over `grid` (default 0.1-10.0 in steps of 0.1) for the T that
    minimizes mean negative log-likelihood on `pairs`. A grid rather than a
    gradient method because this is a scalar over a small, cheap range -- no
    need for machinery that could itself silently fail to converge.

    `pairs` is a list of (Decision, correct_option_id). Decisions with fewer
    than 2 scored options are skipped (nothing to calibrate).
    """
    usable = [(d, c) for d, c in pairs if len(d.logprobs) >= 2]
    if len(usable) < 10:
        raise ValueError(
            f"fit_temperature needs at least 10 usable labeled decisions, got {len(usable)}. "
            "A handful of examples will overfit T to noise rather than measure anything."
        )

    best_t, best_nll = 1.0, _nll(usable, 1.0)
    for t in grid or _DEFAULT_GRID:
        nll = _nll(usable, t)
        if nll < best_nll:
            best_t, best_nll = t, nll

    accuracy = sum(1 for d, c in usable if d.choice == c) / len(usable)
    ece_before = expected_calibration_error(usable, 1.0)
    ece_after = expected_calibration_error(usable, best_t)

    # The guarantee this module exists to protect: rescaling logits by a
    # constant must never move the argmax. If it did, something upstream
    # (softmax, or how logprobs were assembled) is broken, and a caller
    # trusting "confidence changed, choice didn't" needs to know immediately.
    for d, _ in usable:
        raw_choice = max(d.logprobs, key=d.logprobs.get)
        scaled = _rescaled_probs(d.logprobs, best_t)
        scaled_choice = max(scaled, key=scaled.get)
        assert raw_choice == scaled_choice, (
            f"temperature scaling changed argmax for decision {d.id!r}: "
            f"{raw_choice!r} -> {scaled_choice!r}. This should be mathematically "
            "impossible; softmax or logprobs are corrupted somewhere upstream."
        )

    return CalibrationResult(
        temperature=best_t,
        n=len(usable),
        ece_before=round(ece_before, 4),
        ece_after=round(ece_after, 4),
        accuracy=round(accuracy, 4),
    )


def evaluate_out_of_fold(pairs: list[tuple[Decision, str]], folds: int = 5, seed: int = 0) -> CalibrationResult:
    """Fit T on `folds - 1` folds and measure ECE on the held-out fold, repeated
    so every row is held out exactly once. This is what makes an ECE number
    trustworthy rather than curve-fit: fit_temperature() alone can report a
    good-looking ECE just by overfitting T to this exact sample.

    Reports the mean fitted T and mean out-of-fold ECE across folds, plus the
    in-sample ECE-before for comparison. Needs at least `folds * 10` usable rows.
    """
    usable = [(d, c) for d, c in pairs if len(d.logprobs) >= 2]
    if len(usable) < folds * 10:
        raise ValueError(
            f"evaluate_out_of_fold needs at least {folds * 10} usable labeled decisions "
            f"for {folds} folds, got {len(usable)}."
        )

    rng = random.Random(seed)
    shuffled = usable[:]
    rng.shuffle(shuffled)
    fold_size = len(shuffled) // folds

    temperatures, eces, accs = [], [], []
    for i in range(folds):
        start, end = i * fold_size, (i + 1) * fold_size if i < folds - 1 else len(shuffled)
        held_out = shuffled[start:end]
        train = shuffled[:start] + shuffled[end:]

        fit = fit_temperature(train)
        temperatures.append(fit.temperature)
        eces.append(expected_calibration_error(held_out, fit.temperature))
        accs.append(sum(1 for d, c in held_out if d.choice == c) / len(held_out))

    return CalibrationResult(
        temperature=round(sum(temperatures) / folds, 3),
        n=len(usable),
        ece_before=round(expected_calibration_error(usable, 1.0), 4),
        ece_after=round(sum(eces) / folds, 4),
        accuracy=round(sum(accs) / folds, 4),
    )
