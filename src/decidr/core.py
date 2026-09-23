"""`Client`: give it a row, get back a `Decision` with real probabilities
over real option ids. See docs/SPEC.md §4-§6 for the normative behavior
this file implements (shared with the TypeScript port).

decidr reads a probability distribution over a closed set of typed
answers directly from a chat model's next-token log-probabilities, in
one or a small number of forward passes, instead of generating free
text and parsing it.
"""

from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

from .backend import Backend, DecisionError, OpenAIBackend
from .prefix import MAX_DEPTH, Candidate, build_prefix_messages, build_tree, group_by_context, match_step
from .token_cache import TokenCache

DEFAULT_HOST = "http://127.0.0.1:11434/v1"

MAX_ID_LENGTH = 40
MIN_ID_LENGTH = 2
ID_FORMAT = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")
MAX_BRANCHES_PER_LEVEL = 16


@dataclass
class Decision:
    id: str
    choice: str
    probabilities: dict[str, float]
    logprobs: dict[str, float]
    mode: Literal["prefix"]
    unscored: list[str] = field(default_factory=list)
    eliminated: list[str] = field(default_factory=list)
    stopped_early: list[str] = field(default_factory=list)
    raw_answer: str | None = None


def confidence(decision: Decision) -> float:
    """`decision.probabilities.get(decision.choice, 0.0)`, or 0 if
    `choice` somehow isn't a key (shouldn't happen -- `choice` is always
    the argmax of `probabilities`). A function, not a method, since
    `Decision` is a plain data shape."""
    return decision.probabilities.get(decision.choice, 0.0)


def is_reliable(decision: Decision) -> bool:
    """`False` when `unscored` is non-empty, meaning `probabilities` is
    missing real measurements for at least one option. Deliberately
    unaffected by `eliminated`/`stopped_early` -- see docs/HIERARCHY.md
    and docs/PREFIX_MATCHING.md for why those aren't measurement gaps."""
    return not decision.unscored


def validate_row(row: dict, check_id_format: bool = True) -> None:
    """Reject a malformed row up front, with a message naming what's
    wrong -- a bad row should fail here, not produce a confident-looking
    wrong answer."""
    required = {"id", "state", "question", "options"}
    if not required <= row.keys():
        raise DecisionError(f"row is missing fields: {sorted(required - row.keys())}")
    if not all(isinstance(row[k], str) and row[k] for k in ("id", "question")):
        raise DecisionError("id and question must be nonempty strings")

    state = row["state"]
    if not isinstance(state, (str, dict, list)):
        raise DecisionError("state must be a string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise DecisionError("state must be finite JSON-compatible data") from e
    if isinstance(state, list) and all(isinstance(b, dict) and "type" in b for b in state):
        for block in state:
            block_type = block["type"]
            if block_type == "text":
                if not isinstance(block.get("text"), str):
                    raise DecisionError('a "text" content block needs a string "text" field')
                continue
            if block_type in ("image", "video", "audio"):
                has_url = bool(block.get("url"))
                has_data = bool(block.get("data"))
                if has_url == has_data:
                    raise DecisionError(f'a "{block_type}" content block needs exactly one of "url" or "data"')
                continue
            raise DecisionError(f'unknown content block type "{block_type}"')

    options = row["options"]
    if not isinstance(options, list) or len(options) < 2:
        raise DecisionError("options must contain at least 2 entries")
    ids = []
    for opt in options:
        if not isinstance(opt, dict) or not isinstance(opt.get("id"), str) or not isinstance(opt.get("description"), str):
            raise DecisionError("each option needs string id and description fields")
        ids.append(opt["id"])
    if len(ids) != len(set(ids)):
        raise DecisionError("option ids must be unique")

    if check_id_format:
        too_long = [i for i in ids if len(i) > MAX_ID_LENGTH]
        if too_long:
            raise DecisionError(f"option id(s) too long: {too_long!r} (max {MAX_ID_LENGTH} chars). See docs/NAMING_IDS.md.")
        too_short = [i for i in ids if len(i) < MIN_ID_LENGTH]
        if too_short:
            raise DecisionError(
                f"option id(s) shorter than {MIN_ID_LENGTH} characters: {too_short!r} -- a single "
                "character is too likely to collide with another option's first token or a common "
                "filler token in the race. See docs/NAMING_IDS.md."
            )
        badly_shaped = [i for i in ids if not ID_FORMAT.match(i)]
        if badly_shaped:
            raise DecisionError(
                f"option id(s) don't match the required format: {badly_shaped!r}. Ids must be "
                "lowercase letters/digits in underscore-separated segments. See docs/NAMING_IDS.md."
            )
        segments = {i: i.split("_") for i in ids}
        for a, a_segs in segments.items():
            for b, b_segs in segments.items():
                if a != b and b_segs[: len(a_segs)] == a_segs:
                    raise DecisionError(
                        f"option id {a!r} is a prefix of {b!r} -- ambiguous as a hierarchy path. "
                        "See docs/NAMING_IDS.md."
                    )


def softmax(logprobs: list[float], temperature: float = 1.0) -> list[float]:
    if not logprobs:
        return []
    scaled = [lp / temperature for lp in logprobs]
    top = max(scaled)
    exp = [math.exp(v - top) for v in scaled]
    total = sum(exp)
    return [v / total for v in exp] if total else [1.0 / len(exp)] * len(exp)


def _found_tokens(entry: dict) -> dict[str, float]:
    """{token: logprob} out of one response position's top_logprobs.
    Tokens are used exactly as returned -- no stripping -- see
    prefix.py's match_step docstring for why."""
    found: dict[str, float] = {}
    token = entry.get("token")
    if token:
        found[token] = entry.get("logprob", 0.0)
    for alt in entry.get("top_logprobs", []):
        found.setdefault(alt["token"], alt["logprob"])
    return found


def _leaf_ids(node) -> list[str]:
    if not node.children:
        return [o["id"] for o in node.options]
    ids: list[str] = []
    for child in node.children.values():
        ids.extend(_leaf_ids(child))
    return ids


class Client:
    """Talks to a model through a `Backend` -- `OpenAIBackend` by
    default, pointed at a local Ollama's OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        timeout: float = 120.0,
        temperature: float = 1.0,
        backend: Backend | None = None,
        exhaustive: bool = True,
        cache: TokenCache | bool = True,
        max_workers: int = 16,
    ):
        self.model = model
        self.temperature = temperature
        self.backend = backend if backend is not None else OpenAIBackend(base_url=host, timeout=timeout)
        self.exhaustive = exhaustive
        if cache is False:
            self._cache: TokenCache | None = None
        elif isinstance(cache, TokenCache):
            self._cache = cache
        else:
            self._cache = TokenCache()
        # One bounded, shared pool for the whole client's lifetime --
        # never one ThreadPoolExecutor per round or per hierarchy node,
        # which would create unbounded (and, across recursive branches,
        # compounding) thread counts. See docs/SPEC.md §11.1: independent
        # requests within one round, and independent hierarchy branches,
        # MUST be sent concurrently, never sequentially.
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _chat(self, messages: list[dict], max_tokens: int = 1) -> dict:
        return self.backend.chat(self.model, messages, max_tokens)

    def _map(self, fn, items: list) -> list:
        """Run `fn` over `items` concurrently on the shared pool. A
        single item runs inline -- no thread-pool overhead for the
        common one-request case."""
        if len(items) <= 1:
            return [fn(item) for item in items]
        return list(self._pool.map(fn, items))

    # ---- decisions ----------------------------------------------------

    def decide(self, row: dict) -> Decision:
        validate_row(row, check_id_format=True)
        return self._decide_tree(row)

    def decide_all(self, rows: list[dict]) -> list[Decision]:
        return [self.decide(row) for row in rows]

    def score(self, row: dict) -> dict:
        """Grade `state` against an ordered rubric (`row["levels"]`, low
        to high) -- comparable to TypeSafe's `Score` primitive.
        Implemented entirely on top of `decide()`: each level becomes an
        ordinary option, and `score` is the probability-weighted level
        index (`sum(index * probability)`), which can land between two
        integers when the model's distribution spans more than one
        level.

        Returns `{"id": row["id"], "score": float, "decision": Decision}`.
        """
        levels = row["levels"]
        if not isinstance(levels, list) or len(levels) < 2:
            raise DecisionError("row['levels'] must be a list with at least 2 entries")
        decision = self.decide(
            {
                "id": row["id"],
                "state": row["state"],
                "question": row["question"],
                "options": [{"id": lvl["id"], "description": lvl["description"]} for lvl in levels],
            }
        )
        index_by_id = {lvl["id"]: i for i, lvl in enumerate(levels)}
        score = sum(index_by_id[opt_id] * p for opt_id, p in decision.probabilities.items() if opt_id in index_by_id)
        return {"id": row["id"], "score": score, "decision": decision}

    def truth(self, row: dict) -> dict:
        """Is `row["question"]` true of `row["state"]`? Comparable to
        TypeSafe's `Noun` primitive. A fixed two-option `decide()`
        between "true" and "false" -- the probability itself is the
        useful signal, not just which side of 0.5 it falls on.

        Returns `{"id": row["id"], "truth": float, "decision": Decision}`.
        """
        decision = self.decide(
            {
                "id": row["id"],
                "state": row["state"],
                "question": row["question"],
                "options": [
                    {"id": "true", "description": "The statement is true."},
                    {"id": "false", "description": "The statement is false."},
                ],
            }
        )
        return {"id": row["id"], "truth": decision.probabilities.get("true", 0.0), "decision": decision}

    def warmup(self, row: dict) -> Decision:
        """Discover this row's options' real token boundaries up front
        (one batched request for everything not already cached), seed
        the speculative cache, then run `decide()`. See docs/SPEC.md
        §9.4."""
        if self._cache is not None:
            uncached = [o["id"] for o in row["options"] if self._cache.get(self.model, o["id"]) is None]
            if uncached:
                try:
                    discovered = self.backend.discover_tokens_batch(self.model, uncached)
                    for option_id, tokens in discovered.items():
                        if tokens:
                            self._cache.set(self.model, option_id, tokens)
                except Exception:
                    # Discovery is a pure optimization -- decide() below
                    # still works correctly without it.
                    pass
            self._cache.save()
        else:
            self.backend.warmup(self.model)
        return self.decide(row)

    def _decide_prefix(self, row: dict) -> Decision:
        """Score each option by walking its own id text, one real token
        at a time (see docs/PREFIX_MATCHING.md). Stops the moment a
        candidate has no remaining competition for its current prefix
        (SPEC.md §6.1 step 4 / §6.4) -- that candidate is scored on its
        logprob_sum as accumulated so far rather than walked to the end
        of its own id."""
        candidates = [Candidate(option_id=opt["id"], remaining=opt["id"]) for opt in row["options"]]
        raw_answer: str | None = None
        exceeded_depth = True

        # Speculation (SPEC.md §9.3): gated one round at a time on real
        # confirmation, never speculating past a round nothing has
        # confirmed yet. `speculative` maps a not-yet-requested prefix to
        # an already-in-flight future for it.
        speculative: dict[str, object] = {}
        confirmed_tokens: dict[str, list[str]] = {}

        def fire(consumed: str):
            return self._pool.submit(self._chat, build_prefix_messages(row, consumed))

        for depth in range(MAX_DEPTH):
            groups = group_by_context(candidates)
            if not groups:
                exceeded_depth = False
                break

            # Stop-early default: a group down to exactly one candidate
            # has no remaining competition -- mark it and skip the request.
            for consumed in list(groups):
                group = groups[consumed]
                if len(group) == 1:
                    group[0].stopped_early = True
                    del groups[consumed]
            if not groups:
                exceeded_depth = False
                break

            items = list(groups.items())
            futures = [speculative.pop(consumed, None) or fire(consumed) for consumed, _ in items]
            responses = [f.result() for f in futures]

            for i, (consumed, group) in enumerate(items):
                resp = responses[i]
                entries = resp.get("logprobs") or []
                if not entries:
                    for c in group:
                        c.unscored_reason = "server returned no logprobs for this step"
                    continue
                if depth == 0 and i == 0:
                    raw_answer = resp.get("content")

                found = _found_tokens(entries[0])
                for c in group:
                    before = c.consumed
                    match_step([c], found)
                    if c.consumed == before:
                        continue
                    tokens = confirmed_tokens.setdefault(c.option_id, [])
                    tokens.append(c.consumed[len(before):])

                    if self._cache is None or c.done:
                        continue
                    predicted = self._cache.get(self.model, c.option_id)
                    if not predicted:
                        continue
                    observed = confirmed_tokens[c.option_id]
                    still_on_track = all(observed[idx] == predicted[idx] for idx in range(len(observed)))
                    if not still_on_track or len(observed) >= len(predicted):
                        continue
                    next_token = predicted[len(observed)]
                    next_prefix = c.consumed + next_token
                    if next_prefix not in speculative:
                        speculative[next_prefix] = fire(next_prefix)

        if exceeded_depth:
            for c in candidates:
                if not c.done:
                    c.unscored_reason = f"exceeded max disambiguation depth ({MAX_DEPTH})"

        if self._cache is not None:
            for c in candidates:
                if c.unscored_reason is None and not c.stopped_early and c.remaining == "":
                    tokens = confirmed_tokens.get(c.option_id)
                    if tokens:
                        self._cache.set(self.model, c.option_id, tokens)
            self._cache.save()

        logprobs: dict[str, float] = {}
        unscored: list[str] = []
        stopped_early: list[str] = []
        for c in candidates:
            if c.unscored_reason is None:
                logprobs[c.option_id] = c.logprob_sum
                if c.stopped_early:
                    stopped_early.append(c.option_id)
            else:
                unscored.append(c.option_id)

        if not logprobs:
            reasons = "; ".join(f"{c.option_id}: {c.unscored_reason}" for c in candidates)
            raise DecisionError(f"could not score any option: {reasons}")

        ids = list(logprobs)
        probs = softmax([logprobs[i] for i in ids], self.temperature)
        probabilities = dict(zip(ids, probs))
        choice = max(probabilities, key=probabilities.get)

        return Decision(
            id=row.get("id", ""),
            choice=choice,
            probabilities=probabilities,
            logprobs=logprobs,
            mode="prefix",
            unscored=unscored,
            stopped_early=stopped_early,
            raw_answer=raw_answer,
        )

    def _decide_tree(self, row: dict) -> Decision:
        """Descend the id hierarchy one level at a time (see
        docs/HIERARCHY.md). Independent branches at one node are explored
        concurrently on the shared pool (SPEC.md §11.1)."""
        probabilities: dict[str, float] = {}
        eliminated: list[str] = []
        unscored: list[str] = []
        stopped_early: list[str] = []
        raw_answer_holder: list[str | None] = [None]

        def explore(node, path_logprob: float) -> None:
            if len(node.options) == 1:
                probabilities[node.options[0]["id"]] = math.exp(path_logprob)
                return
            if len(node.children) > MAX_BRANCHES_PER_LEVEL:
                raise DecisionError(
                    f"level {node.segment or '(root)'!r} has {len(node.children)} branches, over the "
                    f"limit of {MAX_BRANCHES_PER_LEVEL} -- add another id segment to split it further"
                )
            if len(node.children) == 1:
                [only_child] = node.children.values()
                explore(only_child, path_logprob)
                return

            summaries = node.child_summaries()
            sub_row = {
                "id": f"{row['id']}#{node.segment or 'root'}",
                "state": row["state"],
                "question": row["question"],
                "options": [{"id": seg, "description": desc} for seg, desc in summaries.items()],
            }
            d = self._decide_prefix(sub_row)
            if raw_answer_holder[0] is None:
                raw_answer_holder[0] = d.raw_answer
            if d.choice not in node.children:
                raise DecisionError(f"row {row['id']!r}: internal error, chose an unknown branch {d.choice!r}")

            to_explore: list[tuple] = []
            for seg, child in node.children.items():
                if seg in d.unscored:
                    unscored.extend(o["id"] for o in child.options)
                    continue
                if seg in d.stopped_early:
                    stopped_early.extend(_leaf_ids(child))
                branch_logprob = path_logprob + math.log(d.probabilities[seg])
                is_winner = seg == d.choice
                is_free_leaf = len(child.options) == 1
                if is_winner or self.exhaustive or is_free_leaf:
                    to_explore.append((child, branch_logprob))
                else:
                    eliminated.extend(o["id"] for o in child.options)

            self._map(lambda args: explore(*args), to_explore)

        explore(build_tree(row["options"]), 0.0)

        if not probabilities:
            raise DecisionError(f"row {row['id']!r}: could not score any option in the hierarchy")

        choice = max(probabilities, key=probabilities.get)
        return Decision(
            id=row["id"],
            choice=choice,
            probabilities=probabilities,
            logprobs={k: math.log(v) for k, v in probabilities.items()},
            mode="prefix",
            unscored=unscored,
            eliminated=eliminated,
            stopped_early=stopped_early,
            raw_answer=raw_answer_holder[0],
        )
