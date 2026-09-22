"""Typed semantic decisions against any local Ollama model, in one forward pass.

Send state + question + options, get probabilities back. No answer sentence, no
JSON to repair, no decoding loop -- the answer is read out of the option tokens'
log probabilities at a single position.

Rows use the same shape other single-pass decision engines accept, so they port
between implementations:

    {"id": "route-1",
     "state": "Customer cannot access an account after a password reset.",
     "question": "Which queue should handle this request?",
     "options": [{"id": "access",  "description": "Account access support."},
                 {"id": "billing", "description": "Billing support."}]}

Option ids are scored by their own text -- real ids like "billing" or
"access_denied", never a stand-in letter -- read out of the model's own
logits, one real token at a time (see prefix.py's module docstring for the
matching mechanism, docs/HIERARCHY.md for how ids above a handful of
options are resolved via a hierarchy instead of one flat race).

Scores are read from `top_logprobs`, which is a rank window (Ollama's own
cap of 20), not a requested set: a candidate that doesn't rank inside it is
reported as unscored rather than guessed at, instead of being silently
missing from `probabilities`. See docs/HIERARCHY.md for how large option
sets are kept from hitting this in practice.

`Client` talks to a model through a `Backend` (see backend.py):
`OllamaBackend` by default, needing nothing beyond the standard library, or
`LiteLLMBackend` for any other provider LiteLLM supports, as an optional
extra. Nothing in this module knows or cares which one is in use -- both
return the same normalized `{"content", "logprobs"}` shape.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Literal

from .backend import Backend, DecisionError, OllamaBackend
from .prefix import MAX_DEPTH, Candidate, build_prefix_messages, build_tree, group_by_context, match_step

DEFAULT_HOST = "http://127.0.0.1:11434"


@dataclass
class Decision:
    id: str
    choice: str                       # option id with the highest probability
    probabilities: dict[str, float]   # option id -> probability, sums to 1 over scored options
    logprobs: dict[str, float]        # option id -> raw logprob, before normalizing
    mode: Literal["prefix"]           # how the scores were read; always "prefix" -- see module docstring
    unscored: list[str] = field(default_factory=list)  # options the server could not report
    eliminated: list[str] = field(default_factory=list)  # options under a hierarchy branch that lost a real race
    raw_answer: str | None = None     # the model's own first reply

    @property
    def confidence(self) -> float:
        return self.probabilities.get(self.choice, 0.0)

    def is_reliable(self) -> bool:
        """False when some option went unscored, so the distribution is incomplete
        and `probabilities` is normalized over a subset of what was asked.

        Not affected by `eliminated`: an option under a hierarchy branch that
        lost a real, fair race against its sibling branches is not a
        measurement gap, it's the mechanism correctly ruling it out. See
        docs/HIERARCHY.md."""
        return not self.unscored


# Above this, an id costs more disambiguation rounds than MAX_DEPTH allows in
# the worst case (heavy subword fragmentation -- numbers, punctuation, and
# rare words often split into many short pieces), and would land in
# `unscored` for a reason that had nothing to do with whether it was the
# right answer. See docs/NAMING_IDS.md.
MAX_ID_LENGTH = 40

# Segments of lowercase letters/digits, joined by single underscores. This is
# the opinionated format prefix mode requires: it's what lets an id double as
# a hierarchy path ("billing_refund" -> level 1 "billing", level 2 "refund")
# for options above ID_SEGMENT_LIMIT. No leading/trailing/double underscores,
# no other punctuation -- those would make segment boundaries ambiguous.
# See docs/NAMING_IDS.md.
ID_FORMAT = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")

# Above this many distinct branches at one hierarchy level, that level itself
# needs the same treatment the old flat cap existed for: too many things
# racing at once for top_logprobs's window to see them all -- see
# docs/HIERARCHY.md.
MAX_BRANCHES_PER_LEVEL = 16


def validate_row(row: dict, check_id_format: bool = True) -> None:
    """Reject a malformed row up front, with a message naming what's wrong --
    a bad row should fail here, not produce a confident-looking wrong answer.
    `check_id_format` defaults on; `decide()` always calls with it on for a
    caller's own row. Sub-races `_decide_tree` builds internally (segment
    values, not real option ids) go straight to `_decide_prefix` and never
    pass through this check at all, format rules only ever apply to what a
    caller actually supplies."""
    required = {"id", "state", "question", "options"}
    if not required <= row.keys():
        raise DecisionError(f"row is missing fields: {sorted(required - row.keys())}")
    if not all(isinstance(row[k], str) and row[k] for k in ("id", "question")):
        raise DecisionError("id and question must be nonempty strings")
    state = row["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise DecisionError("state must be a nonempty string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise DecisionError("state must be finite JSON-compatible data") from e
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
            raise DecisionError(
                f"option id(s) too long: {too_long!r} (max {MAX_ID_LENGTH} chars). A long id "
                "needs more disambiguation rounds than MAX_DEPTH allows and can land in "
                "`unscored` for a reason unrelated to whether it was the right answer -- "
                "shorten it and put the full name in that option's description instead. "
                "See docs/NAMING_IDS.md."
            )
        badly_shaped = [i for i in ids if not ID_FORMAT.match(i)]
        if badly_shaped:
            raise DecisionError(
                f"option id(s) don't match the required format: {badly_shaped!r}. "
                "Ids must be lowercase letters/digits in underscore-separated segments "
                "(e.g. \"billing_refund\"), with no leading, trailing, or doubled underscores "
                "and no other punctuation. This is what lets an id double as a hierarchy path "
                "for large option sets -- see docs/NAMING_IDS.md."
            )
        # "billing" and "billing_refund" together are ambiguous: is "billing"
        # a standalone answer, or the umbrella every other option nests
        # under? Rather than guess, require ids that don't nest inside each
        # other -- every id is either fully distinct from every other id's
        # segments, or one continues past where another one ends, never both.
        segments = {i: i.split("_") for i in ids}
        for a, a_segs in segments.items():
            for b, b_segs in segments.items():
                if a != b and b_segs[:len(a_segs)] == a_segs:
                    raise DecisionError(
                        f"option id {a!r} is a prefix of {b!r} -- ambiguous as a hierarchy path "
                        f"(is {a!r} its own answer, or does everything under it belong to {b!r}?). "
                        "Give the shorter one a more specific id, or add a sibling under it "
                        "instead of leaving it as a leaf itself. See docs/NAMING_IDS.md."
                    )


def softmax(logprobs: list[float], temperature: float = 1.0) -> list[float]:
    """Normalize over the supplied options only. The result is conditional on
    this option set -- it says nothing about tokens outside it."""
    if not logprobs:
        return []
    scaled = [lp / temperature for lp in logprobs]
    top = max(scaled)
    exp = [math.exp(v - top) for v in scaled]
    total = sum(exp)
    return [v / total for v in exp] if total else [1.0 / len(exp)] * len(exp)


class Client:
    """Talks to a model through a `Backend` -- `OllamaBackend` by default."""

    def __init__(self, model: str, host: str = DEFAULT_HOST, timeout: float = 120.0,
                 temperature: float = 1.0, backend: Backend | None = None, exhaustive: bool = True):
        self.model = model
        self.temperature = temperature
        self.backend = backend if backend is not None else OllamaBackend(host=host, timeout=timeout)
        # Explore every hierarchy branch (real probabilities for every
        # option, more requests) instead of only the winning path
        # (fewer requests, losing branches with unexplored children go to
        # `eliminated` instead of getting a real number). See docs/HIERARCHY.md.
        self.exhaustive = exhaustive

    # ---- transport ---------------------------------------------------------

    def _chat(self, messages: list[dict]) -> dict:
        return self.backend.chat(self.model, messages)

    @staticmethod
    def _found_tokens(entry: dict) -> dict[str, float]:
        """{token: logprob} out of one response position's top_logprobs, which
        is a rank window, not a requested set -- a token can be common enough
        to matter and still miss it."""
        found: dict[str, float] = {}
        if entry.get("token"):
            found[entry["token"].strip()] = entry.get("logprob", 0.0)
        for alt in entry.get("top_logprobs", []):
            found.setdefault(alt["token"].strip(), alt["logprob"])
        return found

    # ---- decisions ---------------------------------------------------------

    def decide(self, row: dict) -> Decision:
        validate_row(row, check_id_format=True)
        return self._decide_tree(row)

    def _decide_prefix(self, row: dict) -> Decision:
        """Score each option by walking its own id text, one real token at a
        time, discovered from what the model actually returns rather than a
        precomputed tokenizer (there is no way to get one for an arbitrary
        Ollama model over the API -- see prefix.py's module docstring).

        Every option is walked to the end of its own id, even once it no
        longer shares a prefix with anything else, so every final score is
        a genuine P(full id | prompt), not a truncated one that would look
        artificially more likely just for being short.
        """
        candidates = [Candidate(option_id=opt["id"], remaining=opt["id"]) for opt in row["options"]]
        raw_answer = None

        for depth in range(MAX_DEPTH):
            groups = group_by_context(candidates)
            if not groups:
                break
            for consumed, group in groups.items():
                resp = self._chat(build_prefix_messages(row, prefix=consumed))
                entries = resp.get("logprobs") or []
                if not entries:
                    for c in group:
                        c.unscored_reason = "server returned no logprobs for this step"
                    continue
                if depth == 0 and raw_answer is None:
                    raw_answer = resp.get("content")
                found = self._found_tokens(entries[0])
                match_step(group, found)
        else:
            for c in candidates:
                if not c.done:
                    c.unscored_reason = f"exceeded max disambiguation depth ({MAX_DEPTH})"

        logprobs = {c.option_id: c.logprob_sum for c in candidates if c.remaining == ""}
        unscored = [c.option_id for c in candidates if c.remaining != ""]

        if not logprobs:
            raise DecisionError(
                f"row {row['id']!r}: none of the option ids could be fully resolved. "
                f"reasons: {[(c.option_id, c.unscored_reason) for c in candidates]}"
            )

        ids = list(logprobs)
        probs = softmax([logprobs[i] for i in ids], self.temperature)
        probabilities = dict(zip(ids, probs))
        return Decision(
            id=row["id"],
            choice=max(probabilities, key=probabilities.get),
            probabilities=probabilities,
            logprobs=logprobs,
            mode="prefix",
            unscored=unscored,
            raw_answer=raw_answer,
        )

    def _decide_tree(self, row: dict) -> Decision:
        """Descend the id hierarchy (see docs/NAMING_IDS.md for the required
        format, docs/HIERARCHY.md for the full mechanism) one level at a
        time. A node with only one child needs no race -- the id already
        committed to that branch. A node with several runs exactly one
        `_decide_prefix` race over that level's distinct segment values, real
        ids resolved by the same token-walking mechanism used everywhere
        else in this mode, just applied to segment values instead of full
        ids.

        Whether a losing branch gets explored further depends on
        `self.exhaustive` and how much it would cost:

          - A losing branch that's already a single leaf is explored for
            free regardless of `exhaustive` -- its probability comes
            straight out of the race that already ran, no extra request.
          - A losing branch with its own unexplored children costs a real
            request to look inside. With `exhaustive=True` (the default),
            every branch gets explored this way, so every option ends up
            with a real, comparable probability. With `exhaustive=False`,
            only the winning branch is explored past this point, and
            everything else goes to `eliminated` -- fewer requests, but
            those options' true probabilities were never measured.

        A branch whose segment value never showed up in a race's results at
        all goes to `unscored` regardless of `exhaustive`: that's a genuine
        measurement gap, not a decision about how much to explore.

        This degrades gracefully for flat (non-hierarchical) option sets:
        an id with no underscores is a one-segment path, so the root's
        children already are full ids, and this runs exactly one race,
        identical to calling `_decide_prefix` directly.
        """
        probabilities: dict[str, float] = {}
        eliminated: list[str] = []
        unscored: list[str] = []
        raw_answer_holder: list[str | None] = [None]

        def explore(node, path_logprob: float) -> None:
            if len(node.options) == 1:
                probabilities[node.options[0]["id"]] = math.exp(path_logprob)
                return
            if len(node.children) > MAX_BRANCHES_PER_LEVEL:
                raise DecisionError(
                    f"row {row['id']!r}: {len(node.children)} distinct branches at hierarchy "
                    f"level {node.segment!r} exceeds the {MAX_BRANCHES_PER_LEVEL} that can "
                    "reliably race at once. Add another underscore-separated level to these "
                    "ids to split the branching further. See docs/NAMING_IDS.md."
                )
            if len(node.children) == 1:
                [only_child] = node.children.values()  # only one branch possible -- descend for free
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

            for seg, child in node.children.items():
                if seg in d.unscored:
                    unscored.extend(o["id"] for o in child.options)
                    continue
                branch_logprob = path_logprob + math.log(d.probabilities[seg])
                if seg == d.choice or self.exhaustive or len(child.options) == 1:
                    explore(child, branch_logprob)
                else:
                    # A losing branch with its own children we're choosing
                    # not to pay for: which leaf inside it would have won,
                    # or at what probability, was never measured.
                    eliminated.extend(o["id"] for o in child.options)

        explore(build_tree(row["options"]), 0.0)

        if not probabilities:
            raise DecisionError(f"row {row['id']!r}: none of the option ids could be resolved.")

        choice = max(probabilities, key=probabilities.get)
        return Decision(
            id=row["id"],
            choice=choice,
            probabilities=probabilities,
            logprobs={k: math.log(v) for k, v in probabilities.items()},
            mode="prefix",
            unscored=unscored,
            eliminated=eliminated,
            raw_answer=raw_answer_holder[0],
        )

    def decide_all(self, rows: list[dict]) -> list[Decision]:
        return [self.decide(row) for row in rows]
