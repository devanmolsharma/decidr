"""Score option ids by their own text, not by a letter standing in for them.

Letter mode (core.py) sidesteps multi-token labels by asking about a letter
and putting the real meaning in a description. This module scores the ids
themselves, which removes the 16-option letter cap but costs more requests
for ids that share a prefix with another option.

There is no way to ask Ollama how a string tokenizes (checked: no public
endpoint exposes it, and the model file's embedded vocabulary isn't reachable
over the API either). So token boundaries are discovered the only way they
can be: by asking the model and reading what it actually returns.

Mechanism, one option at a time in principle, batched where possible:

  1. Ask the model to continue, and read the top alternatives at that
     position (top_logprobs, since logprob_tokens can't be used here --
     it needs an exact token string in advance, and we don't have one).
  2. For each option whose remaining (unmatched) text starts with one of
     the returned alternatives, consume that piece and add its logprob to
     a running total for that option.
  3. Options still sharing identical consumed-so-far text are still
     genuinely ambiguous and get asked about together in the next request,
     with that shared text appended as the start of the answer. Options
     that diverged continue alone.
  4. Repeat until every option's text is fully consumed (its full-sequence
     probability is complete) or a step finds no matching continuation
     for it, which makes it unscored.

Every option is walked to the end of its own id, not just until it is
distinguishable from the others. Stopping early would compare a short
option's one-token probability against a long option's partial probability,
which is not the same quantity -- shorter ids would look artificially more
likely purely for being shorter. The thing being compared is always
P(full option id | prompt), for every option, regardless of how many other
options happened to share its early characters.

That's the mechanism for one flat race. For option counts too large to
reliably race all at once (see docs/HIERARCHY.md for why no fixed
top_logprobs window scales with option count), `Client._decide_tree` in
core.py uses this same mechanism level by level: option ids are required to
be underscore-segmented (`billing_refund`, not a free-form string), and
`build_tree`/`TreeNode` below turn that segmentation into a real hierarchy.
Each node with more than one distinct next segment runs one race (via the
exact same `_decide_prefix` this module implements) over just those segment
values; a node with only one child needs no race at all. The full id's
probability is the product of every level's chosen-branch probability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

MAX_DEPTH = 6  # steps per still-unresolved option group; see resolve()'s docstring

PREFIX_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Respond with only the id of "
    "the single best option, exactly as given, with no explanation or reasoning."
)


@dataclass
class Candidate:
    option_id: str
    remaining: str          # text still to be matched; "" once fully resolved
    logprob_sum: float = 0.0
    consumed: str = ""      # text matched so far, used to group candidates by shared context
    unscored_reason: str | None = None

    @property
    def done(self) -> bool:
        return self.remaining == "" or self.unscored_reason is not None


def _is_content_block_list(state) -> bool:
    """`state` is a list of multimodal content blocks (each a dict with a
    "type" key) rather than an arbitrary JSON list to be serialized whole."""
    return isinstance(state, list) and all(isinstance(item, dict) and "type" in item for item in state)


def build_prefix_messages(row: dict, prefix: str = "") -> list[dict]:
    """`prefix`: text already emitted, appended as the start of the assistant's
    answer so the next request continues from exactly where the last one
    stopped, instead of re-asking from scratch.

    Spells the options out as plain text rather than a JSON array. A model
    reading a JSON payload tends to continue the JSON syntax itself (a
    quote, a bracket, an "options" or "answer" key) rather than treat an
    option's own text as the obvious next word, which starves every option
    but the most likely one of a fair top_logprobs showing.

    `row["state"]` may also be a list of multimodal content blocks (each
    `{"type": "text"|"image"|"video"|"audio", ...}`) instead of a string or
    plain JSON value -- see backend.py for how each `Backend` translates
    that into its provider's own wire format. The instructions (question +
    options line) are appended as a trailing text block rather than spliced
    into an existing one, so image/video/audio blocks stay intact.
    """
    options_line = ", ".join(opt["id"] for opt in row["options"])
    instructions = f"{row['question']}\nAnswer with exactly one of: {options_line}."

    state = row["state"]
    if _is_content_block_list(state):
        user_content = [*state, {"type": "text", "text": f"\n{instructions}"}]
    else:
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        user_content = f"{state_text}\n\n{instructions}"

    messages = [
        {"role": "system", "content": PREFIX_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    if prefix:
        messages.append({"role": "assistant", "content": prefix})
    return messages


def match_step(candidates: list[Candidate], found: dict[str, float]) -> None:
    """Advance each still-open candidate by whichever returned token is a
    genuine prefix of what's left to match. Mutates `candidates` in place.

    A returned token only counts if it does not overshoot the candidate's
    remaining text -- "none" is not treated as matching a candidate whose
    remaining text is "no", because that is a different word, not the same
    answer with extra characters. Longest matching token wins when more
    than one returned token would fit, since a longer real match consumes
    more of the answer in one step and is never wrong if it fits at all.
    """
    for c in candidates:
        if c.done:
            continue
        best_token, best_logprob = None, None
        for token, logprob in found.items():
            if token and c.remaining.startswith(token):
                if best_token is None or len(token) > len(best_token):
                    best_token, best_logprob = token, logprob
        if best_token is None:
            c.unscored_reason = (
                f"no returned continuation matched the remaining text {c.remaining!r} "
                f"(had {sorted(found)!r} to choose from)"
            )
            continue
        c.consumed += best_token
        c.remaining = c.remaining[len(best_token):]
        c.logprob_sum += best_logprob


def group_by_context(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    """Candidates that still need another step, grouped by the text they've
    each consumed so far. Same consumed text means the same request would be
    sent for either of them, so ask about both in one request instead of two."""
    groups: dict[str, list[Candidate]] = {}
    for c in candidates:
        if not c.done:
            groups.setdefault(c.consumed, []).append(c)
    return groups


@dataclass
class TreeNode:
    """One level of the id hierarchy. `options` holds every option whose id,
    split on "_", passes through this exact node. A node with more than one
    distinct next segment among its options needs a real decision to pick
    which child to descend into; a node with exactly one needs none -- the
    id already committed to that branch, so there's nothing to ask."""
    segment: str                                  # this node's own segment ("" for the root)
    options: list[dict] = field(default_factory=list)  # every option whose id passes through here
    children: dict[str, "TreeNode"] = field(default_factory=dict)  # next segment -> subtree

    def child_summaries(self) -> dict[str, str]:
        """One synthesized description per child branch, built from the
        descriptions of every option still reachable under it -- there is no
        single description for a branch that covers several real options, so
        this is the model's only source of what each branch actually means."""
        strip = len(self.segment) + 1 if self.segment else 0  # +1 skips this node's own "_"
        out = {}
        for seg, node in self.children.items():
            parts = [f"{o['id'][strip:]}: {o['description']}" for o in node.options]
            out[seg] = "; ".join(parts)
        return out


def build_tree(options: list[dict]) -> TreeNode:
    """Build the id hierarchy from every option's id, split on "_". Each
    option must appear at exactly the leaf its full id addresses; two
    options should never fully collide (validate_row's uniqueness check
    already guarantees distinct ids, so this only fails if segmentation
    itself is inconsistent, which the id-format check also rules out)."""
    root = TreeNode(segment="")
    for opt in options:
        node = root
        node.options.append(opt)
        prefix = ""
        for seg in opt["id"].split("_"):
            prefix = seg if not prefix else f"{prefix}_{seg}"
            node = node.children.setdefault(seg, TreeNode(segment=prefix))
            node.options.append(opt)
    return root
