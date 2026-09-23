"""The token-walking race mechanism. See docs/SPEC.md §6.1 for the
normative algorithm this file implements (shared verbatim with the
TypeScript port at decidr-ts/docs/SPEC.md -- this is the Python side of
the same cross-language spec)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

MAX_DEPTH = 6  # rounds allowed per still-open candidate before giving up on it

PREFIX_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Respond with only the id of "
    "the single best option, exactly as given, with no explanation or reasoning."
)


@dataclass
class Candidate:
    """One option's progress through a race. A candidate is `done` once
    its `remaining` text is fully consumed, it's been declared
    unscorable, or it stopped early with no remaining competition
    (SPEC.md §6.1 step 4)."""

    option_id: str
    remaining: str
    consumed: str = ""
    logprob_sum: float = 0.0
    unscored_reason: str | None = None
    stopped_early: bool = False

    @property
    def done(self) -> bool:
        return self.remaining == "" or self.unscored_reason is not None or self.stopped_early


def _is_content_block_list(state: object) -> bool:
    return isinstance(state, list) and all(isinstance(item, dict) and "type" in item for item in state)


def build_prefix_messages(row: dict, prefix: str = "") -> list[dict]:
    """Build the messages for one race step. `prefix`, if non-empty, is
    appended as the start of the assistant's answer so this request
    continues from exactly where a previous step left off."""
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
    """Advance each not-done candidate by whichever token in `found` is
    the longest genuine, non-overshooting prefix of what's left to
    match. Mutates `candidates` in place.

    Tokens are matched exactly as returned -- no whitespace stripping or
    other normalization. A provider's tokenizer may genuinely emit a
    token with leading/trailing whitespace as its own distinct unit, and
    altering it before matching risks a false match against a
    candidate's remaining text that the raw token would not have made."""
    for c in candidates:
        if c.done:
            continue
        best_token: str | None = None
        best_logprob = 0.0
        for token, logprob in found.items():
            if not token or not c.remaining.startswith(token):
                continue
            if best_token is None or len(token) > len(best_token):
                best_token, best_logprob = token, logprob
        if best_token is None:
            c.unscored_reason = (
                f'no returned token matched the next part of "{c.option_id}" '
                f'("{c.remaining}" remaining)'
            )
            continue
        c.consumed += best_token
        c.remaining = c.remaining[len(best_token):]
        c.logprob_sum += best_logprob


def group_by_context(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    """Group not-done candidates by their shared `consumed` prefix --
    same prefix means the same next question, asked once for the whole
    group instead of once per candidate. Order-preserving (dict
    preserves insertion order in Python 3.7+)."""
    groups: dict[str, list[Candidate]] = {}
    for c in candidates:
        if not c.done:
            groups.setdefault(c.consumed, []).append(c)
    return groups


@dataclass
class TreeNode:
    """One level of the id hierarchy. See docs/HIERARCHY.md §6.2."""

    segment: str
    options: list[dict] = field(default_factory=list)
    children: dict[str, "TreeNode"] = field(default_factory=dict)

    def child_summaries(self) -> dict[str, str]:
        """One synthesized description per child, built from every leaf
        option reachable under it, with this node's own segment path
        (and its trailing underscore) stripped from each leaf's id."""
        strip = len(self.segment) + 1 if self.segment else 0
        out: dict[str, str] = {}
        for seg, node in self.children.items():
            parts = [f"{o['id'][strip:]}: {o['description']}" for o in node.options]
            out[seg] = "; ".join(parts)
        return out


def build_tree(options: list[dict]) -> TreeNode:
    """Build the id hierarchy from every option's id, split on '_'."""
    root = TreeNode(segment="")
    for opt in options:
        node = root
        node.options.append(opt)
        path = ""
        for seg in opt["id"].split("_"):
            path = seg if not path else f"{path}_{seg}"
            child = node.children.get(seg)
            if child is None:
                child = TreeNode(segment=path)
                node.children[seg] = child
            child.options.append(opt)
            node = child
    return root
