# Hierarchy resolution for large option sets

Why a flat request doesn't scale with option count, what the id hierarchy does instead, and the real tradeoff it makes.

Code: [`src/decidr/prefix.py`](../src/decidr/prefix.py) (`build_tree`, `TreeNode`), [`src/decidr/core.py`](../src/decidr/core.py) (`Client._decide_tree`). This doc explains the reasoning; the code comments are the shorter version.

## The problem, measured

`top_logprobs` is a rank window — Ollama returns the top 20 tokens by probability at a position, nothing else. When a decision only has a handful of options, all of them plausibly land inside that window. When it has many, most of them don't, and it has nothing to do with whether they're right.

Measured directly against `qwen3.5:4b`, asking which of 30 genuinely distinct support categories fits a login error, in one flat request:

| `top_logprobs` | Categories that appeared at all |
|---|---:|
| 20 (Ollama's cap) | 5 / 30 |
| 100 | 12 / 30 |

Raising the window helped, but the tail didn't move much: by rank 90, logprobs were already around -13 — the model was that unlikely to say those words next in a 30-way free-for-all, not because they're bad categories, but because a handful of dominant candidates soak up nearly all the probability mass when everything competes at once. `shipping` and `refund` (both real, sensible categories) sat at rank 86 and 89. No realistic window size fixes this; it's a property of putting 30 things in one race, not a property of the window.

## What the hierarchy does instead

Option ids are required to be underscore-segmented (`billing_refund`, not a free-form string — see [NAMING_IDS.md](NAMING_IDS.md)). That segmentation *is* a real hierarchy: `build_tree` turns the full set of ids into a tree where each level is one segment. Resolution descends that tree one level at a time:

- A node with only **one** child needs no request at all — the id already committed to that branch, so there's nothing to ask.
- A node with **more than one** child runs exactly one race (the same `_decide_prefix` token-walking mechanism described in [PREFIX_MATCHING.md](PREFIX_MATCHING.md)) over that level's distinct segment values, each shown with a synthesized description built from every leaf option still reachable under it.
- This repeats until a node has exactly one option left: the resolved answer.

```
options: access_denied, access_expired, access_revoked, billing_refund, billing_dispute

level 1: race "access" vs "billing"           -> "access" wins (1 request)
level 2: race "denied" vs "expired" vs "revoked" among access's children (1 request)
```

Measured on the same 30-category case, organized as a real two-level hierarchy: every option got a comparison against real siblings somewhere, and the deciding race at each level only ever had to distinguish a handful of things — exactly the option count that made 3-option and 8-option decisions reliable in the first place.

This degrades gracefully to exactly today's flat behavior for non-hierarchical ids: an id with no underscores is a one-segment path, so the root's children already are full ids, and only one race ever runs — identical to calling the flat single-race mechanism directly.

## The tradeoff, stated plainly

**Only options along the winning path, or that lost a race directly against something on that path, get a real probability.** An option under a branch that was never explored — because its parent branch lost at some level before we ever looked inside it — genuinely has an unmeasured probability. There's no honest number to report for it, because we never ran the comparison that would produce one.

This is why the library tracks two different things:

**`Decision.eliminated`** — options under a branch that lost a race, where that branch had unexplored children. We know it lost; we don't know what its own internal best answer would have scored, because we never asked.

**`Decision.probabilities`** — this includes more than just the final winner. Any losing branch that was *already a single leaf option* (no further children) gets a real, comparable probability too — computed as the product of confidences along its own root-to-leaf path, exactly the same way the eventual winner's number is computed. Two leaves that diverged at different points in the tree are still comparable this way, because both numbers represent the same kind of quantity: P(this exact leaf | the prompt), composed level by level. Only options whose branch was an *unexplored group* — where we stopped one level before deciding which specific leaf inside it was best — are missing a real number and go to `eliminated` instead.

Concretely, from a real run:

```python
probabilities: {'access_denied': 0.361, 'access_revoked': 0.027, 'access_expired': 0.612}
eliminated: ['billing_refund', 'billing_dispute']
```

`billing`'s two options lost the very first race (against `access`) without ever being individually compared to each other — we don't know which one *would* have won, or its score, so both are `eliminated`. All three `access_*` options, on the other hand, went through a real race against each other directly and all have comparable numbers.

`Decision.is_reliable()` checks `unscored` only (a genuine measurement gap — a segment value that never showed up in a race's results at all), never `eliminated`. A branch losing a real, complete race to a real peer is the hierarchy working as intended, not a failure.

## Exhaustive vs. cheap: `Client(exhaustive=...)`

By default (`exhaustive=True`), every branch gets explored, not just the winning path — a losing branch with its own children still gets raced, so its leaves end up with real, comparable probabilities instead of `eliminated`. This costs more requests (one per internal node in the whole tree, not just along one path) in exchange for a complete distribution.

`Client(model=..., exhaustive=False)` reverts to following only the winning path: a losing branch that's already a single leaf is still kept (it cost nothing extra — its number came from the race that just ran), but a losing branch with further children is left unexplored and its leaves go to `eliminated`.

Measured on the same 5-option, 2-level case:

| | `probabilities` | `eliminated` | requests |
|---|---:|---:|---:|
| `exhaustive=False` | 3 (the winning branch's own race) | 2 (the other branch's leaves) | 2 |
| `exhaustive=True` (default) | 4 (one option in the other branch fell outside that race's window and went to `unscored` instead) | 0 | 3 |

Neither setting invents a number for `unscored` — that field means the same thing regardless of `exhaustive`.

## Why branches come from the id, not an arbitrary batch size

An alternative to this design would chunk options into fixed-size groups regardless of what they mean, then run a single-elimination bracket. That has a real, known flaw: two genuinely strong options landing in the same early group means one of them is eliminated by the other despite being stronger than everything in every other group — a property of any single-elimination bracket over an arbitrary grouping.

The hierarchy avoids this by using structure the caller already has a reason to know: the taxonomy itself. `billing_refund` and `billing_dispute` aren't grouped together because of a chunk boundary — they're grouped because they're both under `billing`, a real category the caller chose. Branching reflects the actual shape of the decision space instead of an artificial batch size.

## Branch count per level

A node's own child count is still bounded — `MAX_BRANCHES_PER_LEVEL` (16) in `core.py`. If a single level has more distinct branches than that, `decide()` raises a clear error asking for another level of hierarchy rather than trying to force an oversized race through:

```
DecisionError: 22 distinct branches at hierarchy level '' exceeds the 16
that can reliably race at once. Add another underscore-separated level to
these ids to split the branching further.
```

This is a real constraint on how flat a taxonomy can be at any one level, not a soft suggestion — a level over the cap would just reproduce the original 30-category problem locally, one level down.
