# Naming option ids

The required format for `row["options"][i]["id"]`, why each rule exists, and what to do if your natural ids don't fit it.

## The required format

Lowercase letters and digits, in segments joined by single underscores. No leading, trailing, or doubled underscores, no other punctuation, no spaces, no mixed case.

```python
# good
{"id": "billing", "description": "Charges, invoices, payment problems"}
{"id": "access_denied", "description": "Login rejected due to permissions"}
{"id": "billing_refund", "description": "Customer wants a refund"}

# rejected -- format
{"id": "Billing-Refund", "description": "..."}      # uppercase, hyphen
{"id": "access__denied", "description": "..."}       # doubled underscore
{"id": "_billing", "description": "..."}             # leading underscore

# rejected -- too long
{"id": "the_customer_was_denied_access_due_to_an_expired_authentication_token", "description": "..."}

# rejected -- too short
{"id": "a", "description": "..."}
```

This isn't just tidiness. Each underscore-separated segment is a real level of a hierarchy (`billing_refund` is level-1 `billing`, level-2 `refund`) — see [HIERARCHY.md](HIERARCHY.md) for how that's used to keep large option sets from all racing in one shot. A format the library can rely on structurally is what makes that possible without asking the caller to declare the hierarchy separately.

## Why length matters

Every option is walked toward the end of its own id, one real token at a time, stopping early only once nothing else is still competing for its prefix (see [PREFIX_MATCHING.md](PREFIX_MATCHING.md)). A short id like `billing` usually resolves in one round. A long one can take several, bounded by `MAX_DEPTH` (6 rounds). Push past that bound and the option is marked `unscored` — not because it was wrong, but because resolving it would have taken more rounds than the library is willing to spend chasing one option.

`decidr` rejects any option id longer than 40 characters, at `decide()`-call time, before any request is sent:

```
DecisionError: option id(s) too long: ['a_really_long_descriptive_slug_like_this_one'] (max 40 chars). See docs/NAMING_IDS.md.
```

40 characters is generous relative to real option ids: every id `decidr`'s own tests and examples use is 14 characters or fewer (`access_denied`, `billing_refund`). It's meant to catch ids that were clearly never meant to be an *id* — a sentence, a description, a path — not to be a tight budget real category names bump against.

`decidr` also rejects any option id shorter than 2 characters:

```
DecisionError: option id(s) shorter than 2 characters: ['a'] -- a single character is too likely to collide with another option's first token or a common filler token in the race. See docs/NAMING_IDS.md.
```

A single character is too likely to *be* a whole token, or to sit inside a much more common token than the option was meant to represent — either way it competes poorly for space in the top-20 `top_logprobs` window against unrelated common tokens, for reasons that have nothing to do with whether it's the right answer. Two characters is the cheap, static floor; it doesn't guarantee good token boundaries (nothing short of a live tokenizer probe could), but it rules out the worst case for free.

### Choosing ids for a fast, single-pass decision

If every option in a row genuinely stands on its own — no natural grouping, no shared category — give them ids with **no shared segments and no shared prefixes**, even short ones. `decidr` degrades to exactly one flat race when there's nothing to descend into, and one race is one round: `billing`, `access`, `outage` resolve as fast as the model can answer, no hierarchy overhead.

The opposite mistake is accidentally creating a hierarchy you didn't intend, by giving otherwise-unrelated options a shared leading segment. `lv0`, `lv1`, `lv2` race directly against each other in one round; `level_0`, `level_1`, `level_2` don't — the shared `level` segment makes `build_tree` collapse them under one artificial parent first (a free, no-request descent, since it's the only child of the root), then race `0` vs `1` vs `2` as a *second* round underneath it. The extra round isn't wrong, but it's pure overhead for options that were never really nested to begin with — reserve a shared segment for options that are actually grouped by something real (`billing_refund`/`billing_dispute` both being billing issues), not as an incidental naming convention.

If your natural categories genuinely are nested (department → sub-category, severity level → specific cause), keep the shared segment — that's exactly what the hierarchy is for.

## Ids can't nest inside each other

`billing` and `billing_refund` can't both be options in the same `decide()` call. It's genuinely ambiguous which one is meant: is `billing` a standalone answer distinct from `billing_refund`, or is it the umbrella every `billing_*` option (including `billing_refund`) belongs under? `decidr` doesn't guess:

```
DecisionError: option id 'billing' is a prefix of 'billing_refund' -- ambiguous as a hierarchy path. See docs/NAMING_IDS.md.
```

If `billing` genuinely needs to be its own answer separate from more specific billing sub-categories, give it a more specific id too (`billing_general`) rather than leaving it as a bare prefix of the others.

## If your natural ids don't fit this shape

Rename the id, keep the full meaning in `description`. `Access Denied (expired token)` becomes `id: "access_expired"`, `description: "Login rejected because the authentication token had expired"`. The model reads the description either way — the id is what gets matched against, not what carries meaning.

## What the format check doesn't catch

There's no live check against the model's real tokenizer before running a decision, because there is no general way to get one for an arbitrary provider (no endpoint exposes tokenization, and hosted APIs don't expose one either). The 40-character cap, the 2-character floor, and the segment format are static, cheap, tokenizer-independent proxies for "this is likely to be expensive, ambiguous, or fragile," not a guarantee either way. A 39-character id built entirely from rare subword fragments could still hit `MAX_DEPTH`; `unscored` and `MAX_DEPTH` are what actually resolve the question a live model has to answer.
