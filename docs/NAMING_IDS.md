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
```

This isn't just tidiness. Each underscore-separated segment is a real level of a hierarchy (`billing_refund` is level-1 `billing`, level-2 `refund`) — see [HIERARCHY.md](HIERARCHY.md) for how that's used to keep large option sets from all racing in one shot. A format the library can rely on structurally is what makes that possible without asking the caller to declare the hierarchy separately.

## Why length matters

Every option is walked to the end of its own id, one real token at a time (see [PREFIX_MATCHING.md](PREFIX_MATCHING.md)). A short id like `billing` usually resolves in one round. A long one can take several, bounded by `MAX_DEPTH` (6 rounds). Push past that bound and the option is marked `unscored` — not because it was wrong, but because resolving it would have taken more rounds than the library is willing to spend chasing one option.

`decidr` rejects any option id longer than 40 characters, at `decide()`-call time, before any request is sent:

```
DecisionError: option id(s) too long: ['a_really_long_descriptive_slug_like_this_one']
(max 40 chars). A long id needs more disambiguation rounds than MAX_DEPTH
allows and can land in `unscored` for a reason unrelated to whether it was
the right answer -- shorten it and put the full name in that option's
description instead.
```

40 characters is generous relative to real option ids: every id `decidr`'s own tests and examples use is 14 characters or fewer (`access_denied`, `billing_refund`). It's meant to catch ids that were clearly never meant to be an *id* — a sentence, a description, a path — not to be a tight budget real category names bump against.

## Ids can't nest inside each other

`billing` and `billing_refund` can't both be options in the same `decide()` call. It's genuinely ambiguous which one is meant: is `billing` a standalone answer distinct from `billing_refund`, or is it the umbrella every `billing_*` option (including `billing_refund`) belongs under? `decidr` doesn't guess:

```
DecisionError: option id 'billing' is a prefix of 'billing_refund' -- ambiguous
as a hierarchy path (is 'billing' its own answer, or does everything under it
belong to 'billing_refund'?). Give the shorter one a more specific id, or add
a sibling under it instead of leaving it as a leaf itself.
```

If `billing` genuinely needs to be its own answer separate from more specific billing sub-categories, give it a more specific id too (`billing_general`) rather than leaving it as a bare prefix of the others.

## If your natural ids don't fit this shape

Rename the id, keep the full meaning in `description`. `Access Denied (expired token)` becomes `id: "access_expired"`, `description: "Login rejected because the authentication token had expired"`. The model reads the description either way — the id is what gets matched against, not what carries meaning.

## What the format check doesn't catch

There's no live check against the model's real tokenizer before running a decision, because there is no way to get one — see [PREFIX_MATCHING.md](PREFIX_MATCHING.md)'s "Why there's no shortcut through a tokenizer" section for what was actually checked (no `/api/tokenize` endpoint, no exposed vocabulary via `/api/show`, no reliable repo-name mapping) before concluding this. The 40-character cap and segment format are static, cheap, tokenizer-independent proxies for "this is likely to be expensive or ambiguous," not a guarantee either way. A 39-character id built entirely from rare subword fragments could still hit `MAX_DEPTH`; `unscored` and `MAX_DEPTH` are what actually resolve the question a live model has to answer.
