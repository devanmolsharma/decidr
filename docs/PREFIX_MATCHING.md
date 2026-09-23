# Prefix matching and chain-rule scoring

How `decidr` scores real option ids, why it needs more than one request for some inputs, and why it stops walking an option's id the moment nothing else is still competing for that same prefix.

Code: [`src/decidr/prefix.py`](../src/decidr/prefix.py). This doc explains the reasoning behind it; the module docstring is the shorter version. Shared mechanism with the [TypeScript port](https://github.com/devanmolsharma/decidr-ts) — both built against the same [cross-language spec](https://github.com/devanmolsharma/decidr-ts/blob/main/docs/SPEC.md).

## The problem this solves

The straightforward way to read an option's probability off a model is to read the logprob of its id's first token. That breaks the moment an id isn't a single token — and most real option ids aren't (`access_denied` is several pieces in every tokenizer checked). A model's returned logprobs only cover the position it was actually asked about, so a multi-token id needs its own mechanism to be scored as a whole, not just its first piece.

That means finding out, for an arbitrary model, how a specific string tokenizes — and there is no general way to ask most providers that directly (no public Ollama endpoint exposes tokenization, and hosted APIs don't expose one either).

## The mechanism

1. **Ask, read the alternatives.** The model is given the option ids in the prompt and asked to answer with one of them. The next position's `top_logprobs` is the discovery mechanism — it's what reveals a real token boundary, since there's no other source for one. There's no way to name the exact token to look for in advance (that's the whole reason this mechanism exists at all), so it always reads a rank window, never a requested set.

2. **Match by prefix.** For every option still unresolved, check whether any returned token is a prefix of what's left to match for that option (`remaining`). A match consumes that many characters and adds the token's logprob to a running total.

   The match has to be a genuine prefix, checked in one direction only: the returned token must not extend past the option's remaining text. `none` is not accepted as a match for an option expecting `no` — that would silently misread a different word as a longer match.

   When more than one returned token would validly match, the longest one wins — it resolves more of the answer in the same step and is never wrong if it fits at all.

3. **Collisions batch into the next request.** If two or more options still have exactly the same consumed-so-far text, they're genuinely indistinguishable at this point and stay grouped (`group_by_context`). The next request repeats the original question with that shared prefix appended as the start of the assistant's answer (an `assistant` message containing the text so far), so the model continues from exactly where the group left off rather than restarting the whole answer.

   Options that already diverged from their group continue alone, each in their own request, from that point on.

4. **Stop the moment nothing else is still competing.** The instant an option is the only candidate left in its group — no other option shares its current consumed-so-far prefix anymore — `decidr` stops walking it right there and scores it on whatever `logprob_sum` it has accumulated so far. See "Stopping early: the actual default" below for what this costs and why it's the default anyway.

5. **Sum in log space, normalize once at the end.** Multiplying probabilities is summing their logs, so each option's `logprob_sum` is exactly the sum of the per-step matched logprobs — no separate combination step, no re-deriving the same chain rule differently. Once every option is either resolved (fully or stopped early), or has been marked unscored (a step found no matching continuation for it — see below), the collected sums are passed through the same `softmax` the rest of the library uses, so the reported probabilities sum to 1 over whatever was actually scored.

## Stopping early: the actual default

Here's the case for walking every option to completion instead, which is the strictly fair thing to do: a chain-rule probability accumulates by multiplication, `P(t1) * P(t2|t1) * P(t3|t1,t2) * ...`. Each additional term can only keep the running product the same or make it smaller (probabilities are ≤ 1). So an option that resolves in one token will, all else equal, end up with a *less negative* total log-probability than one that needed three tokens — not because it's semantically more likely, but purely because fewer terms were multiplied in. Comparing a one-token option's full probability against a three-token option's *partial* probability (stopped early, right after disambiguation) is comparing two different quantities and calling the result a fair distribution. It isn't one, strictly speaking.

`decidr`'s default trades that fairness guarantee away for speed. The moment an option is the sole survivor of its group, there's no other option left that it could still be confused with — so instead of paying for more requests to spell out the rest of its id, it's scored on its `logprob_sum` as accumulated up to that point and marked `stopped_early`. Measured live: an 8-option flat race with several multi-token, partially-similar ids (`delay`, `damaged`, `defect`, `delete`, `dispute`, `escalation`, `cancellation`, `verification`) dropped from 5 requests (walking `delete`, `dispute`, and `escalation` to completion, or to their own unscored failure) to **1 request** — every survivor stopped as soon as it stood alone.

What this means for the numbers: an option marked `stopped_early` in `Decision.stopped_early` has a real, genuine partial logprob — not a guess, not a gap (it's still in `probabilities`, not `unscored`) — but it isn't a strictly comparable full `P(id | prompt)` against an option that needed more rounds and got them. In practice this rarely changes which option wins — the survivor at each depth is still the one the model favored at that step — but a caller comparing exact probability values across options should check `stopped_early` first if it matters for their use case.

There is currently no flag to force full resolution instead — `Client(model, cache=False)` doesn't change this behavior (stopping early isn't part of the speculative cache).

## What happens when a step finds nothing

Every request is capped by `top_logprobs` (20, matching OpenAI's own documented hard limit — see [docs/PROVIDERS.md](PROVIDERS.md)). If, at some step, none of the returned alternatives match what an option needs next, that option is marked unscored right there rather than assigned a guessed value and left to keep going. This can happen at any depth, not just the first one: a longer id has more chances to hit a step where its needed continuation didn't make the window.

A depth cap (`MAX_DEPTH` in `prefix.py`, 6 rounds) exists as a safety valve for the case where an option never resolves and never gets marked unscored either — a pathological, non-terminating collision. Past that cap, anything still open is marked unscored rather than looping indefinitely.

## Cost in practice

Most real option sets don't share meaningful prefixes (`access`, `billing`, `shipping` diverge on their very first token), and resolve in exactly one request, with no cap on how many options there can be. The extra requests only happen for option sets that are deliberately similar (`access_denied` / `access_expired` / `access_revoked`), and even then, stopping early (above) means the request count tracks how many *rounds* were needed before every remaining candidate became a sole survivor, not how many options were in the collision, and not how long any individual id is.
