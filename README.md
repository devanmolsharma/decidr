# decidr

Typed decisions from an LLM, in one forward pass. Give it a question and a set of named options; get back which one the model picked and a real probability for each, without the model generating a single word.

![A real Choice + Score request running against Cerebras -- same mechanism as decidr, shown here in the TypeScript port's browser playground](https://raw.githubusercontent.com/devanmolsharma/decidr-ts/main/examples/webui/screenshots/playground-results.png)

**[Try it live](https://devanmolsharma.github.io/decidr-ts/)** -- the
playground above is the [TypeScript port](https://github.com/devanmolsharma/decidr-ts)'s
browser demo (no signup, paste your own API key or use a local Ollama
model); this repo is the Python reference implementation, same
mechanism, same numbers.

```bash
pip install decidr
```

- Skip to: [Quickstart](#quickstart-local-ollama) &middot;
  [Three primitives](#three-primitives-choice-score-noun) &middot;
  [API](#api) &middot; [Limitations](#limitations)
- TypeScript port: [decidr-ts](https://github.com/devanmolsharma/decidr-ts)
  on npm, same mechanism, same id rules, same hierarchy resolution, both
  built against the same [cross-language spec](https://github.com/devanmolsharma/decidr-ts/blob/main/docs/SPEC.md)

## Quickstart: local Ollama

Needs [Ollama](https://ollama.com) running with a model already pulled. No fine-tuning, no other setup.

```python
from decidr import Client

client = Client("qwen3.5:4b")  # talks to Ollama's OpenAI-compatible endpoint at http://127.0.0.1:11434/v1

decision = client.decide({
    "id": "ticket-1",
    "state": "Customer cannot access their account after a password reset. The reset email never arrived.",
    "question": "Which team should handle this?",
    "options": [
        {"id": "access",  "description": "Account access and authentication issues."},
        {"id": "billing", "description": "Billing and payment issues."},
        {"id": "sales",   "description": "Sales and product questions."},
    ],
})

decision.choice          # 'access'
decision.probabilities   # {'access': 0.9965, 'billing': 0.0017, 'sales': 0.0018}
```

From `decidr import confidence`: `confidence(decision)` reads `decision.probabilities[decision.choice]`. That's the whole thing: `state` is what the model reads, `question` is what it's deciding, `options` is the closed set of possible answers, each with an `id` you'll match on in your own code and a `description` for the model to read.

## Quickstart: a hosted model (OpenAI, or anything OpenAI-compatible)

Same `decide()` call, same `row` shape — only the `Client` construction changes:

```python
from decidr import Client, OpenAIBackend

client = Client("gpt-4o-mini", backend=OpenAIBackend(api_key="sk-..."))

decision = client.decide(row)  # same row shape as above
```

`OpenAIBackend` talks to any server that speaks the OpenAI `/v1/chat/completions` wire format — pass `base_url` to point it at a self-hosted or third-party endpoint instead of `https://api.openai.com/v1`. This is also what the default `Client(...)` above uses under the hood, just pointed at a local Ollama instead.

**Only some models support `logprobs`.** Standard chat models (the GPT-4o family and similar) do. Reasoning models (the o-series and similar reasoning models elsewhere) do not — `decide()` will fail with a clear error if you point it at one. See [docs/PROVIDERS.md](docs/PROVIDERS.md) for the full, live-checked list of which providers and models actually return real logprobs.

## Three primitives: Choice, Score, Noun

`decide()` (above) is the **Choice** primitive: pick the best option from a fixed set of ids. Two more primitives are built on top of it, same mechanism, different response shape:

```python
# Score: place the state on an ordered rubric, low to high.
severity = client.score({
    "id": "bug-1",
    "state": "The bug crashes the whole app for every user, no workaround.",
    "question": "How severe is this bug?",
    "levels": [
        {"id": "cosm", "description": "cosmetic"},
        {"id": "mod", "description": "moderate"},
        {"id": "crit", "description": "critical"},
    ],
})
severity["score"]     # e.g. 1.95 -- a probability-weighted position on [0, 2],
                       # not just the top level's index
severity["decision"]  # the underlying Decision, same probabilities/eliminated/etc

# Noun: a single true/false probability -- the number itself is the signal.
critical = client.truth({
    "id": "bug-1",
    "state": "The bug crashes the whole app for every user, no workaround.",
    "question": "Is this bug a critical severity issue?",
})
critical["truth"]  # e.g. 0.9862 -- read the same way confidence() is read
                    # elsewhere: the probability IS the answer, not just
                    # which side of 0.5 it falls on
```

Both are ordinary `decide()` calls under the hood (`score` races the levels as options and computes a weighted index; `truth` races a fixed `"true"`/`"false"` pair) — same id-format rules, same `unscored`/`eliminated`/`stopped_early` semantics apply to the underlying `Decision` either way.

## How option ids work

Options are identified by real, human-readable ids, not letters (`A`/`B`/`C`). The model reads and answers with the id itself, and the id's structure does double duty as your category hierarchy:

- Ids are lowercase alphanumeric segments joined by underscores: `billing_refund`, `bug_crash`, `account_login_2fa`. At least 2 characters — a single character is too likely to collide with another option's first token or a common filler token in the race.
- An id can't be a segment-prefix of another id in the same option set (`billing` and `billing_refund` together are rejected — ambiguous).
- Segments are used as a literal tree. `billing_refund` and `billing_dispute` both live under a `billing` node; deciding which applies is two small races (`billing` vs `bug` vs `account`, then `refund` vs `dispute`) instead of one large unreliable race over every leaf id at once.

By default, **every branch of the hierarchy is explored** (`exhaustive=True`), so every option ends up with a real, comparable probability — more requests, better numbers. Set `exhaustive=False` to only pay for the winning path; ids under branches that lost but weren't explored further show up in `decision.eliminated`, not `decision.probabilities`.

```python
client = Client("qwen3.5:4b", exhaustive=False)
```

Separately from `exhaustive`, `decidr` also stops walking an individual option's id the moment it's the only candidate left racing for its current prefix — there's nothing else it could still be confused with, so paying for more requests to spell out the rest of its id isn't worth it by default. That option is still scored (it's in `decision.probabilities`, not `decision.unscored`) and shows up in `decision.stopped_early`, but on a partial rather than a full log-probability — see [docs/PREFIX_MATCHING.md](docs/PREFIX_MATCHING.md) for what that trades away.

## Backends

`OpenAIBackend`, built on the official `openai` package, is the only backend — and it's enough. It talks to any OpenAI-compatible `/v1/chat/completions` endpoint: OpenAI itself, Ollama's own OpenAI-compatible endpoint (verified live to return real, correct `logprobs`), and other OpenAI-compatible hosts confirmed to forward `logprobs` correctly (see [docs/PROVIDERS.md](docs/PROVIDERS.md)). There is deliberately no separate Ollama-specific backend or multi-provider abstraction layer — [docs/PROVIDERS.md](docs/PROVIDERS.md#gateways-and-unified-multi-provider-clients-none-solve-this-reliably) has the full researched reasoning for why every general-purpose multi-provider client checked (LiteLLM, OpenRouter, Vercel AI SDK, LangChain, and others) has a silent or structural logprobs gap for at least one major provider.

You can write your own backend for another provider by subclassing `Backend` — just one method to implement, `chat`:

```python
from decidr import Backend

class MyBackend(Backend):
    def chat(self, model: str, messages: list[dict], max_tokens: int = 1) -> dict:
        # return {"content": ..., "logprobs": [...]} in the shape documented on Backend.chat
        ...
```

## Latency

Against a hosted API, a cold connection (fresh TLS handshake) is the single biggest cost you actually control. The fix is simple: warm the connection, and (optionally) the id tokenization, before the latency-sensitive call.

```python
# Ahead of time, once the row's options are known:
decision = client.warmup(row)  # pre-warms the connection AND discovers each
                                # option id's real token boundaries, seeding
                                # the speculative cache -- returns a real
                                # Decision, so this can just be your first call

# Later, on the hot path:
decision = client.decide(row)  # faster: warm connection, and (if warmup ran
                                # before) every disambiguation round fires
                                # from a verified guess instead of a cold one
```

`warmup` never changes what `decide()` returns — every speculative guess it seeds is still verified against the real response before being trusted (see [docs/PREFIX_MATCHING.md](docs/PREFIX_MATCHING.md)). It only changes how fast the answer arrives. `Client`'s `cache` option controls the underlying speculative cache (on by default, persisted to `~/.decidr/token-cache.json`) — pass `cache=False` to disable it.

Independent requests within one round, independent hierarchy branches, and separate rows passed to `Client.decide_all(rows)` are all sent concurrently on a shared, bounded thread pool (`Client(..., max_workers=16)` to tune it), not one at a time — `decide_all`'s wall-clock cost for N rows lands close to one row's latency rather than N times it.

**A real, measured caveat specific to this port:** at high fan-out (13
concurrent `truth()` calls in one batch), Python's GIL measurably serializes
the pure-Python bookkeeping each call does around its HTTP request — a
single `truth()` call is 166–193ms (matching decidr-ts), but 13 of them
fired concurrently on the thread pool land around 740–1330ms total, not
close to one call's latency the way decidr-ts's concurrent dispatch does.
Confirmed by isolating the raw HTTP layer: 13 concurrent `backend.chat()`
calls with no `decide()` logic around them complete in ~366ms, the same
range as decidr-ts — so the gap is specifically in this port's per-call
Python-side work under GIL contention, not the network or the pool itself.
Lower fan-out (a handful of concurrent rows) doesn't show this effect.

## Benchmarks

Measured live (median of 5 runs per cell, each isolated in its own
process to avoid cross-provider network contention), 10 realistic
scenarios (1–13 questions each, e.g. support-ticket triage, resume
screening, content moderation, code review), comparing this port against
[TypeSafe](https://console.typesafe.ai) (a comparable product that
answers several questions about one shared input in a single batched
request — decidr has no equivalent yet, see decidr-ts's
[BATCHING_DESIGN.md](https://github.com/devanmolsharma/decidr-ts/blob/main/docs/BATCHING_DESIGN.md)
for a real, not-yet-built design for one). Every decidr cell fires
`Client.truth()` concurrently across that scenario's questions via the
shared thread pool, one real HTTPS request per question; every TypeSafe
cell is its one real batched request:

| Scenario | Questions | decidr + Cerebras (`qwen-3.8-27b`) | TypeSafe (`jev-latest`) |
|---|---:|---:|---:|
| single-question-triage | 1 | 175ms | 255ms |
| billing-dispute | 13 | 1050ms | 231ms |
| resume-screen | 5 | 220ms | 260ms |
| content-moderation | 8 | 310ms | 191ms |
| medical-intake | 2 | 180ms | 218ms |
| legal-doc-review | 6 | 407ms | 236ms |
| code-review | 4 | 187ms | 237ms |
| single-question-fraud | 1 | 173ms | 307ms |
| email-routing | 3 | 195ms | 252ms |
| product-review-analysis | 10 | 367ms | 205ms |

At low question counts, this port's per-request floor is competitive
with (sometimes faster than) TypeSafe's batched call. At higher question
counts, the GIL-contention effect described above dominates and the gap
widens well past what decidr-ts sees on the identical scenarios and
provider — decidr-ts's 13-question scenario lands around 350ms against
the same Cerebras model, roughly a third of this port's 1050ms. None of
this changes correctness — `Client.truth()` still returns the same
calibrated, `logprobs`-derived probability either way; it's purely a
latency characteristic of this port's concurrency model under high
fan-out.

## Reading a `Decision`

| Field | |
|---|---|
| `choice` | option id with the highest probability |
| `probabilities` | option id → probability for every option that was actually compared |
| `unscored` | options that genuinely could not be measured — a real gap, be cautious about trusting the result if this is non-empty |
| `eliminated` | options that lost a real comparison but weren't explored further (only with `exhaustive=False`) — not a gap, just not a full distribution |
| `stopped_early` | options scored on a partial logprob because no competition remained for their prefix — a genuine but partial measurement |
| `logprobs` | raw log probabilities behind `probabilities`, before normalizing |
| `raw_answer` | the model's own first reply, for debugging |

```python
from decidr import confidence, is_reliable

if not is_reliable(decision):
    # some option's real answer never showed up in the model's response at
    # all -- decide what your app should do here (retry, fall back, flag it)
    ...

route_to(decision.choice)
```

## Calibration

`decision.probabilities` are real softmax'd logprobs, not hand-waved confidence scores — but "real" doesn't automatically mean "calibrated." `fit_temperature` finds one scalar temperature that rescales them to better match actual outcomes, and reports Expected Calibration Error (ECE) before/after so you can see the improvement rather than assume it:

```python
from decidr.calibrate import fit_temperature, evaluate_out_of_fold

labeled = [(client.decide(row), row["correct_id"]) for row in your_labeled_rows]

evaluate_out_of_fold(labeled, folds=5)
# CalibrationResult(temperature=1.8, n=200, ece_before=0.15, ece_after=0.06, accuracy=0.83)
```

`temperature` is a scalar you can then pass to `Client(..., temperature=1.8)` to make future probabilities track real accuracy more closely. It only rescales confidence, via a monotone transform — `choice` never changes; `fit_temperature` asserts this rather than trusting it.

## API

**`Client(model, host=DEFAULT_HOST, timeout=120.0, temperature=1.0, backend=None, exhaustive=True, cache=True, max_workers=16)`**

- `model` — the model name your backend expects.
- `host` — only used when no `backend` is given; builds the default `OpenAIBackend(base_url=host)`.
- `backend` — `OpenAIBackend(...)` for any real provider. See [docs/PROVIDERS.md](docs/PROVIDERS.md).
- `exhaustive` — `True` (default) compares every option against every sibling for a full `probabilities` distribution, more requests. `False` follows only the winning branch at each level, fewer requests, incomplete branches go to `eliminated`.
- `cache` — `True` (default) keeps a speculative token-boundary cache at `~/.decidr/token-cache.json`; pass `False` to disable, or a `TokenCache` instance to share one across clients.
- `max_workers` — size of the shared thread pool used to fire independent requests concurrently.

**`client.decide(row) -> Decision`**, **`client.score(row) -> dict`**, **`client.truth(row) -> dict`** — one decision. **`client.decide_all(rows) -> list[Decision]`** — a list, sequentially. **`client.warmup(row) -> Decision`** — pre-warm, then decide.

`Client` is a context manager (`with Client(...) as client:`) that shuts down its thread pool on exit; call `client.close()` directly if you're not using it as one.

## Why this works without the model generating anything

A model computes a probability over its whole vocabulary before it samples a word. `decidr` reads that distribution directly — one forward pass, no decoding loop, no text to parse back into a decision.

Reading it out reliably is the actual engineering problem, and it's covered in linked docs rather than here, since none of it is required to just use the library:

- [docs/PREFIX_MATCHING.md](docs/PREFIX_MATCHING.md) — how a multi-token option id gets scored at all, since there's no way to ask a model how a string tokenizes in advance
- [docs/HIERARCHY.md](docs/HIERARCHY.md) — why large option sets need more than one comparison, measured, and what `eliminated` actually means
- [docs/NAMING_IDS.md](docs/NAMING_IDS.md) — the id format rules and why each one exists
- [docs/PROVIDERS.md](docs/PROVIDERS.md) — the `Backend` interface, the full checked provider matrix, writing your own

## Limitations

- An option can land in `unscored` if its real next token misses the model's top-20 logprob window within one comparison — rare with a reasonable hierarchy, not impossible.
- Verified live against Cerebras (`qwen-3.8-27b`) and unit-tested against a stubbed OpenAI transport; not every provider in [docs/PROVIDERS.md](docs/PROVIDERS.md)'s table has been individually live-verified by this project.
- Small models (under 1B parameters) often won't treat a bare option id as a plausible next answer at all.

## License

MIT
