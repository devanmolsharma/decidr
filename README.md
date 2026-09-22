# decidr

Typed decisions from an LLM, in one forward pass. Give it a question and a set of named options; get back which one the model picked and a real probability for each, without the model generating a single word.

```bash
pip install decidr
```

## Quickstart: local Ollama

Needs [Ollama](https://ollama.com) running with a model already pulled. No fine-tuning, no other setup.

```python
from decidr import Client

client = Client(model="qwen3.5:4b")

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
decision.confidence      # 0.9965
decision.probabilities   # {'access': 0.9965, 'billing': 0.0017, 'sales': 0.0018}
```

That's the whole thing: `state` is what the model reads, `question` is what it's deciding, `options` is the closed set of possible answers, each with an `id` you'll match on in your own code and a `description` for the model to read.

## Quickstart: a hosted model (OpenAI, Anthropic, etc.)

```bash
pip install decidr[litellm]
```

Same `decide()` call, same `row` shape — only the `Client` construction changes:

```python
from decidr import Client, LiteLLMBackend

client = Client(
    model="gpt-4o-mini",
    backend=LiteLLMBackend(api_key="sk-..."),
)

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
```

`LiteLLMBackend` routes through [LiteLLM](https://github.com/BerriAI/litellm), so anything LiteLLM can reach, `decidr` can: `model="claude-3-5-haiku-20241022"` with `LiteLLMBackend(api_key="...")` for Anthropic, `model="bedrock/anthropic.claude-3-haiku..."` for Bedrock, and so on — the model string is whatever LiteLLM expects for that provider. Details and a real caveat worth knowing first: [docs/PROVIDERS.md](https://github.com/devanmolsharma/decidr/blob/main/docs/PROVIDERS.md).

## Writing a `row`

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Your own identifier for this decision, echoed back on the result. Not sent to the model. |
| `state` | string, dict, or list | The evidence the model reads to make the call. |
| `question` | string | What it's deciding. |
| `options` | list of `{id, description}` | The closed set of answers. `id` is what comes back as `decision.choice`; `description` is what the model reads. |

Option ids: lowercase letters/digits in underscore-separated segments (`billing`, `access_denied`), under 40 characters, and no id may be a prefix of another (`billing` and `billing_refund` can't coexist — see [docs/NAMING_IDS.md](https://github.com/devanmolsharma/decidr/blob/main/docs/NAMING_IDS.md)). This is checked before any request is sent, so a bad id fails immediately with a message naming the problem, not partway through.

There's no cap on how many options a decision can have. Above a handful, `decidr` uses the ids' own underscore segments as a hierarchy to keep every individual comparison small and reliable instead of racing everything at once — see [docs/HIERARCHY.md](https://github.com/devanmolsharma/decidr/blob/main/docs/HIERARCHY.md) for what that means for your ids and for request cost.

## Reading a `Decision`

| Field | |
|---|---|
| `choice` | option id with the highest probability |
| `confidence` | probability of `choice` |
| `probabilities` | option id → probability for every option that was actually compared |
| `unscored` | options that genuinely could not be measured — a real gap, be cautious about trusting the result if this is non-empty |
| `eliminated` | options that lost a real comparison but weren't explored further (only with `exhaustive=False`, see below) — not a gap, just not a full distribution |
| `is_reliable()` | `False` when `unscored` is non-empty; unaffected by `eliminated` |
| `logprobs` | raw log probabilities behind `probabilities`, before normalizing |
| `raw_answer` | the model's own first reply, for debugging |

```python
if not decision.is_reliable():
    # some option's real answer never showed up in the model's response at
    # all -- decide what your app should do here (retry, fall back, flag it)
    ...

route_to(decision.choice)
```

## Calibration

`confidence` out of the box is directional, not a calibrated probability — a raw model's logits usually aren't well-calibrated. If you have labeled examples (rows where you already know the right `id`), fit a correction:

```python
from decidr.calibrate import fit_temperature, evaluate_out_of_fold

labeled = [(client.decide(row), row["correct_id"]) for row in your_labeled_rows]

evaluate_out_of_fold(labeled, folds=5)
# CalibrationResult(temperature=1.8, n=200, ece_before=0.15, ece_after=0.06, accuracy=0.83)
```

`temperature` is a scalar you can then pass to `Client(..., temperature=1.8)` to make future `confidence` values track real accuracy more closely. It only rescales confidence — `choice` never changes.

## API

**`Client(model, host="http://127.0.0.1:11434", temperature=1.0, backend=None, exhaustive=True)`**

- `model` — the model name, meaning depends on the backend (an Ollama tag by default, a LiteLLM-style model string with `LiteLLMBackend`).
- `host` — only used when no `backend` is given; builds the default `OllamaBackend(host=host)`.
- `backend` — `OllamaBackend()` (default) or `LiteLLMBackend(...)` for a hosted provider. See [docs/PROVIDERS.md](https://github.com/devanmolsharma/decidr/blob/main/docs/PROVIDERS.md).
- `exhaustive` — `True` (default) compares every option against every sibling for a full `probabilities` distribution, more requests. `False` follows only the winning branch at each level, fewer requests, incomplete branches go to `eliminated`.

**`client.decide(row) -> Decision`** — one decision. **`client.decide_all(rows) -> list[Decision]`** — a list, sequentially.

## Why this works without the model generating anything

A model computes a probability over its whole vocabulary before it samples a word. `decidr` reads that distribution directly — one forward pass, no decoding loop, no text to parse back into a decision. It's not novel as a technique (TypeSafe's Jev and [SemIf](https://github.com/TheoLeeCJ/SemIf) both do versions of it); this is a small, dependency-light implementation of it.

Reading it out reliably is the actual engineering problem, and it's covered in three linked docs rather than here, since none of it is required to just use the library:

- [docs/PREFIX_MATCHING.md](https://github.com/devanmolsharma/decidr/blob/main/docs/PREFIX_MATCHING.md) — how a multi-token option id gets scored at all, since there's no way to ask a model how a string tokenizes in advance
- [docs/HIERARCHY.md](https://github.com/devanmolsharma/decidr/blob/main/docs/HIERARCHY.md) — why large option sets need more than one comparison, measured, and what `eliminated` actually means
- [docs/PROVIDERS.md](https://github.com/devanmolsharma/decidr/blob/main/docs/PROVIDERS.md) — the `Backend` interface, LiteLLM specifics, writing your own

## Limitations

- Sequential, no batching.
- An option can land in `unscored` if its real next token misses the model's top-20 logprob window within one comparison — rare with a reasonable hierarchy, not impossible.
- `LiteLLMBackend` is unit-tested against LiteLLM's documented response shape; a live call to a real hosted provider hasn't been run in this project (no API key available here).
- Verified against `qwen3.5:4b`. Small models (under 1B parameters) often won't treat a bare option id as a plausible next answer at all.

## License

MIT
