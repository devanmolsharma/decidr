# Providers

How to point `Client` at a specific model provider, with a full working example for each. For why the interface is shaped this way, see [Writing your own backend](#writing-your-own-backend) at the bottom.

**`decidr` needs a model that returns `logprobs`, and not every model does.** This isn't specific to one provider — it's most consistently true of reasoning-focused models (OpenAI's o-series and similar reasoning models on other providers), which commonly reject or ignore `logprobs` entirely because their API contract is built around a hidden reasoning step rather than a plain next-token distribution. Standard chat models (the GPT-4o family, and most open-weight chat models) support it. If `decide()` fails with something like "could not score any option," a reasoning model is the first thing to check.

This library ships exactly one backend, built on the official `openai` PyPI package: `OpenAIBackend`. It isn't OpenAI-only — it speaks the OpenAI-compatible `/v1/chat/completions` wire format, which covers OpenAI itself, Ollama's own OpenAI-compatible endpoint (verified live to return real, correct `logprobs`), and most self-hosted or hosted-elsewhere servers that speak the same shape (see [Any other OpenAI-compatible server](#any-other-openai-compatible-server) below). There is deliberately no separate Ollama-specific backend and no dependency on a general-purpose multi-provider gateway (LiteLLM, OpenRouter, and similar) — see [Gateways and unified multi-provider clients: none solve this reliably](#gateways-and-unified-multi-provider-clients-none-solve-this-reliably) below for the researched reasoning.

## Ollama (default, local)

```python
from decidr import Client

client = Client("qwen3.5:4b")  # talks to Ollama's OpenAI-compatible endpoint at http://127.0.0.1:11434/v1
```

```python
client = Client("qwen3.5:4b", host="http://192.168.1.50:11434/v1")  # a remote Ollama
```

Models with a reasoning/thinking mode enabled by default (common on newer Ollama models) are handled automatically: `OpenAIBackend` tries `reasoning_effort="none"` on its first request and falls back transparently if the provider doesn't recognize that field.

## OpenAI

```python
from decidr import Client, OpenAIBackend

client = Client("gpt-4o-mini", backend=OpenAIBackend(api_key="sk-..."))

decision = client.decide({
    "id": "ticket-1",
    "state": "Customer cannot access their account after a password reset. The reset email never arrived.",
    "question": "Which team should handle this?",
    "options": [
        {"id": "access", "description": "Account access and authentication issues."},
        {"id": "billing", "description": "Billing and payment issues."},
        {"id": "sales", "description": "Sales and product questions."},
    ],
})
```

**Only OpenAI's standard chat models support `logprobs`** — `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, and that family. OpenAI's reasoning models (the o-series and other reasoning-focused models) reject `logprobs` outright, so `decidr` can't work with them; use a standard model, not a reasoning one. `api_key` can also come from the `OPENAI_API_KEY` environment variable, in which case `OpenAIBackend()` needs no arguments.

## Any other OpenAI-compatible server

```python
from decidr import Client, OpenAIBackend
import os

client = Client(
    "some-model",
    backend=OpenAIBackend(
        base_url="https://my-inference-host.example.com/v1",
        api_key=os.environ["MY_PROVIDER_API_KEY"],
    ),
)
```

`OpenAIBackend(base_url=..., api_key=...)` works against anything that implements `POST {base_url}/chat/completions` in the OpenAI shape and supports `logprobs`/`top_logprobs` — many self-hosted inference servers (vLLM, and others with an OpenAI-compatible front end) and some hosted third-party APIs qualify. Whether logprobs specifically are supported and forwarded correctly is up to that server; if `decide()` fails with a "could not score any option" error against a server you expected to support it, check that server's own OpenAI-compatibility docs for `logprobs` first.

### How to tell if a provider will work, before wiring it up

There's no universal registry of who supports `logprobs` — it changes as providers ship features, so treat the table below as a starting point to verify, not a guarantee. Two ways to check a specific provider without writing any `decidr` code first:

1. **Read that provider's own API reference** for its chat completions endpoint and search it for `logprobs` / `top_logprobs`. If the parameter isn't documented at all, or is documented as accepted-but-ignored (as Anthropic's OpenAI-compat layer does), the provider won't work here regardless of what `decidr` does.
2. **Send one raw request yourself** before involving `decidr`, and check whether a `logprobs` object actually comes back on the choice:

   ```bash
   curl https://your-provider.example.com/v1/chat/completions \
     -H "Authorization: Bearer $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "your-model",
       "messages": [{"role": "user", "content": "Say hi."}],
       "max_tokens": 1,
       "logprobs": true,
       "top_logprobs": 5
     }'
   ```

   If `choices[0].logprobs.content` is present and non-empty, the provider supports it. If it's missing, `null`, or the request errors on the `logprobs`/`top_logprobs` fields, it doesn't — `OpenAIBackend` will fail the same way `decide()` would, just without spending a hierarchy's worth of requests finding out.

### Providers known to support `logprobs` via an OpenAI-compatible endpoint

Checked against each provider's own documentation and, where noted, live measurement — not assumed. Providers change this without notice, so verify with the request above before depending on it in production. An independent research pass across the full landscape (2026) found that of 813 real OpenRouter endpoints tested empirically, only 23% actually returned logprobs when requested — so "OpenAI-compatible" is not a reliable predictor of logprobs support on its own, and every provider below was checked individually:

| Provider | Notes |
|---|---|
| OpenAI (direct or Azure OpenAI) | Standard chat models only (`gpt-4o` family, `gpt-4.1`, and similar); reasoning models (o-series, GPT-5.x/6 reasoning variants) reject `logprobs`. Azure has full parity with direct OpenAI here. |
| Together AI | Documented support for `logprobs`/`top_logprobs`; note `logprobs` and streaming (`stream=True`) are mutually exclusive on their API — `decidr` doesn't stream, so this doesn't affect it. |
| Fireworks AI | OpenAI-compatible endpoint; `top_logprobs` cap is lower than OpenAI's (around 5, same as Azure) — `OpenAIBackend` always requests 20, so expect the response to come back capped rather than erroring, which is fine for `decidr`'s mechanism (it just means fewer alternatives to match against per step). |
| Cerebras (standard models only), NVIDIA NIM | OpenAI-compatible endpoints, documented `logprobs` support. On Cerebras specifically: verified live against their two public models — `qwen-3.8-27b` (standard chat model) returns real, correct `logprobs`; `gpt-oss-120b` (a reasoning model) does not — confirmed live with `logprobs=True, top_logprobs=5` returning a response with no `logprobs` field at all (`finish_reason: "length"` after the single requested token, with empty `content`), and separately rejects the `reasoning_effort="none"` this backend sends by default with an explicit 400 ("Unsupported reasoning effort: none. Supported values are 'low', 'medium', and 'high'") rather than silently ignoring it. This is the same reasoning-model exclusion as OpenAI's o-series above, not a Cerebras-specific gap. |
| Self-hosted vLLM | Implements the OpenAI-compatible server spec including `logprobs`/`top_logprobs` directly, plus `prompt_logprobs` (input-token logprobs, not exposed by any hosted provider here) — the most complete logprobs surface of any option in this table. This is the same request shape Together, Fireworks, and several other hosted providers build on. |
| Self-hosted llama.cpp server | `n_probs` parameter, OpenAI-*similar* (not identical) shape. One real caveat: these are **post-sampling** logprobs, reflecting whatever temperature/top-k/top-p was applied, not the raw model distribution — decidr always requests temperature 0 specifically to make this moot, but it's worth knowing this endpoint's numbers can differ from a raw-logit read in general. |

**Confirmed NOT usable, corrected from an earlier version of this doc that assumed otherwise:**

| Provider | Verdict |
|---|---|
| **Groq** | **Does not support `logprobs` at all.** Groq's own docs state plainly that `logprobs`, `logit_bias`, and `top_logprobs` "are currently not supported and will result in a 400 error if they are supplied." An earlier version of this table listed Groq as supported — that was wrong; do not route decidr through Groq. |
| xAI (Grok) | Partial and unreliable: documented support up to a small window (0-8), but confirmed silently ignored (no error, no logprobs) on newer models (grok-4.20 and later). Treat as unusable without per-model live verification. |
| DeepSeek | Not available in "thinking" mode; even in standard mode, V3.2 has been observed returning meaningless placeholder values (`0` or `-9999`) rather than real logprobs or a clear error. Do not trust this provider's logprobs without independently sanity-checking the actual values returned, not just their presence. |
| Cohere (Chat v2) | Returns a bare `logprobs: true/false`, no `top_logprobs`/rank window at all, and the tokens come back as opaque `token_ids` requiring Cohere's own tokenizer to map back to text — not usable by `decidr`'s mechanism, which needs real token strings to match against option ids. |
| Google Gemini / Vertex AI | `responseLogprobs`/`logprobs` exist but only on Vertex AI (not the consumer Gemini API), only for non-streaming calls, and disabled entirely on several newer model versions — verify per exact model id before use. |
| Mistral (La Plateforme) | Not in Mistral's own documented API parameters. (Self-hosted Mistral models served through vLLM do get real logprobs — that's vLLM's support, not Mistral's own API.) |
| AWS Bedrock (Converse API) | Not in Bedrock's unified Converse schema. `additionalModelRequestFields` theoretically allows passing arbitrary per-model parameters through, but this is undocumented and unverified per model — do not assume it works without testing the specific model. |
| Anthropic (Claude) | No `logprobs` field anywhere, on any route — see [below](#anthropic-claude-is-not-currently-reachable). |

**Reasoning models on any provider** are a separate, orthogonal exclusion: their API contract is built around a hidden reasoning step instead of a plain next-token distribution, and `logprobs` is typically rejected or ignored as a result, regardless of whether that provider supports logprobs for its standard chat models.

### Gateways and unified multi-provider clients: none solve this reliably

If you're tempted to reach for one client library or gateway to cover many providers at once (LiteLLM, OpenRouter, Vercel AI SDK, LangChain, or similar) instead of picking a backend per provider: don't, for `decidr` specifically. This was checked directly across the landscape, not assumed:

- **LiteLLM** silently drops `logprobs`/`top_logprobs` when routing Ollama through its OpenAI-compatible code path (the root cause is upstream in Ollama's own compatibility layer, not LiteLLM itself). Its `drop_params` option defaults to `False` (it raises rather than silently dropping) but most real deployments enable it, reintroducing silent drops.
- **OpenRouter** documents `logprobs` as a normalized parameter, but empirically only ~23% of its endpoints actually honor it, and — confirmed directly — the *same model slug* can route to a logprobs-supporting upstream or a non-supporting one depending purely on which provider OpenRouter's router happens to pick that request, unless you explicitly pin the upstream provider and set `require_parameters: true`. The default behavior on an unsupported route is a normal `200 OK` with `logprobs: null` — no error, no signal anything went wrong.
- **Vercel AI SDK** removed its normalized cross-provider logprobs support entirely in v5.
- **LangChain**'s Ollama integration (`ChatOllama`) has a long-standing open, unresolved bug around logprobs.
- Every general-purpose "unified LLM SDK" checked (PydanticAI, any-llm, Token.js, aisuite) either doesn't normalize logprobs at all, or silently returns `None`/`null` for providers it can't support — the same failure mode as LiteLLM, just less documented.

The common thread: logprobs is a low-traffic feature in every general-purpose abstraction, and it's consistently the first thing dropped or left unverified. An abstraction that silently returns nothing is strictly worse for `decidr` than a direct, provider-specific call that errors loudly — silence produces a confusing failure deep inside a real `decide()` call instead of an immediate, clear one. Build backends against each provider's own official SDK or direct API instead.

## Anthropic (Claude) is not currently reachable

Checked directly, not assumed: Claude's native Messages API (`/v1/messages`) has no `logprobs` field at all, and Anthropic's own OpenAI-compatible endpoint explicitly documents `logprobs` as an unsupported parameter that gets silently ignored rather than an error. Neither route gives `decidr` anything to score a decision from, so there is no working `AnthropicBackend` to add here — one would run and then always fail with "could not score any option," which is worse than not having it at all, since it would invite spending an API call on something that can never produce a `Decision`.

If Anthropic adds logprobs support to the Messages API in the future, `OpenAIBackend` (pointed at a compatible endpoint) or a small dedicated backend would become viable then; until it does, Claude models aren't a fit for this mechanism regardless of client library.

## What's verified, and what isn't

`OpenAIBackend`'s request construction and response handling are covered by unit tests against a stubbed transport, and have been verified live against both a real Cerebras endpoint and the multimodal (image) path. Third-party OpenAI-compatible hosts (Together AI, Fireworks, and others listed above) are covered by their own documented `logprobs` support, checked directly against their docs, but have not all been individually verified live by this project against a real account on every one of them.

## Writing your own backend

For a provider whose API you'd rather call directly, or that doesn't speak the OpenAI-compatible shape: subclass `Backend` and implement `chat`.

```python
from decidr import Backend

class MyBackend(Backend):
    def chat(self, model: str, messages: list[dict], max_tokens: int = 1) -> dict:
        # Send `messages` to the model, asking it to predict `max_tokens`
        # tokens (usually just 1) at temperature 0. Return:
        return {
            "content": "...",  # the model's own reply, or None
            "logprobs": [{      # zero entries, or one per generated position
                "token": "...",
                "logprob": -0.1,
                "top_logprobs": [{"token": "...", "logprob": -0.1}],
            }],
        }

    # warmup() and discover_tokens_batch() have default implementations
    # built purely on chat() -- override them only if your provider has
    # something cheaper.
```

An empty `logprobs` list means "this call returned no logprob information at all" — `decidr` treats that as a hard error rather than guessing at a decision with no numbers behind it. `messages` is the same OpenAI-style `[{"role": ..., "content": ...}]` list every built-in backend receives; how you turn that into a request for your provider is up to you.
