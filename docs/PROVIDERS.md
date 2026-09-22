# Providers

How to point `Client` at a specific model provider, with a full working example for each. For why the interface is shaped this way, see [Writing your own backend](#writing-your-own-backend) at the bottom.

**`decidr` needs a model that returns `logprobs`, and not every model does.** This isn't specific to one provider — it's most consistently true of reasoning-focused models (OpenAI's o-series and similar reasoning models on other providers), which commonly reject or ignore `logprobs` entirely because their API contract is built around a hidden reasoning step rather than a plain next-token distribution. Standard chat models (GPT-4o-family and most open-weight chat models) support it — Claude models currently do not, on any route (see [Anthropic (Claude) is not currently reachable](#anthropic-claude-is-not-currently-reachable) below). If `decide()` fails with something like "server returned no logprobs," a reasoning model is the first thing to check.

## Ollama (default, local)

```python
from decidr import Client

client = Client(model="qwen3.5:4b")  # talks to http://127.0.0.1:11434
```

```python
client = Client(model="qwen3.5:4b", host="http://192.168.1.50:11434")  # a remote Ollama
```

Nothing to install beyond `decidr` itself.

## OpenAI

```bash
pip install decidr[litellm]
```

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

**Only OpenAI's standard chat models support `logprobs`** — `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, and that family. OpenAI's reasoning models (the o-series and other reasoning-focused models) reject `logprobs` outright, so `decidr` can't work with them; use a standard model, not a reasoning one. `api_key` can also come from the `OPENAI_API_KEY` environment variable, in which case `LiteLLMBackend()` needs no arguments.

## Anthropic (Claude) is not currently reachable

Checked directly, not assumed (this section previously suggested `LiteLLMBackend` could reach Claude — it can't, and that example was never actually run live; see the "What's verified" section below): Claude's native Messages API (`/v1/messages`) has no `logprobs` field at all, and Anthropic's own OpenAI-compatible endpoint explicitly documents `logprobs` as an unsupported parameter that gets silently ignored rather than an error. `LiteLLMBackend` can't get logprobs out of either route, because the provider itself never sends them — no client library, including LiteLLM, can manufacture a field the API doesn't return.

Pointing `LiteLLMBackend` at a Claude model will run and then fail with something like "server returned no logprobs for this step," the same failure mode as any other provider that doesn't support logprobs (see [PREFIX_MATCHING.md](PREFIX_MATCHING.md)).

If Anthropic adds logprobs support to the Messages API in the future, this section will be updated; until it does, Claude models aren't a fit for this mechanism regardless of client library.

## Any other LiteLLM provider

```python
from decidr import Client, LiteLLMBackend

client = Client(
    model="bedrock/meta.llama3-1-8b-instruct-v1:0",
    backend=LiteLLMBackend(),  # picks up AWS credentials from the environment
)
```

`LiteLLMBackend(**kwargs)` forwards every keyword straight to LiteLLM's own `completion()` — `api_key`, `api_base`, `aws_region_name`, whatever that provider needs. The `model` string follows [LiteLLM's own provider naming](https://docs.litellm.ai/docs/providers). This covers many providers this way: OpenAI, Bedrock (non-Claude models), Vertex AI, a self-hosted vLLM endpoint, and more — as long as the underlying model actually returns `logprobs`, which is a property of the model/provider, not of LiteLLM. A Claude model routed through Bedrock or Vertex still has no `logprobs` to return, the same as calling Anthropic directly.

## How to tell if a provider will work, before wiring it up

There's no universal registry of who supports `logprobs` — it changes as providers ship features, so treat the table below as a starting point to verify, not a guarantee. Two ways to check a specific provider without writing any `decidr` code first:

1. **Read that provider's own API reference** for its chat completions endpoint and search it for `logprobs` / `top_logprobs`. If the parameter isn't documented at all, or is documented as accepted-but-ignored (as Anthropic's OpenAI-compat layer does), the provider won't work here regardless of what `decidr` or LiteLLM does.
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

   If `choices[0].logprobs.content` is present and non-empty, the provider supports it. If it's missing, `null`, or the request errors on the `logprobs`/`top_logprobs` fields, it doesn't — `LiteLLMBackend` will fail the same way `decide()` would, just without spending a hierarchy's worth of requests finding out.

### Providers known to support `logprobs`

Checked against each provider's own documentation, not assumed — but providers change this without notice, so verify with the request above before depending on it in production:

| Provider | Notes |
|---|---|
| OpenAI | Standard chat models only (`gpt-4o` family, `gpt-4.1`); reasoning models (o-series) reject `logprobs`. |
| Together AI | Documented support for `logprobs`/`top_logprobs`; note `logprobs` and streaming (`stream=True`) are mutually exclusive on their API — `decidr` doesn't stream, so this doesn't affect it. |
| Groq | OpenAI-compatible endpoint; verify current `top_logprobs` cap before relying on a specific value. |
| Fireworks AI | OpenAI-compatible endpoint; reported `top_logprobs` cap is lower than OpenAI's (around 5) — `OllamaBackend`/`LiteLLMBackend` always request the max, so expect the response to come back capped rather than erroring, which is fine for `decidr`'s mechanism (it just means fewer alternatives to match against per step). |
| Self-hosted vLLM | Implements the OpenAI-compatible server spec including `logprobs`/`top_logprobs` directly; this is the same shape Together, Fireworks, and several other hosted providers build on. |

**Not currently usable**, regardless of routing: Anthropic/Claude (see [above](#anthropic-claude-is-not-currently-reachable)) and any reasoning-focused model on any provider (the API contract is built around a hidden reasoning step instead of a plain next-token distribution, and `logprobs` is typically rejected or ignored as a result).

## `LiteLLMBackend` is not a way to reach a local Ollama

Checked directly, not assumed: LiteLLM's own Ollama integration (`ollama/` and `ollama_chat/` model prefixes) does not forward `logprobs`/`top_logprobs` at all —

```
litellm.exceptions.UnsupportedParamsError: ollama_chat does not support
parameters: ['logprobs', 'top_logprobs'], for model=qwen3.5:4b.
```

— even though Ollama's own API supports them natively, which is the entire premise `OllamaBackend` runs on. If you're running against a local Ollama, use `OllamaBackend` (the default) — it's simpler, needs nothing extra installed, and is the only path that actually returns logprobs from Ollama today.

## What's verified, and what isn't

`OllamaBackend`'s request construction and response handling are covered by live tests against a real running model (throughout this project's test suite) and by unit tests against a mocked transport (`tests/test_backend.py`).

`LiteLLMBackend`'s response normalization is unit-tested against objects matching LiteLLM's documented `ChatCompletion` shape, and its request construction is verified the same way. What is **not** verified in this project: an actual live call to OpenAI or any other hosted provider through it — that needs an API key this project doesn't have. The OpenAI example above is correct usage, not confirmed wire behavior. If a specific provider does something LiteLLM's documented shape doesn't predict, that's the likely place to look first.

Anthropic is the one case that's more than just unverified: it's confirmed, via Anthropic's own API docs, to have no `logprobs` field at all — see [Anthropic (Claude) is not currently reachable](#anthropic-claude-is-not-currently-reachable) above. That's not a gap in this project's test coverage; it's a gap in what the provider sends.

## Writing your own backend

For a provider LiteLLM doesn't cover, or to avoid the LiteLLM dependency for a provider whose API you'd rather call directly: subclass `Backend` and implement one method.

```python
from decidr import Backend

class MyBackend(Backend):
    def chat(self, model: str, messages: list[dict]) -> dict:
        # Send `messages` to the model, asking it to predict exactly one
        # token at temperature 0. Return:
        return {
            "content": "...",           # the model's own reply, or None
            "logprobs": [{              # zero entries, or exactly one
                "token": "...",
                "logprob": -0.1,
                "top_logprobs": [{"token": "...", "logprob": -0.1}, ...],
            }],
        }
```

An empty `logprobs` list means "this call returned no logprob information at all" — `decidr` treats that as a hard error rather than guessing at a decision with no numbers behind it. `messages` is the same OpenAI-style `[{"role": ..., "content": ...}]` list every built-in backend receives; how you turn that into a request for your provider is up to you.
