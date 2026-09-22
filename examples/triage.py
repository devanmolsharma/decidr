"""Route support tickets, score evidence, and read tone -- three decisions,
one forward pass each, against whatever model you already have in Ollama.

    python examples/triage.py [model] [host]
"""

import sys
import time

from decidr import Client, DecisionError

ROWS = [
    {
        "id": "route",
        "state": "Customer cannot access an account after a password reset. The reset email never arrived.",
        "question": "Which queue should handle this request?",
        "options": [
            {"id": "access", "description": "Account access and authentication support."},
            {"id": "billing", "description": "Billing and payment support."},
            {"id": "sales", "description": "Sales and product evaluation."},
        ],
    },
    {
        "id": "evidence",
        "state": "The deployment completed at 14:02 UTC. Health checks passed in all three zones. No rollback was initiated.",
        "question": "Is there evidence that the deployment succeeded?",
        "options": [
            {"id": "yes", "description": "The deployment succeeded."},
            {"id": "no", "description": "The deployment did not succeed."},
            {"id": "insufficient", "description": "The evidence is insufficient to decide."},
        ],
    },
    {
        "id": "tone",
        "state": "This is the THIRD time you've charged me twice. I want a refund NOW or I'm calling my bank.",
        "question": "How frustrated is the customer?",
        "options": [
            {"id": "calm", "description": "Calm and neutral."},
            {"id": "annoyed", "description": "Mildly annoyed."},
            {"id": "angry", "description": "Very angry."},
        ],
    },
]


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5:4b"
    host = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:11434"

    client = Client(model=model, host=host)
    try:
        mode = "exact" if client.supports_exact() else "ranked"
    except DecisionError as e:
        print(f"could not reach Ollama: {e}")
        return 1

    print(f"model={model}  mode={mode}\n")
    for row in ROWS:
        started = time.perf_counter()
        try:
            d = client.decide(row)
        except DecisionError as e:
            print(f"{row['id']:10s} error: {e}")
            continue
        ms = round((time.perf_counter() - started) * 1000)

        ranked = sorted(d.probabilities.items(), key=lambda kv: -kv[1])
        spread = "  ".join(f"{k}={v:.3f}" for k, v in ranked)
        print(f"{row['id']:10s} {d.choice:13s} {d.confidence:6.1%}  {ms:5d}ms   {spread}")
        if not d.is_reliable():
            print(f"{'':10s} unscored (outside the top-20 window): {d.unscored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
