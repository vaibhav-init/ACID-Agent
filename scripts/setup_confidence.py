"""Download the local confidence model (Qwen3-0.6B) and run a self-test.

Usage:  python scripts/setup_confidence.py
Requires: pip install -e ".[confidence]"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acid_agent.confidence import (  # noqa: E402
    code_span_divergence,
    decision_divergence,
    exploration_redundancy,
    get_scorer,
)


def main():
    print("Loading local confidence model (first run downloads ~1.2 GB)...")
    scorer = get_scorer()
    c1 = scorer.score("The capital of France is Paris.", "Answer the question.")
    c2 = scorer.score("The capital of France is Paris.", "Q: What is the capital of France? A:")
    print(f"sanity confidences: {c1:.4f} vs {c2:.4f}")

    dd = decision_divergence(
        ["group by month after normalizing date separators"],
        "data contains two date formats: '-' and '/'",
        "average monthly revenue",
    )
    cd = code_span_divergence(
        "df['date'] = df['date'].str.replace('/', '-')",
        "data contains two date formats: '-' and '/'",
        "average monthly revenue",
    )
    red = exploration_redundancy(
        "the file has 100 rows with columns a,b", "the file has 100 rows with columns a,b"
    )
    print(f"decision_divergence={dd:.4f}  code_span_divergence={cd:.4f}  redundancy={red:.4f}")
    print("OK: confidence engine is working.")


if __name__ == "__main__":
    main()