"""Download the local confidence model (Qwen3-0.6B-Base) and run a self-test.

Usage:  python scripts/setup_confidence.py
Requires: pip install -e ".[confidence]"

The self-test is directional, not just numeric: it feeds one code span that
FOLLOWS the evidence and one that CONTRADICTS it, and checks that the
contradicting span scores higher surprise. A gate whose sign is flipped passes
every numeric check and still rejects the wrong attempts, so the sign is the
thing worth asserting here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acid_agent.config import get_settings  # noqa: E402
from acid_agent.confidence import (  # noqa: E402
    code_span_surprise,
    decision_surprise,
    exploration_redundancy,
    get_scorer,
    probability_contrast,
)

TASK = "average monthly revenue"
EVIDENCE = "the date column mixes two formats: '2024-01-05' and '05/01/2024'"


def main():
    cfg = get_settings()
    print(f"Loading {cfg.confidence_model} (first run downloads ~1.2 GB)...")
    scorer = get_scorer()
    lp1 = scorer.score_logp("The capital of France is Paris.", "Answer the question.")
    lp2 = scorer.score_logp(
        "The capital of France is Paris.", "Q: What is the capital of France? A:"
    )
    print(f"sanity mean logprob (nats): {lp1:.4f} vs {lp2:.4f}  (higher = more expected)")
    if lp2 <= lp1:
        print("  WARN: the question context did not raise the answer's likelihood; "
              "check the model is the -Base checkpoint.")

    prelude = "df = pd.read_csv('sales.csv')" + chr(10)
    aligned = code_span_surprise(
        prelude + "df['date'] = pd.to_datetime(df['date'], format='mixed')", EVIDENCE, TASK
    )
    contradicting = code_span_surprise(
        prelude + "df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')", EVIDENCE, TASK
    )
    if not (aligned.get("available") and contradicting.get("available")):
        # A probe with no spans is a gate component that never fires, so say so
        # rather than printing two zeros that look like a verdict.
        print(f"span surprise  NO SPANS EXTRACTED -> {contradicting.get('reason')}")
    else:
        print(f"span surprise  aligned={aligned['max_surprise']:.4f}  "
              f"contradicting={contradicting['max_surprise']:.4f}  "
              f"(retry above {cfg.span_surprise_retry}, {contradicting['n_scored']} spans scored)")
        if contradicting["max_surprise"] <= aligned["max_surprise"]:
            print("  WARN: the contradicting span did not score higher surprise. The signal "
                  "is weak on this example; do not trust the gate until this separates.")

    contrast = probability_contrast(
        [{
            "anchor_id": "mixed_date_formats",
            "decision_type": "date_parsing",
            "current_policy": "parse dates with a single fixed format",
            "expected_policy": "parse dates with a mixed-format parser",
        }],
        TASK,
        EVIDENCE,
    )
    if contrast.get("available"):
        p = contrast["probes"][0]
        print(f"contrast ratio P(current)/P(alternative)={p['ratio']:.4f} -> {p['status']} "
              f"(retry below {cfg.contrast_retry_ratio})")
    else:
        print(f"contrast: {contrast.get('reason')}")

    dec = decision_surprise(
        ["group by month after normalizing date separators"], EVIDENCE, TASK
    )
    print(f"decision surprise={dec['max_surprise']:.4f} (DIAGNOSTIC, never gates)")

    red_same = exploration_redundancy(
        "the file has 100 rows with columns a,b", "the file has 100 rows with columns a,b"
    )
    red_new = exploration_redundancy(
        "revenue is stored in cents, not dollars", "the file has 100 rows with columns a,b"
    )
    print(f"redundancy PMI/token  repeat={red_same:.4f}  novel={red_new:.4f}  "
          f"(stop above {cfg.redundancy_threshold})")
    if red_same <= red_new:
        print("  WARN: a repeated observation did not score more redundant than a novel one.")

    print("OK: confidence engine is working.")


if __name__ == "__main__":
    main()
