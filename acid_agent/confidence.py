"""Confidence engine (paper §2.2.2).

Confidence(C | context) = exp(mean token-level log-prob of C given context),
computed by a small local model (Qwen3-0.6B) because API providers hide logprobs.

Three divergence measures used by the validation gate:
  * decision_divergence      — executed decisions scored with vs. without evidence (min over decisions; low => ungrounded => retry)
  * code_span_divergence     — decision-relevant code spans scored with vs. without evidence (max over spans; low => retry)
  * exploration_redundancy   — new observation scored with vs. without prior observations (high => redundant => stop exploring)
"""

import ast
import math

from .config import get_settings

_scorer = None


class ConfidenceScorer:
    def __init__(self):
        self.model_name = get_settings().confidence_model
        self.device = get_settings().confidence_device
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is not None:
            return
        import os

        # Reduce fragmentation OOM on small GPUs (MX450 2GB) before torch loads
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype
        ).to(device)
        self._torch = torch
        self._device = device

    def score(self, target: str, context: str = "") -> float:
        """exp(mean token logprob of target given context), in [~0, 1]."""
        self._load()
        assert self._tok is not None and self._model is not None
        tok, torch_ = self._tok, self._torch
        if not target.strip():
            return 0.0
        ctx_ids = tok(context[-4000:], add_special_tokens=False).input_ids[-1200:]
        tgt_ids = tok(target, add_special_tokens=False).input_ids[:512]
        ids = torch_.tensor([ctx_ids + tgt_ids], device=self._device)
        with torch_.no_grad():
            logits = self._model(ids).logits[0]
        logprobs = torch_.log_softmax(logits.float(), dim=-1)
        start = len(ctx_ids)
        lps = [
            logprobs[start - 1 + i, t].item() for i, t in enumerate(tgt_ids) if start - 1 + i >= 0
        ]
        if not lps:
            return 0.0
        return math.exp(sum(lps) / len(lps))


def get_scorer() -> ConfidenceScorer:
    global _scorer
    if _scorer is None:
        _scorer = ConfidenceScorer()
    return _scorer


def divergence(conf_with: float, conf_without: float) -> float:
    """Relative gap between two confidence scores, in [0, 1].

    Raw |diff| of exp(mean-logprob) values is tiny in practice (small-model
    confidences live near 0), so we normalize by the larger score. This keeps
    the paper's threshold semantics (0.25 / 0.50 / 0.45) meaningful:
    0 => evidence changed nothing, 1 => evidence changed everything.
    """
    denom = max(conf_with, conf_without, 1e-9)
    return abs(conf_with - conf_without) / denom


def _ctx_with_evidence(task: str, evidence_summary: str, kind: str) -> str:
    return f"""Task: {task}

Evidence:
{evidence_summary}

{kind}:"""


def _ctx_without_evidence(task: str, kind: str) -> str:
    return f"""Task: {task}

{kind}:"""


def decision_divergence(decisions: list[str], evidence_summary: str, task: str) -> float:
    """Min over executed decisions of |P(d | task+evidence) - P(d | task)|.
    Low value => no decision gains real support from the evidence => retry."""
    s = get_scorer()
    divs = []
    for d in decisions:
        with_e = s.score(d, _ctx_with_evidence(task, evidence_summary, "Decision"))
        without_e = s.score(d, _ctx_without_evidence(task, "Decision"))
        divs.append(divergence(with_e, without_e))
    return min(divs) if divs else 0.0


def extract_code_spans(code: str, max_spans: int = 6) -> list[str]:
    """Decision-relevant spans via static analysis: control-flow blocks + key pandas calls."""
    spans: list[str] = []
    lines = code.splitlines()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [code.strip()[:500]] if code.strip() else []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While)):
            seg = " ".join(
                ln.strip() for ln in lines[node.lineno - 1 : node.end_lineno] if ln.strip()
            )
            if 0 < len(seg) <= 500:
                spans.append(seg)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {
                "merge", "groupby", "join", "pivot_table", "dropna",
                "fillna", "corr", "resample", "rolling", "agg",
            }:
                line = lines[node.lineno - 1].strip()
                if line:
                    spans.append(line[:300])
    seen, out = set(), []
    for sp in spans:
        if sp not in seen:
            seen.add(sp)
            out.append(sp)
    return out[:max_spans]


def code_span_divergence(code: str, evidence_summary: str, task: str) -> float:
    """Max over decision-relevant spans of |P(span | task+evidence) - P(span | task)|.
    Low max => code not grounded in evidence => retry."""
    s = get_scorer()
    spans = extract_code_spans(code)
    if not spans:
        return 1.0  # nothing decision-relevant found; don't block on this signal
    divs = []
    for sp in spans:
        with_e = s.score(sp, _ctx_with_evidence(task, evidence_summary, "Code"))
        without_e = s.score(sp, _ctx_without_evidence(task, "Code"))
        divs.append(divergence(with_e, without_e))
    return max(divs)


def exploration_redundancy(new_observation: str, prior_summaries: str) -> float:
    """|P(new_obs | priors) - P(new_obs)|. High => new obs adds little beyond priors => stop."""
    s = get_scorer()
    ctx_with = f"""Prior findings:
{prior_summaries}

New observation:"""
    with_priors = s.score(new_observation, ctx_with)
    without_priors = s.score(new_observation, "New observation:")
    return divergence(with_priors, without_priors)