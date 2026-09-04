"""Confidence engine (paper §2.2.2), aligned with the reference implementation.

All signals are computed as **mean token log-probabilities in nats** and compared
as raw deltas. Nothing is exponentiated before thresholding.

    logp(C | context) = mean token log-prob of C given context

Three probes, mirroring `vendor/acid-paper-ref/da_agent/utils/`:

  * ``code_span_surprise``  (their ``evidence_surprise.py`` / paper §2.2.2)
        delta = logp(span | task + evidence + code_prefix) - logp(span | task + code_prefix)
    The paper and the released code disagree on what to do with that delta, so
    both metrics are computed and `Settings.gate_semantics` picks the verdict:
        reference: surprise   = max(0, -delta)          -> retry ABOVE 0.50
        paper:     divergence = |C_w - C_wo|/max(...)   -> retry BELOW 0.50
    See `Settings.gate_semantics` for the citation on each side.

  * ``probability_contrast`` (their ``probability_contrast.py``)
        ratio = exp(logp(current_policy) - logp(alternative_policy))
    Both scored as continuations of the *same* prefix. Probes exist only for
    conflicts the backbone LLM already flagged, so this is a referee on an
    identified disagreement, not a general-purpose detector.
    ratio < 0.25 => retry, < 0.75 => watch.

  * ``decision_surprise``   (their ``anchor_decision_surprise.py``)
        Same delta form, over extracted decisions instead of code spans.
    **Diagnostic only** — the reference states it "never decides retry policy by
    itself", so it is recorded and never gates.

  * ``exploration_redundancy`` (their ``adaptive_exploration_metric.py``)
        PMI/token = [logp(obs | prior_obs) - logp(obs)] / n_tokens
    High PMI means the new observation is already predicted by a previous one =>
    redundant => stop exploring.

Historical note: this module used to exponentiate to probabilities, take
|a - b|, and normalize by max(a, b) — which discarded the sign, so evidence
*supporting* the code and evidence *contradicting* it produced the same number,
and the gate then required that number to be HIGH to pass. That is backwards
from the reference on both counts.
"""

import ast
import math
import re

from .config import get_settings

_scorer = None

NEG_INF = float("-inf")


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

        # Reduce fragmentation OOM on small GPUs (MX450 ~1.65GB, shared with the
        # desktop) before torch loads; fall back to CPU when CUDA can't fit.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.float16 if device == "cuda" else torch.float32
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=dtype
            ).to(device)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            device = "cpu"
            dtype = torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=dtype
            ).to(device)
        self._torch = torch
        self._device = device

    def score_logp(self, target: str, context: str = "") -> float:
        """Mean token log-prob of `target` given `context`, in nats (<= 0).

        Returns -inf for an empty target so callers can drop it rather than
        silently treating "no tokens" as a confident 0.0 the way exp() did.
        """
        self._load()
        assert self._tok is not None and self._model is not None
        tok, torch_ = self._tok, self._torch
        if not target.strip():
            return NEG_INF
        ctx_ids = tok(context[-4000:], add_special_tokens=False).input_ids[-1200:]
        tgt_ids = tok(target, add_special_tokens=False).input_ids[:512]
        try:
            return self._score_ids(ctx_ids, tgt_ids)
        except torch_.OutOfMemoryError:
            # Mid-scoring OOM used to escape to validation.py, which swallows it
            # and lets the unit pass ungated — the gate would silently turn off
            # mid-run. Migrate to CPU once and keep scoring for real.
            self._fallback_to_cpu()
            return self._score_ids(ctx_ids, tgt_ids)

    def score(self, target: str, context: str = "") -> float:
        """exp(score_logp) — probability form. Kept for reporting only.

        Never threshold on this: exp() of a small model's mean logprob lives near
        zero, which is exactly what forced the old relative normalization.
        """
        lp = self.score_logp(target, context)
        return 0.0 if lp == NEG_INF else math.exp(lp)

    def _fallback_to_cpu(self):
        assert self._model is not None
        self._torch.cuda.empty_cache()
        self._model = self._model.to("cpu").float()
        self._device = "cpu"

    def _score_ids(self, ctx_ids: list[int], tgt_ids: list[int]) -> float:
        assert self._model is not None
        torch_ = self._torch
        ids = torch_.tensor([ctx_ids + tgt_ids], device=self._device)
        with torch_.no_grad():
            logits = self._model(ids).logits[0]

        # Only rows that predict a target token matter. Taking log_softmax over
        # the whole sequence materializes seq x vocab in fp32 (~1 GB here) and
        # OOMs a small GPU; slicing + chunking caps it at a few tens of MB.
        start = len(ctx_ids)
        i0 = max(0, 1 - start)  # row start-1+i must be a real position
        rows = logits[start - 1 + i0 : start - 1 + len(tgt_ids)]
        targets = tgt_ids[i0:]
        if not targets:
            return NEG_INF

        total, n = 0.0, 0
        for off in range(0, len(targets), 64):
            block = rows[off : off + 64].float()
            lp = torch_.log_softmax(block, dim=-1)
            idx = torch_.tensor(targets[off : off + 64], device=lp.device).unsqueeze(1)
            total += lp.gather(1, idx).squeeze(1).sum().item()
            n += idx.shape[0]
        if not n:
            return NEG_INF
        return total / n


def get_scorer() -> ConfidenceScorer:
    global _scorer
    if _scorer is None:
        _scorer = ConfidenceScorer()
    return _scorer


def _finite(x: float) -> bool:
    return x is not None and x != NEG_INF and math.isfinite(x)


def surprise(logp_with_evidence: float, logp_without_evidence: float) -> float:
    """REFERENCE metric. max(0, -(logp_with - logp_without)) in nats.

    High => the evidence SUPPRESSES the span => the code contradicts what
    exploration found => retry. One-sided on purpose: evidence that makes a span
    MORE likely is support, and support is not a defect, so it clamps to 0.
    """
    if not (_finite(logp_with_evidence) and _finite(logp_without_evidence)):
        return 0.0
    return max(0.0, -(logp_with_evidence - logp_without_evidence))


def relative_divergence(logp_with_evidence: float, logp_without_evidence: float) -> float:
    """PAPER metric (§2.2.2). |C_with - C_without| / max(C_with, C_without).

    C = exp(mean token log-prob), per the paper's definition of confidence.
    LOW => the evidence changed nothing => the code is not grounded in it =>
    retry. Symmetric and in [0, 1].

    Normalized by the larger score because raw differences of exp(mean logprob)
    from a 0.6B model sit near zero, which would put every attempt under any
    threshold the paper states. The paper does not specify a normalization; this
    is the reading that makes its 0.25 / 0.50 / 0.45 numbers usable.
    """
    if not (_finite(logp_with_evidence) and _finite(logp_without_evidence)):
        return 1.0  # unscorable => don't block on this signal
    a, b = math.exp(logp_with_evidence), math.exp(logp_without_evidence)
    return abs(a - b) / max(a, b, 1e-9)


# ---------------------------------------------------------------- prefixes

def _cond_prefix(task: str, evidence: str, code_prefix: str) -> str:
    task_part = f"# TASK\n{task.strip()}\n\n" if task.strip() else ""
    return (
        task_part
        + "# EXPLORATION SUMMARY\n"
        + evidence.strip()
        + "\n\n# GENERATED PYTHON CODE PREFIX\n"
        + "The following code continuation was generated by the agent.\n"
        + "```python\n"
        + code_prefix
    )


def _prior_prefix(task: str, code_prefix: str) -> str:
    task_part = f"# TASK\n{task.strip()}\n\n" if task.strip() else ""
    return (
        task_part
        + "# GENERATED PYTHON CODE PREFIX\n"
        + "The following code continuation was generated by the agent.\n"
        + "```python\n"
        + code_prefix
    )


def _decision_cond_prefix(task: str, evidence: str) -> str:
    task_part = f"# TASK\n{task.strip()}\n\n" if task.strip() else ""
    return task_part + "# EXPLORATION SUMMARY\n" + evidence.strip() + "\n\n# DECISION\n"


def _decision_prior_prefix(task: str) -> str:
    task_part = f"# TASK\n{task.strip()}\n\n" if task.strip() else ""
    return task_part + "# DECISION\n"


def _contrast_prefix(task: str, evidence: str, decision_type: str) -> str:
    readable = re.sub(r"[_-]+", " ", decision_type or "decision").strip()
    return (
        f"# TASK\n{task.strip()}\n\n# EVIDENCE\n{evidence.strip()}\n\n"
        "Given the task and evidence, choose the safer computational policy "
        f"for {readable}.\nPolicy:"
    )


# ---------------------------------------------------------------- code spans

class CodeSpan:
    """A decision-relevant statement plus the code that precedes it."""

    __slots__ = ("text", "line_no", "category", "code_prefix")

    def __init__(self, text: str, line_no: int, category: str, code_prefix: str):
        self.text = text
        self.line_no = line_no
        self.category = category
        self.code_prefix = code_prefix

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CodeSpan(line={self.line_no}, category={self.category!r}, text={self.text!r})"


_SKIP_STMTS = (ast.Import, ast.ImportFrom, ast.Pass, ast.Break, ast.Continue)
_OUTPUT_NAMES = {"print", "display"}
_OUTPUT_ATTRS = {"show", "head", "tail", "info", "describe"}


def _split_semicolon_lines(code: str) -> str:
    if "\n" in code or code.count(";") < 2:
        return code
    return "\n".join(part.strip() for part in code.split(";") if part.strip())


def _line_offsets(code: str) -> list[int]:
    offsets, cursor = [0], 0
    for line in code.splitlines(keepends=True):
        cursor += len(line)
        offsets.append(cursor)
    if not code.endswith("\n"):
        offsets.append(len(code))
    return offsets


def _offset_for(offsets: list[int], line_no: int, col: int) -> int:
    if line_no <= 0:
        return 0
    return min(offsets[min(line_no - 1, len(offsets) - 1)] + col, offsets[-1])


def _node_range(code: str, offsets: list[int], node: ast.AST) -> tuple[int, int]:
    start = _offset_for(offsets, getattr(node, "lineno", 1), getattr(node, "col_offset", 0))
    end = _offset_for(
        offsets,
        getattr(node, "end_lineno", getattr(node, "lineno", 1)),
        getattr(node, "end_col_offset", getattr(node, "col_offset", 0)),
    )
    if end <= start:
        end = start + len(ast.get_source_segment(code, node) or "")
    return start, min(end, len(code))


def _is_output_expr(node: ast.AST) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    if isinstance(func, ast.Name):
        return func.id in _OUTPUT_NAMES
    if isinstance(func, ast.Attribute):
        return func.attr in _OUTPUT_ATTRS
    return False


def _assignment_value(node: ast.AST) -> ast.AST | None:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return node.value
    return None


def _is_empty_container_init(value: ast.AST | None) -> bool:
    if value is None:
        return False
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        return not value.elts
    if isinstance(value, ast.Dict):
        return not value.keys
    if not isinstance(value, ast.Call) or value.args or value.keywords:
        return False
    func = value.func
    if isinstance(func, ast.Name):
        return func.id in {"list", "dict", "set", "tuple"}
    return isinstance(func, ast.Attribute) and func.attr == "DataFrame"


def _skip_statement(node: ast.AST) -> bool:
    if isinstance(node, _SKIP_STMTS):
        return True
    if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)):
        return True  # docstring
    if _is_output_expr(node):
        return True
    return _is_empty_container_init(_assignment_value(node))


def _feature_counts(node: ast.AST) -> dict[str, int]:
    f = dict.fromkeys(
        ("comparison", "boolean_logic", "subscript", "calculation", "function_call",
         "control_flow", "comprehension", "assignment", "return"), 0)
    for child in ast.walk(node):
        if isinstance(child, ast.Compare):
            f["comparison"] += 1
        elif isinstance(child, ast.BoolOp):
            f["boolean_logic"] += 1
        elif isinstance(child, ast.Subscript):
            f["subscript"] += 1
        elif isinstance(child, ast.BinOp):
            f["calculation"] += 1
        elif isinstance(child, ast.Call):
            f["function_call"] += 1
        elif isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try,
                               ast.With, ast.AsyncWith)):
            f["control_flow"] += 1
        elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            f["comprehension"] += 1
        elif isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            f["assignment"] += 1
        elif isinstance(child, ast.Return):
            f["return"] += 1
    return f


def _category(f: dict[str, int], node: ast.AST) -> str:
    if f["control_flow"]:
        return "control_flow"
    if f["comparison"] and f["subscript"]:
        return "conditional_selection"
    if f["comparison"]:
        return "comparison"
    if f["comprehension"]:
        return "comprehension"
    if f["calculation"]:
        return "calculation"
    if f["subscript"]:
        return "selection"
    if f["function_call"]:
        return "function_call"
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return "assignment"
    if isinstance(node, ast.Return):
        return "return"
    return "statement"


def _decision_nodes(tree: ast.Module) -> list[ast.AST]:
    """Top-level statements, descending one level into def/class bodies."""
    nodes: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nodes.extend(getattr(node, "body", []) or [])
        else:
            nodes.append(node)
    return nodes


def extract_code_spans(code: str, max_spans: int = 10) -> list[str]:
    """Decision-relevant span texts."""
    return [sp.text for sp in extract_code_span_objects(code, max_spans)]


def extract_code_span_objects(code: str, max_spans: int = 10) -> list[CodeSpan]:
    """Every non-trivial AST statement, with the source that precedes it.

    Matches the reference: it takes statements generally and *subtracts* the ones
    that carry no analytical decision (imports, docstrings, prints, empty
    container init) rather than matching an allow-list of pandas method names.
    A hand-picked allow-list silently found zero spans on ordinary code — e.g.
    `pd.to_datetime(...)`, the exact call the mixed-date-format trap turns on —
    and a probe with no spans is a gate component that never fires.

    Each span carries the code before it as `code_prefix`, so it is scored in the
    position it actually occupies rather than standalone.
    """
    normalized = _split_semicolon_lines(code or "")
    try:
        tree = ast.parse(normalized)
    except SyntaxError:
        text = normalized.strip()[:500]
        return [CodeSpan(text, 1, "unparsed", "")] if text else []

    offsets = _line_offsets(normalized)
    spans: list[CodeSpan] = []
    seen: set[str] = set()
    for node in _decision_nodes(tree):
        if _skip_statement(node):
            continue
        start, end = _node_range(normalized, offsets, node)
        text = normalized[start:end].strip()
        if not text or text.startswith("#"):
            continue
        key = re.sub(r"\s+", " ", text)
        if key in seen:
            continue
        seen.add(key)
        spans.append(CodeSpan(
            text=text[:500],
            line_no=getattr(node, "lineno", 1),
            category=_category(_feature_counts(node), node),
            code_prefix=normalized[:start][-4000:],
        ))
    return spans[:max_spans]


def code_span_surprise(code: str, evidence_summary: str, task: str) -> dict:
    """Score decision-relevant code spans under BOTH gate semantics.

    Every span is scored once and both metrics are derived from the same two
    log-probs, so a single run yields the numbers for the paper rule and the
    reference rule at once — which is what makes the disagreement between them
    measurable rather than a matter of opinion.

      ``max_surprise``   reference: max over spans, retry ABOVE the threshold
      ``max_divergence`` paper:     max over spans, retry BELOW the threshold

    Both aggregate with max because the paper says "the *maximum* code-span
    confidence divergence is below 0.50" and the reference takes the max span
    surprise. Neither penalises a unit whose spans could not be scored.
    """
    s = get_scorer()
    spans = extract_code_span_objects(code, get_settings().max_code_spans)
    out: dict = {
        "available": False, "max_surprise": 0.0, "max_divergence": 1.0,
        "spans": [], "n_scored": 0,
    }
    if not spans:
        out["reason"] = "no decision-relevant span found"
        return out

    for sp in spans:
        cond = s.score_logp(sp.text, _cond_prefix(task, evidence_summary, sp.code_prefix))
        prior = s.score_logp(sp.text, _prior_prefix(task, sp.code_prefix))
        if not (_finite(cond) and _finite(prior)):
            continue
        delta = cond - prior
        out["spans"].append({
            "line_no": sp.line_no,
            "category": sp.category,
            "span": sp.text[:200],
            "cond_logp": round(cond, 4),
            "prior_logp": round(prior, 4),
            "delta": round(delta, 4),
            "surprise": round(surprise(cond, prior), 4),
            "divergence": round(relative_divergence(cond, prior), 4),
        })
        out["n_scored"] += 1

    if not out["n_scored"]:
        out["reason"] = "no span scored successfully"
        return out
    out["available"] = True
    out["max_surprise"] = round(max(sp["surprise"] for sp in out["spans"]), 4)
    out["max_divergence"] = round(max(sp["divergence"] for sp in out["spans"]), 4)
    return out


# ---------------------------------------------------------------- decisions

def decision_surprise(decisions: list[str], evidence_summary: str, task: str) -> dict:
    """Anchor-conditioned decision surprise. DIAGNOSTIC ONLY — never gates.

    Mirrors the reference's ``anchor_decision_surprise``, which documents itself
    as "diagnostic-only: it never decides retry policy by itself". Recorded so
    the relationship between decision support and outcomes stays measurable.
    """
    s = get_scorer()
    out: dict = {"available": False, "max_surprise": 0.0, "decisions": [], "n_scored": 0}
    if not decisions:
        out["reason"] = "no decisions extracted"
        return out

    cond_pref = _decision_cond_prefix(task, evidence_summary)
    prior_pref = _decision_prior_prefix(task)
    for d in decisions:
        cond = s.score_logp(d, cond_pref)
        prior = s.score_logp(d, prior_pref)
        if not (_finite(cond) and _finite(prior)):
            continue
        delta = cond - prior
        sup = max(0.0, -delta)
        out["decisions"].append({
            "decision": d[:200],
            "delta": round(delta, 4),
            "surprise": round(sup, 4),
            "support_status": _support_status(sup),
        })
        out["n_scored"] += 1

    if not out["n_scored"]:
        out["reason"] = "no decision scored successfully"
        return out
    out["available"] = True
    out["max_surprise"] = round(max(d["surprise"] for d in out["decisions"]), 4)
    return out


def _support_status(surprise_value: float) -> str:
    s = get_settings()
    if surprise_value > s.decision_surprise_retry:
        return "unsupported"
    if surprise_value > s.decision_surprise_warn:
        return "weak_support"
    return "supported"


# ---------------------------------------------------- probability contrast

def probability_contrast(conflicts: list[dict], task: str, evidence_summary: str) -> dict:
    """Score LLM-flagged evidence-vs-code conflicts as a two-option choice.

    `conflicts` come from the backbone's anchor-alignment pass; each carries the
    policy the code implements (`current_policy`) and the one the evidence
    supports (`expected_policy`). Both are scored as continuations of the SAME
    prefix, so the ratio is a direct preference between two competing policies
    rather than an ablation of the context.

    No flagged conflict => no probe => no signal. That is the point: the small
    model arbitrates a disagreement the big model already found.
    """
    s = get_scorer()
    cfg = get_settings()
    out: dict = {"available": False, "min_ratio": None, "probes": [], "n_scored": 0}
    if not conflicts:
        out["reason"] = "no LLM-flagged evidence/code conflict"
        return out

    for c in conflicts[: cfg.max_contrast_probes]:
        current = (c.get("current_policy") or "").strip()
        expected = (c.get("expected_policy") or "").strip()
        if not current or not expected or current.lower() == expected.lower():
            continue
        prefix = _contrast_prefix(task, evidence_summary, c.get("decision_type") or "decision")
        lp_cur = s.score_logp(" " + current, prefix)
        lp_alt = s.score_logp(" " + expected, prefix)
        if not (_finite(lp_cur) and _finite(lp_alt)):
            continue
        delta = lp_cur - lp_alt
        try:
            ratio = math.exp(delta)
        except OverflowError:
            ratio = float("inf") if delta > 0 else 0.0
        out["probes"].append({
            "anchor_id": c.get("anchor_id", ""),
            "decision_type": c.get("decision_type", ""),
            "current": current[:200],
            "alternative": expected[:200],
            "current_logp": round(lp_cur, 4),
            "alternative_logp": round(lp_alt, 4),
            "delta": round(delta, 4),
            "ratio": round(ratio, 4),
            "status": _contrast_status(ratio),
        })
        out["n_scored"] += 1

    if not out["n_scored"]:
        out["reason"] = "no conflict probe could be scored"
        return out
    out["available"] = True
    out["min_ratio"] = round(min(p["ratio"] for p in out["probes"]), 4)
    return out


def _contrast_status(ratio: float) -> str:
    s = get_settings()
    if ratio < s.contrast_retry_ratio:
        return "retry"
    if ratio < s.contrast_warning_ratio:
        return "watch"
    return "pass"


# ---------------------------------------------------------------- exploration

def exploration_redundancy(new_observation: str, prior_summaries: str) -> float:
    """PMI per token of the new observation against the priors, in nats.

        PMI/tok = logp(obs | priors) - logp(obs)

    High => the priors already predict this observation => redundant => stop.
    Positive and unbounded above; compare against `redundancy_threshold`.
    """
    s = get_scorer()
    if not new_observation.strip() or not prior_summaries.strip():
        return 0.0
    cond = s.score_logp(
        new_observation,
        f"<observation>{prior_summaries[-4000:]}</observation>\n<observation>",
    )
    marg = s.score_logp(new_observation, "<observation>")
    if not (_finite(cond) and _finite(marg)):
        return 0.0
    return cond - marg
