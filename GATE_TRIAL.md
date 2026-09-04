# Transaction Gate Trial — 2026-09-03

A/B test of the semantic transaction gate (`claude-acid`) against the same model with no
gate (`claude`), on KramaBench `archeology`.

- **Model:** `claude-sonnet-4-6`, both arms
- **Runs:** 89 baseline (easy tier) + 12 baseline (hard probe) + 16 ACID
- **Thresholds:** decision ≥ 0.25 · code-span ≥ 0.35 (`.env`), plus a 0.50 re-run
- **Reproduce:** `python scripts/regrade.py --suite krama --domain archeology`,
  `python scripts/gate_stats.py`, `python scripts/compare_arms.py`

---

## 1. Result

`archeology-easy-4` — the only task in the easy tier where the ungated baseline is inconsistent.
Ten runs per arm, identical model, workspace and grader.

| arm | runs | accuracy | variance |
|---|---|---|---|
| baseline (no transaction machinery) | `0 1 1 1 1 1 0 1 1 1` | **80.0%** | 0.178 |
| ACID (gate on, span ≥ 0.35) | `1 1 1 1 1 1 1 1 1 1` | **100.0%** | 0.000 |

Wall clock: 0.60 min/run baseline vs 6.43 min/run ACID — a **10.7×** cost.

### This difference is NOT statistically significant

```
Fisher exact test, two-tailed:  p = 0.474
```

Ten-for-ten against eight-for-ten is a two-observation difference. At these rates the
comparison would need roughly **30 runs per arm** to reach p < 0.05 (n=25 → p=0.050;
n=30 → p=0.024). The direction is consistent and reproduced across 16 ACID runs spanning four
configurations, none of which ever failed this task — but "ACID improves accuracy" is not a
claim this data supports yet.

The consistency difference is the more defensible half: baseline variance 0.178 against 0.000,
and baseline failure was reproduced twice in ten runs rather than once in three.

---

## 2. Why only two tasks count

Baseline swept the full easy tier first — 6 tasks × 3 runs.

| task | runs | score | variance | verdict |
|---|---|---|---|---|
| `archeology-easy-3` | 1 · 1 · 1 | 100.0% | 0.000 | ceiling |
| **`archeology-easy-4`** | 0 · 1 · 1 | 66.7% | **0.333** | **discriminating** |
| `archeology-easy-6` | 1 · 1 · 1 | 100.0% | 0.000 | ceiling |
| **`archeology-easy-8`** | 0 · 0 · 0 | 0.0% | 0.000 | **floor** |
| `archeology-easy-10` | 1 · 1 · 1 | 100.0% | 0.000 | ceiling |
| `archeology-easy-11` | 1 · 1 · 1 | 100.0% | 0.000 | ceiling |
| **tier** | — | **77.8%** | **0.236** | 1 of 6 varies |

The tier consistency figure of 0.236 is generated **entirely** by `easy-4` — every other task
has zero variance. A tier-wide consistency number is one task wearing a coat of five others.

The synthetic suite (`eval/kramabench.py`) was checked first and discarded: baseline scores
**100% (9/9)** there, including the mixed-date-format trap, which a current model sidesteps by
reasoning that dates are irrelevant to a temperature average. A gate cannot improve on perfect.

---

## 2b. The easy tier is saturated

After fixing the seed loader and the graders, the baseline arm was run across every gradeable
easy task in all six domains — 38 tasks, 89 runs.

| domain | tasks | runs | score | ceiling | floor | varies |
|---|---|---|---|---|---|---|
| archeology | 6 | 25 | 80.0% | 4 | 1 | **1** |
| astronomy | 4 | 8 | 100.0% | 4 | 0 | 0 |
| biomedical | 3 | 6 | 66.7% | 2 | 1 | 0 |
| environment | 6 | 12 | 83.3% | 5 | 1 | 0 |
| legal | 13 | 26 | 100.0% | 13 | 0 | 0 |
| wildfire | 6 | 12 | 100.0% | 6 | 0 | 0 |
| **total** | **38** | **89** | **89.9%** | **34** | **3** | **1** |

**An ungated one-shot `claude -p` scores 89.9% on this tier, and 34 of 38 tasks are at a
perfect ceiling.** There is almost no headroom for any architecture to demonstrate improvement,
and exactly one task in the tier (`archeology-easy-4`) shows any run-to-run variance — so the
paper's consistency claim can only be tested on a single task here.

The three floor tasks are not all agent failures:

- `archeology-easy-8` — under-specified ("unique sources" in a free-text bibliography admits
  several defensible normalizations; both arms answer deterministically and differently).
- `biomedical-easy-2` — a genuine decision failure: the agent excluded a patient with a missing
  age (13 of 14 records) and got 68.08 against an expected 68.5.
- `environment-easy-6` — a broken task: the question asks for 2002–2023 inclusive but
  `data_sources` only ships through 2022.

This is the trial's main limitation, and it is a property of the benchmark tier rather than of
the implementation. Meaningful evaluation of the ACID machinery needs the **hard** tier.

---

## 2c. The hard tier has the headroom (partial)

A baseline probe of the `archeology` hard tier — 6 gradeable tasks, 2 runs each — for contrast
with the saturated easy tier:

| task | runs | score | verdict |
|---|---|---|---|
| `archeology-hard-1` | 0 · 0 | 0.0% | floor |
| `archeology-hard-2` | 0 · 0 | 0.0% | floor |
| `archeology-hard-5` | 1 · 1 | 100.0% | ceiling |
| **`archeology-hard-7`** | 1 · 0 | **50.0%** | **discriminating** |
| `archeology-hard-9` | 0 · 0 | 0.0% | floor |
| `archeology-hard-12` | 0 · 0 | 0.0% | floor |
| **tier** | — | **25.0%** | |

Baseline scores **25.0%** here against **89.9%** on the easy tier. This is where the ACID
machinery could actually be tested, and `archeology-hard-7` is a second task showing genuine
run-to-run variance — the property the paper's consistency claim needs.

This probe was cut short for budget reasons; biomedical and astronomy hard were not completed,
and no ACID arm was run on `hard-7`. **That is the single highest-value next experiment:**
baseline to n=10 and ACID to n=5 on `archeology-hard-7` would give a two-task result on a tier
that is not saturated.

---

## 3. What the gate actually did

All 13 validation attempts across the 6 ACID runs at threshold 0.35:

```
run          unit    verdict   decision   code-span   note
easy-4 r0    u0 a1   PASS      0.863      0.370
easy-4 r0    u1 a1   PASS      0.891      1.000
easy-4 r1    u0 a1   PASS      0.872      0.453
easy-4 r1    u1 a1   PASS      0.884      0.449
easy-4 r2    u0 a1   PASS      0.936      0.361
easy-4 r2    u1 a1   REJECT    0.564      1.000     reflection red
easy-4 r2    u1 a2   PASS      0.316      1.000     retry accepted
easy-8 r0    u0 a1   PASS      0.705      0.642
easy-8 r0    u1 a1   REJECT    0.640      0.909     "counting unsplit compound
                                                     strings (872) contradicts…"
easy-8 r0    u1 a2   PASS      0.460      0.873     retry accepted
easy-8 r1    u0 a1   PASS      0.898      0.830
easy-8 r1    u1 a1   PASS      0.854      0.847
easy-8 r2    u0 a1   PASS      0.549      1.000
```

**13 attempts, 2 rejected (15%). Both by LLM reflection. Zero by either divergence signal.**

- Decision divergence ranged 0.316–0.936 against a 0.25 threshold — never the limiting signal.
- Code-span divergence never fell below the configured 0.35.
- The paper's novel contribution — confidence divergence from a local model's token
  log-probabilities — did **no gating work** in this trial. The work was done by a
  conventional LLM self-critique step.

The one rejection with visible reasoning is instructive: on `easy-8` the gate caught a unit
that had decided to count unsplit compound strings, rolled the workspace back, and the retry
passed. The transaction loop executed exactly as designed. The answer was still wrong — **82**
against a truth of **52** — because the gate enforces coherence with gathered evidence, not
correctness.

---

## 4. Threshold sensitivity

`.env` sets `CODE_SPAN_DIVERGENCE_MIN=0.35`; `config.py`'s default is `0.50`. `easy-4` was
re-run three times at 0.50, tagged `span050` for fresh workspaces and its own memory scope.

| threshold | score | attempts | rejected | mean run | units failed |
|---|---|---|---|---|---|
| span ≥ 0.35 | 100.0% | 7 | 1 (14%) | 9.30 min | none |
| span ≥ 0.50 | 100.0% | 9 | 4 (44%) | 10.27 min | 1 unit failed |

**Tripling the gate's activity changed nothing about the answer.** Run times were
8.0 / 6.1 / 13.8 min at 0.35 against 10.9 / 6.7 / 13.2 at 0.50 — overlapping ranges, no
separable cost difference at three runs each.

The third run at 0.50 is the most revealing in the trial:

```
u0 a1  REJECT  dd=0.695  span=0.230
u0 a2  REJECT  dd=0.481  span=0.300
u0 a3  REJECT  dd=0.465  span=0.435   -> max_retries exhausted, UNIT FAILED
u1 a1  REJECT  dd=0.500  span=0.314
u1 a2  PASS    dd=0.606  span=1.000
```

Unit 0 failed outright; the run still answered correctly. The failed unit rolled back to zero
trace, contributed nothing to `evidence_summary`, and the supervisor assembled a correct answer
from the unit that committed. **That is the atomicity design working under real pressure.**

*Retracted counterfactual:* four of the original 13 attempts had spans between 0.35 and 0.50,
inviting the reading that 0.50 "would have" rejected them. But `generate_execute` writes fresh
code every run and its spans range 0.230–1.000, so the threshold interacts with different code
each time. Only a direct re-run answers it — this one did.

---

## 4b. Ablation — is the gate doing the work?

`easy-4` re-run three times with `GATE_BYPASS=1`: all four signals still computed and
persisted, but every verdict forced to PASS, so no retry or rollback can fire. Tagged
`ablate` for fresh workspaces and memory scope.

| arm | score | n |
|---|---|---|
| baseline (no transaction machinery) | 80.0% | 10 |
| ACID, gate on (span ≥ 0.35) | 100.0% | 3 |
| ACID, gate on (span ≥ 0.50) | 100.0% | 3 |
| **ACID, gate forced OFF** | **100.0%** | 3 |

The ablation is not vacuous — it **overrode two real rejections**:

```
run 15:33 (gate bypassed)
  u0 a1   reflection=False              -> gate WOULD have rejected
  u1 a1   span=0.283  (< 0.35 min)      -> gate WOULD have rejected
  both units committed; answer CORRECT; 7.0 min
```

In the one ablation run where the gate would have fired twice, forcing it to pass produced a
correct answer immediately. **Those two rejections would have been false positives** — the gate
would have rolled back and retried work that was already right.

**Conclusion.** The accuracy gain over baseline survives with the gate completely disabled.
Together with the threshold study — where tripling the rejection rate (14% → 44%) also changed
nothing — the evidence points one way: the gain comes from **transactional decomposition plus
read-only exploration**, not from validation. The gate's only measured effects in this trial are
two false-positive rejections and a ~10× wall-clock cost.

Caveat: n=3 for the ablation, with overrides occurring in one of the three runs. This is a
clean demonstration, not a powered study.

---

## 5. Harness defects found and fixed

Six bugs surfaced during setup. Five would have corrupted the numbers rather than crashed —
silently, by producing scores that looked like agent failure.

1. **Numeric grader read the wrong number.** `_num_grader` took the *first* number in the
   reply — "across all 99 rows the average is 5.28" was graded against `99`. Three correct
   answers scored 0.0; the synthetic suite reported 66.7% when the true score was 100%. Now
   scans every number, matching `kramabench_tasks.grade_numeric`.
2. **Repeat runs shared one workspace.** The synthetic path built slugs without a run index, so
   all three runs of a task reused one directory *and* one memory scope. `Workspace.create`
   reuses an existing directory and commits what it finds, so runs 2–3 inherited run 1's
   committed `unit*.py`. The consistency metric was measuring contamination.
3. **String grader failed on diacritics.** `grade_exact` did a raw substring match, so
   "Sao Paulo" scored 0.0 against `São Paulo` — about to fire on two tasks in this sweep. Now
   casefolds and strips combining marks.
4. **No checkpointing on multi-hour evals.** Both eval paths saved only after every run
   finished; an interrupted ACID arm would have written nothing. Both now checkpoint per run
   and report `runs_completed`.
5. **No isolation between configurations.** Re-running a task under different settings reused
   the previous workspace and memory. Added `--tag`.
6. **Seed files silently failed to load.** `seed_files` only checked the exact relative path
   under `input/`, and ignored glob entries. `legal/` nests its 135 files in subdirectories, so
   **all 14 of its easy tasks seeded an empty workspace** and would have scored 0.00 — looking
   like agent failure, not a loader bug. `get_available_domains()` still reported legal as
   available, because it only checks that the workload JSON and input directory exist. Now
   resolves by exact path, glob, directory recursion, then basename. All 42 easy tasks across
   6 domains load data.

New tools: `scripts/regrade.py` (re-score history from Postgres, no re-runs),
`scripts/gate_stats.py` (what the gate did), `scripts/compare_arms.py` (A/B table).

---

## 6. What this does and does not show

**Supported by the data:**
- The easy tier is saturated: an ungated one-shot scores 89.9% over 38 tasks / 89 runs, with
  34 tasks at a perfect ceiling and exactly one showing run-to-run variance.
- On that one task, ACID never failed in 16 runs across four configurations; baseline failed
  2 of 10.
- The transaction loop works mechanically — reflection rejected an attempt, the workspace rolled
  back, the retry was accepted.
- Atomicity holds under real failure: at span ≥ 0.50 a unit was rejected three times and failed
  outright, and the run still produced a correct answer from the units that committed.
- The confidence-divergence signals were never the binding constraint in 13 attempts at 0.35.

**Not supported:**
- **Statistical significance.** 10/10 vs 8/10 gives Fisher exact p = 0.474. This is an
  underpowered result, not a demonstrated improvement.
- **That the gate causes the gain.** It rejected nothing on the deciding task. Forcing every
  verdict to PASS preserved 100% while overriding two rejections that would have been false
  positives. Tripling the rejection rate (14% → 44%) changed nothing.
- **Any generalization.** One discriminating task in the easy tier; the hard-tier probe found a
  second (`archeology-hard-7`) but no ACID arm was run on it.
- **That the gate improves correctness.** On `easy-8` it fired, forced a revision, and the answer
  moved from 55 to 82 against a truth of 52.

**Honest one-line summary.** The transactional scaffolding — decomposition plus read-only
exploration — is the plausible source of the observed advantage; the validation gate, which is
the paper's actual contribution, has no evidence supporting it here and measurable costs
(two false-positive rejections, ~10× wall clock).

## 7. Next

1. **Power the `easy-4` result.** Ten runs per arm on that task (~1 h of ACID time) turns a
   one-flip signal into something reportable.
2. **Ablate the gate.** Run the ACID arm with validation forced to always pass. If the accuracy
   gain survives, decomposition is doing the work and the mechanism claim does not hold here.
   The threshold study already points this way — 44% and 14% rejection produced identical scores.
3. **Find more discriminating tasks.** Four of six easy tasks were useless. Baseline is ~0.6
   min/run — sweep the other five domains cheaply, then spend ACID time only where baseline is
   imperfect but not hopeless.
4. **Drop `easy-8` or fix its ground truth** before it dilutes another average.

---

# 2026-09-04 — Hard-tier follow-up: the gate tested where it could win

Items 2 and 3 above are now executed, on `archeology-hard-7` (baseline 1/2 — the one
hard-tier task that is imperfect but not hopeless). NOTE: `--task-ids` takes **list
indices**, not task-id numbers — `archeology-hard-7` is index **6**.

Four arms, fresh runs (baseline merges with the Sep-3 hard sweep for n=5):

| arm | hard-7 | n | answers seen |
|---|---|---|---|
| `claude` (harness control) | **60%** (3/5) | 5 | 274 ×3, 295 ×1, 0 ×1 |
| `claude-react` (first data ever) | 33% (1/3) | 3 | 274, 295, FAIL |
| `claude-acid` + `GATE_BYPASS=1` (decomposition only) | **0%** (0/3) | 3 | 294/295 every run |
| `claude-acid`, full gate (reference semantics) | **0%** (0/3) | 3 | 294/295 every run |

Fisher exact: acid vs claude p=0.061; react vs claude p=1.0.

## Findings

1. **The transactional arm is worse than both controls on the discriminating task.**
   All six ACID runs (bypass and gate alike) answered 294 or 295: decision extraction
   commits to "within 0.1 degrees in both latitude and longitude" — a box (Chebyshev)
   neighborhood — while the grader's 274 counts Euclidean distance. Six runs, same
   wrong method, zero variance. The paper's consistency advantage appears here as
   *consistently wrong*: the transaction commits early to one interpretation and the
   unit loop cannot re-open it, while the harness sometimes re-derives the method.

2. **The gate never fired where it mattered.** Under reference semantics the deciding
   units scored span surprise ≤ 0.11 (retry threshold 0.50); 0 rejections in 7 gate
   attempts. Reflection — scoped to the unit goal — explicitly endorsed the wrong
   method ("Chebyshev distances computed correctly via full cross-product"). Every
   gate component measures *evidence↔code consistency*, not *method correctness*:
   a faithful implementation of a wrong decision passes all four components.

3. **Combined with the easy-tier ablation, the gate now has no effect in either
   direction**: it does not cause the easy-tier gain (bypass preserved 100%) and it
   does not prevent the hard-tier failure (full gate = bypass = 0%). The claim the
   data supports: decomposition + rollback helps where tasks are short, and costs
   the exploratory freedom the harness uses to recover from a wrong first
   interpretation on hard ones.

4. **Calibration** (59 attempts, 22 runs, `scripts/calibrate_gate.py`): paper@0.50
   rejects 33% of attempts, paper@0.25 rejects 11%, reference@0.50 rejects 11%.
   `.env` now pins `GATE_SEMANTICS=reference` (the authors' released rule) with
   `SPAN_DIVERGENCE_MIN=0.25` recorded as the calibrated paper-mode fallback.

## Honest one-line summary (updated)

Decomposition helps on easy tasks and hurts on hard ones; the validation gate is
orthogonal to both — it polices self-consistency, which is exactly the thing a
confidently wrong transaction has.
