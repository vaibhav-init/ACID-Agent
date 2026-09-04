"""What the transaction gate actually DID during a set of runs.

A score delta between arms is not evidence on its own — the gate has to be shown
rejecting attempts and forcing revisions. This reports, per run: units planned,
attempts made, gate rejections, which component was red, and the surprise values.

Usage:
  python scripts/gate_stats.py                      # last 24h, claude-acid
  python scripts/gate_stats.py --agent claude-acid --since-min 180
"""

import argparse
from statistics import mean

from acid_agent.config import get_conn, get_settings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="claude-acid")
    ap.add_argument("--since-min", type=int, default=1440)
    a = ap.parse_args()

    with get_conn() as c:
        runs = c.execute(
            """SELECT run_id, left(task_text, 60), status, final_answer,
                      round(extract(epoch from (coalesce(finished_at, now()) - started_at))/60.0, 1),
                      to_char(started_at, 'HH24:MI')
               FROM runs
               WHERE agent_type=%s AND started_at > now() - make_interval(mins => %s)
               ORDER BY started_at""",
            (a.agent, a.since_min),
        ).fetchall()

        if not runs:
            print(f"no {a.agent} runs in the last {a.since_min} min")
            return 1

        print(f"{a.agent}: {len(runs)} run(s)\n")
        cfg = get_settings()
        tot_att = tot_rej = tot_watch = 0
        all_span, all_div, all_dec, all_ratio = [], [], [], []

        for rid, task, status, ans, mins, t0 in runs:
            units = c.execute(
                "SELECT unit_index, status, attempts, goal FROM units WHERE run_id=%s ORDER BY unit_index",
                (rid,),
            ).fetchall()
            vals = c.execute(
                """SELECT unit_index, attempt, passed, execution_ok, reflection_ok,
                          max_span_surprise, max_span_divergence, gate_semantics,
                          contrast_min_ratio, decision_surprise,
                          review_decision, watchlist, left(feedback, 70)
                   FROM validations WHERE run_id=%s ORDER BY unit_index, attempt""",
                (rid,),
            ).fetchall()

            rej = [v for v in vals if not v[2]]
            tot_att += len(vals)
            tot_rej += len(rej)
            committed = sum(1 for u in units if u[1] == "committed")
            # An unfinished run is live or abandoned; elapsed time is the honest
            # signal (an ACID run takes ~15-20 min), so show it rather than guess.
            state = f"{status} ({mins}min elapsed)" if status == "running" else f"{status} in {mins}min"
            print(f"run {str(rid)[:8]}  started {t0}  {state}  "
                  f"units={len(units)} (committed {committed})  gate: {len(vals)} attempts, {len(rej)} rejected")

            for (ui, att, passed, ex_ok, rf_ok, span, div, sem, ratio, dec,
                 decision, watch, fb) in vals:
                if span is not None:
                    all_span.append(span)
                if div is not None:
                    all_div.append(div)
                if dec is not None:
                    all_dec.append(dec)
                if ratio is not None:
                    all_ratio.append(ratio)
                if decision == "watch":
                    tot_watch += 1
                red = []
                if not ex_ok:
                    red.append("exec")
                if not rf_ok:
                    red.append("reflection")
                # Direction matters: HIGH span surprise means the evidence
                # suppresses the code, LOW contrast ratio means the evidence
                # prefers the alternative. Both are the red end.
                # Which rule was in force decides which end is red.
                if sem == "reference" and span is not None and span > cfg.span_surprise_retry:
                    red.append(f"span_surprise={span:.3f}")
                if sem == "paper" and div is not None and div < cfg.span_divergence_min:
                    red.append(f"span_divergence={div:.3f}")
                if ratio is not None and ratio < cfg.contrast_retry_ratio:
                    red.append(f"contrast_ratio={ratio:.3f}")
                mark = "PASS" if passed else "REJECT"
                if passed and decision == "watch":
                    mark = "WATCH"
                extra = f"  red={','.join(red)}" if red else ""
                sps = f"{span:.3f}" if span is not None else "  n/a"
                dvs = f"{div:.3f}" if div is not None else "  n/a"
                rts = f"{ratio:.3f}" if ratio is not None else "  n/a"
                dcs = f"{dec:.3f}" if dec is not None else "  n/a"
                print(f"    u{ui} a{att}  {mark:6} [{sem or '?'}] surprise={sps} "
                      f"divergence={dvs} contrast={rts} dec={dcs}(diag){extra}")
                if watch:
                    print(f"           watch: {'; '.join(watch)[:100]}")
                if not passed and fb:
                    print(f"           feedback: {fb.strip()}")
            if ans:
                print(f"    answer: {' '.join(ans.split())[:110]}")
            print()

        print(f"TOTAL: {tot_att} gate attempts, {tot_rej} rejected "
              f"({tot_rej/tot_att*100:.0f}%), {tot_watch} passed-with-watch"
              if tot_att else "TOTAL: no gate attempts")
        if all_span:
            print(f"  span_surprise   n={len(all_span)} mean={mean(all_span):.3f} "
                  f"max={max(all_span):.3f}  (retry above {cfg.span_surprise_retry})")
        elif tot_att:
            # The gate fails open, so a missing surprise means the local scorer
            # never loaded and validation ran on execution + reflection only.
            print("  span_surprise   NOT COMPUTED -> local scorer unavailable, "
                  "gate ran on execution + reflection only")
        else:
            print("  span_surprise   (no gate attempts yet - runs still in exploration)")
        if all_div:
            # Both metrics are recorded on every attempt, so the counterfactual
            # for the OTHER rule is free -- that is the paper-vs-code comparison.
            would_retry_paper = sum(1 for d in all_div if d < cfg.span_divergence_min)
            would_retry_ref = sum(1 for x in all_span if x > cfg.span_surprise_retry)
            print(f"  span_divergence n={len(all_div)} mean={mean(all_div):.3f} "
                  f"min={min(all_div):.3f}  (paper: retry below {cfg.span_divergence_min})")
            print(f"  counterfactual  paper rule would reject {would_retry_paper}/{len(all_div)}"
                  f", reference rule {would_retry_ref}/{len(all_span)}")
        if all_ratio:
            print(f"  contrast_ratio  n={len(all_ratio)} mean={mean(all_ratio):.3f} "
                  f"min={min(all_ratio):.3f}  (retry below {cfg.contrast_retry_ratio})")
        elif tot_att:
            print("  contrast_ratio  no probes -> no evidence/code conflict was flagged")
        if all_dec:
            print(f"  decision_surprise n={len(all_dec)} mean={mean(all_dec):.3f} "
                  f"max={max(all_dec):.3f}  (DIAGNOSTIC, never gates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
