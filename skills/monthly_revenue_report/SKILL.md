# Skill: monthly_revenue_report

Validated workflow: compute per-month revenue totals from an orders CSV.

## Usage

    python skills/monthly_revenue_report/skill.py PATH_TO_CSV --date-col date --value-col revenue

## Guarantees

- Normalizes mixed date formats (`-` and `/` separators) before grouping.
- Drops rows with missing dates/values and reports how many were dropped.
- Validated by test_skill.py before being trusted by the router.
