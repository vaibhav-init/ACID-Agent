"""Quick live check of structured output (function_calling on DeepSeek)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel

from acid_agent.llm import ask, ask_structured  # noqa: E402


class UnitPlan(BaseModel):
    goals: list[str] = []


reply = ask("Reply with exactly one word: OK")
print("plain reply:", reply)

plan: UnitPlan = ask_structured(
    "Break this task into 2 short goals: 'Compute total revenue for region north from orders.csv'",
    UnitPlan,
)
print("structured goals:", plan.goals)
assert plan.goals, "no goals returned"
print("STRUCTURED OUTPUT OK")

