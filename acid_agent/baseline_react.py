"""DA-Agent style ReAct baseline, driven by the backbone CLI.

This is the analogue of the reference implementation's `prompt` agent
(`vendor/acid-paper-ref/da_agent/agent/agents.py:PromptAgent`): a plain
Thought/Action/Observation loop over a Bash/Python/SQL/Terminate action space,
with NO transactional machinery — no exploration phase, no decision extraction,
no validation gate, no rollback.

Why this exists alongside `baseline_harness.py`:

    baseline_harness -> the agent *harness* (agentic, file access, its own
                        planning and self-correction). Strong, but the harness
                        contributes capability the paper's baseline never had.
    baseline_react   -> the *model* with no harness. Every step is a stateless
                        completion; this module owns the loop, the action
                        parsing and the execution.

Both run on the CLI's subscription auth; there is no API key here.

Isolation is load-bearing and was NOT free. A plain agent session keeps
file-reading tools, and the first live run of this module answered correctly in
one step without ever emitting an action — the model had simply gone and read
the CSV itself. What holds is `llm.ask_isolated`: the read-only `plan` agent
with the working directory set to a fresh empty temp dir. The model therefore
sees only what this loop puts in the prompt, and reaches the workspace only
through actions this module executes on its behalf.

For the same reason the prompt advertises the work dir as `.` rather than the
real absolute path: handing over `workspaces/<slug>/` would just tell the model
where to look.

The reference runs its actions inside a Docker sandbox; this runs them in the
git-backed Workspace, so a baseline run is still cleanly isolated per task.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from dataclasses import dataclass

from .config import get_settings
from .llm import ask_isolated as ask
from .tracing import traced

MAX_OBS_CHARS = 4000
PARSE_RETRY_LIMIT = 3


# ------------------------------------------------------------------ actions

@dataclass
class Action:
    pass


@dataclass
class Bash(Action):
    code: str

    def __repr__(self) -> str:
        return f'Bash(code="{self.code}")'


@dataclass
class Python(Action):
    code: str
    filepath: str | None = None

    def __repr__(self) -> str:
        return f'Python(file_path="{self.filepath}"):\n```python\n{self.code}\n```'


@dataclass
class SQL(Action):
    code: str
    file_path: str | None = None
    output: str | None = None

    def __repr__(self) -> str:
        return f'SQL(file_path="{self.file_path}", command="{self.code}", output="{self.output}")'


@dataclass
class Terminate(Action):
    output: str | None = None

    def __repr__(self) -> str:
        return f'Terminate(output="{self.output}")'


def _unquote(text: str) -> str:
    text = text.strip()
    for q in ('"""', "'''", '"', "'"):
        if len(text) >= 2 * len(q) and text.startswith(q) and text.endswith(q):
            return text[len(q) : -len(q)]
    return text


def _balanced_arg(text: str, start: int) -> str | None:
    """Read to the paren that closes the one just opened, ignoring quoted parens."""
    depth, i, quote = 1, start, None
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


def _split_args(content: str) -> list[str]:
    parts, buf, quote, depth = [], [], None, 0
    for ch in content:
        if quote:
            if ch == quote:
                quote = None
            buf.append(ch)
            continue
        if ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


def parse_action(output: str) -> Action | None:
    """Pull one action out of a Thought/Action response.

    Mirrors the reference's cascade: isolate the Action segment first, then try
    each action class in order, then retry once on lightly mangled text.
    """
    if not output or not output.strip():
        return None

    action_string = ""
    for p in (
        r'["\']?Action["\']?:? (.*?)Observation',
        r'["\']?Action["\']?:? (.*?)Thought',
        r'["\']?Action["\']?:? (.*?)$',
        r'^(.*?)Observation',
    ):
        m = re.search(p, output, flags=re.DOTALL)
        if m:
            action_string = m.group(1).strip()
            break
    if not action_string:
        action_string = output.strip()

    for candidate in (action_string, action_string.replace("\\_", "_").replace("'''", "```")):
        for parser in (_parse_python, _parse_sql, _parse_bash, _parse_terminate):
            action = parser(candidate)
            if action is not None:
                return action
    return None


def _parse_python(text: str) -> Python | None:
    # Python() is tried before Bash() because its body is a fenced block that a
    # looser matcher would happily swallow as a shell string.
    for p in (
        r'Python\(file_?path=(.*?)\).*?```python[ \t]*\r?\n(.*?)```',
        r'Python\(file_?path=(.*?)\).*?```[ \t]*\r?\n(.*?)```',
    ):
        matches = re.findall(p, text, flags=re.DOTALL)
        if matches:
            filepath, code = matches[-1][0], matches[-1][1]
            return Python(code=code.strip(), filepath=_unquote(filepath))
    return None


def _parse_sql(text: str) -> SQL | None:
    m = re.search(r"SQL\(", text)
    if not m:
        return None
    content = _balanced_arg(text, m.end())
    if content is None:
        return None
    parts = _split_args(content)
    if len(parts) < 3:
        return None
    file_path, command, output = parts[0].strip(), ",".join(parts[1:-1]).strip(), parts[-1].strip()
    for prefix in ("file_path=", "filepath="):
        file_path = file_path.removeprefix(prefix)
    for prefix in ("command=", "code="):
        command = command.removeprefix(prefix)
    output = output.removeprefix("output=")
    return SQL(file_path=_unquote(file_path), code=_unquote(command), output=_unquote(output))


def _parse_bash(text: str) -> Bash | None:
    m = re.search(r"Bash\(code=", text)
    if not m:
        return None
    code = _balanced_arg(text, m.end())
    return Bash(code=_unquote(code)) if code is not None else None


def _parse_terminate(text: str) -> Terminate | None:
    m = re.search(r"Terminate\(output=", text)
    if not m:
        return None
    out = _balanced_arg(text, m.end())
    return Terminate(output=_unquote(out)) if out is not None else None


ACTION_SPACE = """## Bash Action
* Signature: Bash(code="shell_command")
* Description: Executes a valid non-interactive shell command.
* Example: Bash(code="ls -l")

## Python Action
* Signature: Python(file_path="path/to/python_file"):
```python
executable_python_code
```
* Description: Creates a python file with the fenced content (overwriting if it exists) and then executes it.
* Example: Python(file_path="./hello.py"):
```python
print("Hello, world!")
```

## SQL Action
* Signature: SQL(file_path="path/to/database_file", command="sql_command", output="path/to/output.csv" or "direct")
* Description: Executes an SQL command against a SQLite-compatible database file. With output="direct" the rows are printed.
* Example: SQL(file_path="data.db", command="SELECT * FROM users", output="direct")

## Terminate Action
* Signature: Terminate(output="your_answer")
* Description: Ends the task. Put your FINAL ANSWER in the output field, or "FAIL" if it cannot be completed.
* Example: Terminate(output="The average monthly revenue is 5.28")"""


SYSTEM_PROMPT = """# CONTEXT #
You are a data scientist proficient in analyzing data. You excel at using Bash commands and Python code to solve data-related problems. You are working in a Bash environment with all necessary Python libraries installed. You are starting in the {work_dir} directory, which contains all the data needed for your tasks. You can only use the actions provided in the ACTION SPACE to solve the task. The maximum number of steps you can take is {max_steps}.

# ACTION SPACE #
{action_space}

# IMPORTANT: YOU HAVE NO DIRECT FILE ACCESS #
Any file tools you appear to have (Read, Glob, Grep, ...) point at an empty
scratch directory and CANNOT see the data. Ignore them completely — an empty
listing from them means nothing. The ONLY way to observe or affect the data is
to emit an Action; the environment executes it in the data directory and returns
the result as the next Observation. Never conclude data is missing because your
own tools found nothing; run a Bash or Python Action and read the Observation.

# NOTICE #
1. You need to fully understand the action space and its arguments before using it.
2. You should first understand the environment and conduct data analysis on the given data before handling the task.
3. You can't take some problems for granted. For example, you should check the existence of files before reading them (with an Action).
4. If the function execution fails, you should analyze the error and try to solve it.
5. Never answer from memory or assumption. Every number you report must come from an Observation produced by an Action you ran.
6. Before finishing the task, ensure all instructions are met and verify the existence and correctness of any generated files.
7. Each step should only CREATE new files. Never modify or delete existing files. If you need to change data, write the result to a new file with a different name.

# RESPONSE FORMAT #
For each task input, your response must contain:
1. One analysis of the task and the current environment, reasoning to determine the next action (prefix "Thought: ").
2. One action string from the ACTION SPACE (prefix "Action: ").

# EXAMPLE INTERACTION #
Observation: ...(the output of the last action, provided by the environment; you do not generate it)

Thought: ...
Action: ...

# TASK #
{task}"""


# ---------------------------------------------------------------- execution

def _run_bash(ws, code: str, timeout_s: int) -> str:
    try:
        proc = subprocess.run(
            code,
            shell=True,
            cwd=ws.root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {timeout_s}s"
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    if proc.returncode != 0:
        out = f"(exit code {proc.returncode})\n{out}"
    return out.strip() or "(command produced no output)"


def _run_python(ws, action: Python, timeout_s: int) -> str:
    name = (action.filepath or "step.py").lstrip("./")
    path = ws.root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(action.code, encoding="utf-8")
    r = ws.run_script(name, timeout_s=timeout_s)
    out = r.stdout + (("\n" + r.stderr) if r.stderr else "")
    if not r.ok:
        out = f"(exit code {r.returncode})\n{out}"
    return out.strip() or "Python script executed successfully. No output."


def _run_sql(ws, action: SQL, timeout_s: int) -> str:
    # Routed through the Python action's executor rather than a sqlite3 binary,
    # so SQL needs no extra dependency and lands in the same workspace.
    direct = (action.output or "direct").lower().startswith("direct")
    dest = "" if direct else f"df.to_csv({action.output!r}, index=False)\nprint('wrote {action.output}')"
    code = (
        "import sqlite3, pandas as pd\n"
        f"con = sqlite3.connect({action.file_path!r})\n"
        f"df = pd.read_sql_query({action.code!r}, con)\n"
        + ("print(df.to_string())\n" if direct else dest + "\n")
    )
    return _run_python(ws, Python(code=code, filepath="_sql_step.py"), timeout_s)


def _execute(ws, action: Action, timeout_s: int) -> tuple[str, bool]:
    if isinstance(action, Terminate):
        return (action.output or ""), True
    if isinstance(action, Bash):
        return _run_bash(ws, action.code, timeout_s), False
    if isinstance(action, Python):
        return _run_python(ws, action, timeout_s), False
    if isinstance(action, SQL):
        return _run_sql(ws, action, timeout_s), False
    return f"Unsupported action: {action!r}", False


# --------------------------------------------------------------------- loop

@traced("baseline_react")
def run_baseline_react(task: str, ws, tracer=None, run_id=None) -> str:
    """Thought/Action/Observation loop until Terminate or the step budget runs out."""
    s = get_settings()
    system = SYSTEM_PROMPT.format(
        # Deliberately not the real path -- see the module docstring.
        work_dir=".",
        max_steps=s.react_max_steps,
        action_space=ACTION_SPACE,
        task=task,
    )

    history: list[str] = []
    # Seed the first observation with a real listing. The model's own file tools
    # point at an empty sandbox, so without this it "checks" the directory, sees
    # nothing, and terminates with FAIL before ever emitting an action.
    obs = "You are in the folder now. Its contents:\n" + _run_bash(
        ws, "ls -la", s.react_action_timeout_s
    )
    answer = ""
    last_action_repr: str | None = None
    repeated_once = False
    parse_failures = 0

    for step in range(s.react_max_steps):
        # The backbone call is stateless, so the transcript is rebuilt into the prompt
        # each turn. Trimmed to the last N exchanges, like the reference's
        # max_memory_length, so a long run cannot outgrow the context.
        window = history[-(s.react_max_memory * 2):]
        prompt = system + "\n\n" + "\n\n".join(window + [f"Observation: {obs}", "Thought:"])

        try:
            response = ask(prompt)
        except Exception as e:
            if tracer:
                tracer.log("react_llm_failed", step=step, error=str(e)[:300])
            break

        action = parse_action(response)
        if action is None:
            parse_failures += 1
            if parse_failures > PARSE_RETRY_LIMIT:
                if tracer:
                    tracer.log("react_parse_give_up", step=step)
                break
            obs = "Failed to parse an action from your response. Reply with 'Action: <one action from the ACTION SPACE>'."
            continue
        parse_failures = 0

        # A model that repeats an identical action is stuck; the reference nudges
        # once and aborts on the second repeat rather than burning the budget.
        if last_action_repr is not None and repr(action) == last_action_repr:
            if repeated_once:
                if tracer:
                    tracer.log("react_repeated_action", step=step, action=repr(action)[:200])
                break
            obs = "The action is the same as the last one. Provide a different action."
            repeated_once = True
            continue
        repeated_once = False
        last_action_repr = repr(action)

        thought = ""
        tm = re.search(r"Thought:\s*(.*?)(?:\n\s*Action:|$)", response, flags=re.DOTALL)
        if tm:
            thought = tm.group(1).strip()

        result, done = _execute(ws, action, s.react_action_timeout_s)
        if tracer:
            tracer.log(
                "react_step",
                run_id=str(run_id) if run_id else None,
                step=step,
                action_type=type(action).__name__,
                action=repr(action)[:500],
                observation=result[:1500],
                done=done,
            )
        if done:
            answer = result
            break

        history.append(f"Observation: {obs}")
        history.append(f"Thought: {thought}\n\nAction: {action!r}")
        obs = result[:MAX_OBS_CHARS]

    if not answer:
        # Budget exhausted without Terminate: ask once for the answer from the
        # transcript, so the run is scored on its work rather than on an empty
        # string. This mirrors what the grader would otherwise see as a 0.
        try:
            answer = ask(
                f"Task: {task}\n\nWork log:\n"
                + "\n\n".join(history[-(s.react_max_memory * 2):])
                + f"\n\nLast observation: {obs}\n\n"
                "State the FINAL ANSWER to the task in one short line. If it cannot be "
                "determined from the work log, reply exactly: FAIL"
            ).strip()
        except Exception as e:
            if tracer:
                tracer.log("react_final_answer_failed", error=str(e)[:300])

    if tracer:
        tracer.log("react_done", run_id=str(run_id) if run_id else None, answer=answer[:1000])
    return answer
