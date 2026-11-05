---
name: teacher
description: "Econometrics skill in Teacher mode: explain or teach a statistics or econometrics method. Type /econometrics:teacher followed by the topic."
disable-model-invocation: true
---

# Teacher mode (command)

The user chose the **Teacher** mode of the `econometrics` skill explicitly. This command holds no method instructions of its own: every number comes from the same tested scripts and rules as the main skill.

1. Load the main skill now: call the Skill tool with `econometrics:econometrics` and follow it, in Teacher mode, without asking which mode to use. If the Skill tool cannot load it, read `../../SKILL.md` (two folders up) and follow it the same way.
2. Read `../../references/teaching.md` (the curriculum map in section 6 names the function and section behind each topic), then route the question as the main skill's section 2 says.
3. Deliver a method card: intuition, model, assumptions, estimator, simulation (`../../scripts/simulation.py`), code, diagnostics, interpretation, pitfalls, and an exercise with its solution.

The user's request: $ARGUMENTS
