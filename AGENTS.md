# Ashat_AI — Agent Guide

**What this is:** Python 3.9 data-science project skeleton (likely "Food AI"). No source code yet — just AI Factory tooling configured.

## Project state
- No source files, no build system, no tests, no CI.
- `data/` and `models/` dirs are empty placeholders.
- `requirements.txt` and `README.md` exist but are empty.
- `sample.ipynb` is the default PyCharm template — not real code.

## Tooling
- **Virtual env:** `.venv/` — Python 3.9.6, standard data-science stack (numpy, pandas, matplotlib, jupyterlab).
- **AI Factory** fully configured via `.ai-factory.json` for both `cursor` and `opencode` agents (all `aif-*` skills installed).
- **OpenCode:** `.opencode/` — package installs `@opencode-ai/plugin`.
- **Cursor:** `.cursor/skills/` mirrors OpenCode skills.
- No linter, formatter, type checker, or task runner configured.

## Commands
```bash
source .venv/bin/activate   # activate virtual env
jupyter lab                  # start JupyterLab
jupyter notebook             # start classic notebook
```

## Conventions
- Add dependencies to `requirements.txt` as they are introduced.
- Keep project-level code (`.py` modules) at the repo root until a package structure emerges.
- Notebooks go in `data/` or root; `models/` stores serialized artifacts.
