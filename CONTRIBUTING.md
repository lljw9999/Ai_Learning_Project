# Contributing

## Development Setup

1. Create virtual environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run tests:

```bash
python -m pytest
```

## Branching

- Use short descriptive branch names.
- Keep each PR focused on one concern (modeling, service, infra, docs, etc.).

## Pull Requests

- Fill the PR template completely.
- Include exact commands used for validation.
- If metrics change, include before/after values and dataset split details.
- Update README when behavior, defaults, or commands change.

## Quality Gates

- CI must pass.
- No obvious data leakage in train/eval paths.
- Backward compatibility for API endpoint shape should be preserved unless explicitly intentional.
