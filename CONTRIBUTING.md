# Contributing

This is a personal project, so contributions are not expected. If you spot a bug or have a suggestion, feel free to open an issue.

## Running locally

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check .
```

## Code style

- ruff for linting and formatting (`ruff check .` and `ruff format .`)
- Python 3.11+, type hints on all new functions
- No em dashes in comments or docs

## Pull requests

Keep PRs focused on one thing. Include tests for new behavior.
