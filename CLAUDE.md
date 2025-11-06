# Claude Code Development Guide

## Environment Setup

For complete environment setup and mandatory requirements, see **AGENTS.md**.

### Quick Start

```bash
# Initialize/sync the virtual environment
uv sync

# Run the Streamlit app
uv run streamlit run streamlit_app.py
```

### Quick Reference

- **All Python commands**: Use `uv run` prefix (e.g., `uv run python script.py`)
- **Add packages**: Use `uv add package-name` (never use `pip install`)
- **Dependency file**: `pyproject.toml` (not `requirements.txt`)

### Claude-Specific Notes

When working on this project:
- Always refer to AGENTS.md for the authoritative environment and dependency rules
- Use the todo list (TodoWrite) to track multi-step tasks
- Remember to use `uv run` for ALL Python invocations
- Commit `uv.lock` when dependencies change
