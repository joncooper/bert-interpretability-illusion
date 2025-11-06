# Development Instructions for AI Agents

## Python Environment Setup

This project uses **[uv](https://docs.astral.sh/uv/)** for fast Python dependency management and virtual environment handling.

### Environment Details
- **Tool**: uv (https://docs.astral.sh/uv/)
- **Python Version**: 3.12+ (specified in `.python-version`)
- **Dependency File**: `pyproject.toml` (managed by uv)
- **Lock File**: `uv.lock` (for reproducible builds)

### Initial Setup

```bash
# Initialize/sync the virtual environment
uv sync

# Run the Streamlit app
uv run streamlit run streamlit_app.py
```

### Mandatory Requirements for All Agents

⚠️ **These rules MUST be followed without exception:**

1. **ALL Python invocations MUST use `uv run`**
   - ✅ `uv run python script.py`
   - ✅ `uv run streamlit run streamlit_app.py`
   - ✅ `uv run jupyter notebook`
   - ❌ Never use `python` directly
   - ❌ Never use `streamlit` without `uv run`
   - ❌ Never use `pip install`

2. **ALL package additions MUST use `uv add`**
   - ✅ `uv add numpy scipy`
   - ✅ `uv add --dev pytest black`
   - ❌ Never use `pip install`
   - ❌ Never modify `requirements.txt` directly
   - ❌ Never manually edit `pyproject.toml` dependency versions

3. **DO NOT touch `requirements.txt`**
   - It is legacy and retained only for reference
   - All dependency management goes through `pyproject.toml` and `uv add`

4. **Always commit `uv.lock` after dependency changes**
   - Running `uv add` will update `uv.lock`
   - Include lock file changes in commits for reproducibility

### Project Context

- **streamlit_app.py** - Interactive explainer for BERT interpretability illusion paper
- **PAPER_NOTES.md** - Detailed technical summary of the research
- **REPLICATION_PLAN.md** - Step-by-step implementation guide
- **IDEAS.md** - Extension and research ideas
