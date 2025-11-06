# bert-interpretability-illusion

Active research project investigating the "interpretability illusion" phenomenon in BERT and its architectural variants.

## Research Mission

This repository contains an ongoing investigation into whether the interpretability illusion discovered in BERT—where individual neurons appear to encode meaningful concepts within one dataset but show completely different patterns in another—is fundamental to how these models learn or an artifact of specific design choices.

## The Core Finding

The original paper (Bolukbasi et al., 2021) revealed that BERT neurons which appear to cleanly encode concepts like "song titles" or "historical dates" when analyzed on one dataset will encode entirely unrelated patterns when examined on different text corpora. This suggests that what looks like meaningful feature detection may actually be dataset-specific memorization—an "interpretability illusion."

## Research Approach

We conduct experiments through building interactive tools and visualizations that allow real-time exploration of model behavior, dataset geometry, and neuron activations. This approach enables both rigorous investigation and immediate insight into patterns as they emerge.

## Research Phases

### Phase 1: Exact Replication
**Status**: Planning/In Progress
Reproduce the original paper's experiments with identical methodology to establish baseline findings.

### Phase 2: Methodological Critique and Improvements
**Status**: Documented (see doc/IDEAS.md)
- Evaluate corpus selection and quality issues
- Improve annotation/labeling methodology (current approach has significant weaknesses)
- Apply modern interpretability techniques developed since 2021 (sparse autoencoders, causal interventions, mechanistic interpretability)

### Phase 3: BERT Architectural Variants
**Status**: Planning
Test whether the illusion persists across BERT-family models: RoBERTa, DeBERTa, ALBERT, DistilBERT, etc.

## Repository Structure

```
bert-interpretability-illusion/
├── doc/                      # Research documentation
│   ├── PAPER_NOTES.md       # Detailed technical summary of original paper
│   ├── REPLICATION_PLAN.md  # Step-by-step experimental methodology
│   ├── IDEAS.md             # Research extensions and hypotheses
│   └── DESIGN.md            # Technical design notes
├── streamlit_app.py         # Interactive explainer (current)
├── AGENTS.md                # Development environment requirements
├── CLAUDE.md                # Quick reference for Claude agents
└── pyproject.toml           # Project dependencies (managed via uv)
```

## Getting Started

### For Researchers
1. Start with `doc/PAPER_NOTES.md` for comprehensive understanding of the original findings
2. Review `doc/REPLICATION_PLAN.md` for experimental methodology
3. See `doc/IDEAS.md` for planned extensions and open questions

### For Developers
1. See `AGENTS.md` for environment setup and development requirements
2. Use `uv sync` to initialize the Python environment
3. All Python code must be run via `uv run` (e.g., `uv run streamlit run streamlit_app.py`)

### To Explore Current Tools
```bash
# Set up environment
uv sync

# Run the interactive explainer
uv run streamlit run streamlit_app.py
```

## Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python dependency management. See **AGENTS.md** for complete environment setup and mandatory development practices.

## Paper Reference

```bibtex
@article{bolukbasi2021interpretability,
  title={An Interpretability Illusion for BERT},
  author={Bolukbasi, Tolga and Pearce, Adam and Yuan, Ann and Coenen, Andy and Reif, Emily and Viégas, Fernanda and Wattenberg, Martin},
  journal={arXiv preprint arXiv:2104.07143},
  year={2021}
}
```