# bert-interpretability-illusion

Replicate, interpret, and extend the paper "An interpretability illusion for BERT".

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

## Documentation

For comprehensive documentation on replicating and extending this research, see **[doc/README.md](doc/README.md)** which provides:

- **Quick start guides** for different audiences (learners, researchers, developers)
- **[PAPER_NOTES.md](doc/PAPER_NOTES.md)** - Detailed technical analysis of the paper
- **[REPLICATION_PLAN.md](doc/REPLICATION_PLAN.md)** - Step-by-step experimental methodology
- **[DESIGN.md](doc/DESIGN.md)** - Technical architecture and implementation details
- **[IDEAS.md](doc/IDEAS.md)** - Research extensions and modern model testing

## Repository Structure

```
bert-interpretability-illusion/
├── doc/                      # Research documentation
│   ├── README.md            # Documentation guide and navigation
│   ├── PAPER_NOTES.md       # Detailed technical summary
│   ├── REPLICATION_PLAN.md  # Step-by-step experimental methodology
│   ├── IDEAS.md             # Research extensions and hypotheses
│   ├── DESIGN.md            # Technical design notes
│   ├── CONSOLIDATION_NOTES.md  # Documentation consolidation record
│   └── 2104.07143v1.pdf     # Original paper
├── streamlit_app.py         # Interactive explainer (current)
├── AGENTS.md                # Development environment requirements
├── CLAUDE.md                # Quick reference for Claude agents
├── pyproject.toml           # Project dependencies (managed via uv)
├── .python-version          # Python version specification
└── uv.lock                  # Locked dependencies
```

## Getting Started

### For Researchers
1. Start with **[doc/README.md](doc/README.md)** for navigation to all documentation
2. Read **[doc/PAPER_NOTES.md](doc/PAPER_NOTES.md)** for comprehensive understanding of the original findings
3. Review **[doc/REPLICATION_PLAN.md](doc/REPLICATION_PLAN.md)** for experimental methodology
4. See **[doc/IDEAS.md](doc/IDEAS.md)** for planned extensions and open questions

### For Developers
1. See **AGENTS.md** for environment setup and development requirements
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
