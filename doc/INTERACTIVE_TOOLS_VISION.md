# Interactive Research Tools Vision

**Status:** Future work - to be implemented after/alongside replication

**Context:** This document captures ideas for transforming the static explainer into interactive research tools that support active investigation rather than just education.

## The Gap

**Current State:** `streamlit_app.py` is an educational explainer about the paper using simulated/hardcoded data

**Future Vision:** Interactive research workbench for conducting and exploring the replication and extension work

## Proposed Tool Suite

### Phase 1: Replication Support Tools

#### 1. Neuron Explorer (`tools/neuron_explorer.py`)
**Purpose:** Interactively explore any neuron's activations across your actual data

**Features:**
- Select neuron (1-768) from dropdown
- Load YOUR embeddings from replication experiments
- Display top-N activating sentences from each dataset
- Side-by-side comparison across datasets
- Export patterns for annotation

**Key Difference from Current App:** Works with YOUR data, not paper's examples

#### 2. Dataset Comparator (`tools/dataset_comparator.py`)
**Purpose:** Upload and compare your replicated datasets

**Features:**
- Upload QQP, QNLI, Wiki, Books sentence embeddings
- Compute UMAP/t-SNE on YOUR data
- Overlay neuron directions on the embedding space
- Measure dataset separation metrics
- Interactive cluster exploration

**Value:** Validates YOUR replication of dataset idiosyncrasy

#### 3. Activation Viewer (`tools/activation_viewer.py`)
**Purpose:** Real-time sentence activation inspection

**Features:**
- Load any direction (neuron or custom)
- Sort sentences by activation score
- Filter by dataset
- Highlight tokens contributing to activation
- Save interesting examples

#### 4. Pattern Annotation Interface (`tools/annotation_interface.py`)
**Purpose:** Replicate the human annotation study with YOUR data

**Features:**
- Multi-annotator support
- Blind annotation (hide neuron/dataset info)
- Pattern labeling with categories
- Agreement calculation (Cohen's kappa)
- Export annotations for analysis

**Value:** Your own annotation data for methodological critique

### Phase 2: Methodological Critique Tools

#### 5. Corpus Quality Analyzer (`tools/corpus_analyzer.py`)
**Purpose:** Assess dataset quality issues

**Features:**
- Token frequency distributions
- Sentence length distributions
- Lexical diversity metrics
- Topic modeling visualization
- Outlier detection
- Dataset bias identification

**Value:** Supports Phase 2 critique of corpus selection

#### 6. Annotation Agreement Tracker (`tools/annotation_tracker.py`)
**Purpose:** Enhanced annotation study with better methodology

**Features:**
- Real-time inter-annotator agreement
- Pattern taxonomy builder
- Disagreement resolution interface
- Annotator calibration metrics
- Quality control dashboard

#### 7. Locality Score Calculator (`tools/locality_calculator.py`)
**Purpose:** Compute locality scores for YOUR neurons

**Features:**
- Load embeddings and neurons
- Calculate k-NN distances
- Compare meaningful vs meaningless neurons
- Visualize local clusters
- Statistical significance testing

### Phase 3: Model Comparison Tools

#### 8. Multi-Model Dashboard (`tools/model_comparator.py`)
**Purpose:** Compare illusion across BERT variants

**Features:**
- Load multiple models (BERT, RoBERTa, DeBERTa, etc.)
- Extract embeddings from all models
- Test same neuron indices across models
- Compare pattern consistency
- Cross-model illusion metrics

**Value:** Core of Phase 3 research

#### 9. Cross-Architecture Illusion Detector (`tools/illusion_detector.py`)
**Purpose:** Automated illusion detection pipeline

**Features:**
- Run full analysis on any model
- Detect dataset-dependent patterns
- Measure pattern diversity
- Generate illusion reports
- Compare against baseline

#### 10. Embedding Space Comparator (`tools/embedding_comparator.py`)
**Purpose:** Understand geometric differences across models

**Features:**
- UMAP/t-SNE of multiple model embeddings
- Dataset clustering comparison
- Direction similarity analysis
- Concept transfer testing
- Geometric property measurements

## Implementation Strategy

### Priority 1: Enable Replication
Start with tools 1-4 to support exact replication with YOUR data

### Priority 2: Support Critique
Build tools 5-7 for methodological improvements

### Priority 3: Extension Research
Implement tools 8-10 for cross-model research

## Technical Approach

### Architecture
```
tools/
├── common/
│   ├── embedding_loader.py   # Load embeddings from various sources
│   ├── model_wrapper.py      # Unified interface to BERT variants
│   ├── visualization.py      # Shared plotting utilities
│   └── metrics.py            # Common analysis metrics
├── neuron_explorer.py
├── dataset_comparator.py
├── activation_viewer.py
└── ...
```

### Data Flow
1. **Replication experiments** generate embeddings → saved to `data/embeddings/`
2. **Tools** load embeddings via `common/embedding_loader.py`
3. **Interactive exploration** updates understanding
4. **Findings** documented in the tools themselves (literate programming approach)

### Technology Stack
- **Frontend:** Streamlit (consistent with current app) or Marimo (for literate style)
- **Visualization:** Plotly (already used), Altair (for more complex interactions)
- **ML/Data:** PyTorch, transformers, scikit-learn, UMAP
- **Storage:** HDF5 for embeddings, SQLite for annotations

## Integration with Current Explainer

**Proposed Structure:**
```
bert-interpretability-illusion/
├── paper_explainer.py        # Current app (renamed, educational)
├── research_dashboard.py     # New main research interface
└── tools/                    # Individual research tools
    ├── neuron_explorer.py
    ├── dataset_comparator.py
    └── ...
```

**Navigation:**
- `paper_explainer.py`: "Want to do research? See research_dashboard.py"
- `research_dashboard.py`: Links to all tools + "Need context? See paper_explainer.py"

## Why This Matters

**From README:**
> "We conduct experiments through building interactive tools and visualizations that allow real-time exploration of model behavior, dataset geometry, and neuron activations."

**Current State:** Have static explainer
**This Vision:** Actual interactive research tools
**Result:** Research-as-exploration, findings emerge from interaction

## Next Steps

1. **After replication data exists:** Implement Priority 1 tools
2. **During methodological critique:** Build Priority 2 tools
3. **For extension research:** Create Priority 3 tools

Each tool becomes both:
- A research instrument for discovery
- A communication artifact showing findings

This is the "literate interpretability research" approach we discussed.
