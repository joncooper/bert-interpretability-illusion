# Technical Design Document: Replicating "An Interpretability Illusion for BERT"

**Version**: 1.0  
**Date**: November 2025  
**Status**: Design Phase

## Executive Summary

This document provides a comprehensive technical design for replicating the experiments from "An Interpretability Illusion for BERT" (Bolukbasi et al., 2021). It covers:
- Software architecture and codebase structure
- Required libraries and dependencies
- Detailed dataset acquisition and preparation
- Implementation strategy for each experiment
- Testing and validation approach
- Timeline and resource requirements

## 1. Technology Stack

### 1.1 Core Libraries

#### Python Environment
```
Python: 3.9+  (3.10 recommended for stability)
```

#### Deep Learning & NLP
```yaml
torch: ^2.0.0                    # PyTorch for model operations
transformers: ^4.35.0            # HuggingFace models and tokenizers
datasets: ^2.14.0                # HuggingFace datasets library
tokenizers: ^0.14.0              # Fast tokenization
```

**Rationale**: 
- Latest stable transformers ensures compatibility with modern hardware
- Datasets library provides unified interface to many corpora
- Torch 2.0+ offers performance improvements via compile

#### Scientific Computing
```yaml
numpy: ^1.24.0                   # Numerical operations
scipy: ^1.11.0                   # Statistical tests, distance metrics
pandas: ^2.0.0                   # Data manipulation and analysis
scikit-learn: ^1.3.0             # SVM classifier, metrics
```

#### Visualization
```yaml
matplotlib: ^3.7.0               # Basic plotting
seaborn: ^0.12.0                 # Statistical visualizations
umap-learn: ^0.5.4               # UMAP dimensionality reduction
plotly: ^5.17.0                  # Interactive visualizations (optional)
```

#### Utilities
```yaml
tqdm: ^4.66.0                    # Progress bars
pyyaml: ^6.0                     # Configuration files
python-dotenv: ^1.0.0            # Environment variables
click: ^8.1.0                    # CLI interface
loguru: ^0.7.0                   # Logging
```

#### Development & Testing
```yaml
pytest: ^7.4.0                   # Testing framework
pytest-cov: ^4.1.0               # Coverage reporting
black: ^23.9.0                   # Code formatting
ruff: ^0.0.292                   # Linting
mypy: ^1.5.0                     # Type checking
jupyter: ^1.0.0                  # Notebooks for exploration
```

### 1.2 Hardware Requirements

**Minimum**:
- GPU: 8GB VRAM (RTX 3060, T4, V100)
- RAM: 16GB system memory
- Storage: 50GB free space
- CPU: 4+ cores

**Recommended**:
- GPU: 16GB+ VRAM (RTX 4090, A10, A100)
- RAM: 32GB+ system memory
- Storage: 100GB+ SSD
- CPU: 8+ cores

**Notes**:
- BERT-base fits in 8GB but batch processing benefits from more
- Dataset storage and embeddings require substantial disk space
- CPU cores important for data preprocessing parallelization

### 1.3 Development Environment

**Recommended Setup**:
```bash
# Using conda for environment management
conda create -n bert-illusion python=3.10
conda activate bert-illusion

# Or using venv
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

**IDE Recommendations**:
- VSCode with Python, Jupyter extensions
- PyCharm Professional (for debugging)
- JupyterLab for exploratory analysis

## 2. Codebase Architecture

### 2.1 Project Structure

```
bert-interpretability-illusion/
├── README.md
├── pyproject.toml              # Project dependencies (Poetry)
├── setup.py                    # Alternative: setuptools
├── requirements.txt            # Pinned dependencies
├── .env.example                # Environment variables template
├── .gitignore
│
├── config/                     # Configuration files
│   ├── datasets.yaml          # Dataset specifications
│   ├── models.yaml            # Model configurations
│   ├── experiments.yaml       # Experiment parameters
│   └── paths.yaml             # Data/output paths
│
├── data/                      # Data directory (gitignored)
│   ├── raw/                   # Original downloaded datasets
│   ├── processed/             # Cleaned and formatted data
│   ├── embeddings/            # Cached sentence embeddings
│   └── cache/                 # HuggingFace cache
│
├── src/                       # Source code
│   ├── __init__.py
│   │
│   ├── data/                  # Data acquisition and processing
│   │   ├── __init__.py
│   │   ├── loaders.py         # Dataset loading functions
│   │   ├── preprocessors.py   # Text preprocessing
│   │   └── samplers.py        # Dataset sampling strategies
│   │
│   ├── models/                # Model wrappers
│   │   ├── __init__.py
│   │   ├── bert_wrapper.py    # BERT model interface
│   │   ├── embedding.py       # Embedding extraction
│   │   └── registry.py        # Model registry pattern
│   │
│   ├── analysis/              # Core analysis code
│   │   ├── __init__.py
│   │   ├── neurons.py         # Neuron activation analysis
│   │   ├── directions.py      # Random direction generation
│   │   ├── locality.py        # Locality score computation
│   │   ├── monotonicity.py    # Token frequency analysis
│   │   └── clustering.py      # Dataset separation metrics
│   │
│   ├── annotation/            # Annotation tools
│   │   ├── __init__.py
│   │   ├── interface.py       # Annotation UI (Streamlit/Gradio)
│   │   ├── storage.py         # Annotation database
│   │   └── agreement.py       # Inter-annotator agreement
│   │
│   ├── visualization/         # Plotting and visualization
│   │   ├── __init__.py
│   │   ├── embeddings.py      # UMAP plots
│   │   ├── distributions.py   # Histograms, box plots
│   │   ├── heatmaps.py        # Confusion matrices
│   │   └── interactive.py     # Interactive dashboards
│   │
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── metrics.py         # Distance metrics, correlations
│       ├── io.py              # File I/O helpers
│       └── logging.py         # Logging configuration
│
├── scripts/                   # Executable scripts
│   ├── 01_download_datasets.py
│   ├── 02_extract_embeddings.py
│   ├── 03_run_neuron_analysis.py
│   ├── 04_compute_locality.py
│   ├── 05_train_classifier.py
│   ├── 06_generate_visualizations.py
│   └── 07_run_full_pipeline.py
│
├── notebooks/                 # Jupyter notebooks
│   ├── 00_environment_test.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   ├── 03_neuron_investigation.ipynb
│   ├── 04_pattern_annotation.ipynb
│   └── 05_results_analysis.ipynb
│
├── tests/                     # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures
│   ├── test_data/             # Test data samples
│   ├── test_loaders.py
│   ├── test_embeddings.py
│   ├── test_analysis.py
│   └── test_metrics.py
│
├── docs/                      # Documentation
│   ├── PAPER_NOTES.md         # Paper summary
│   ├── REPLICATION_PLAN.md    # Step-by-step plan
│   ├── IDEAS.md               # Extension ideas
│   ├── DESIGN.md              # This document
│   ├── API.md                 # Code API documentation
│   └── DATASETS.md            # Dataset documentation
│
└── results/                   # Experimental outputs
    ├── embeddings/            # Saved embeddings
    ├── annotations/           # Annotation results
    ├── figures/               # Generated plots
    ├── tables/                # Result tables (CSV/LaTeX)
    └── checkpoints/           # Experiment checkpoints
```

### 2.2 Key Design Patterns

#### Configuration Management
```python
# config/experiments.yaml
experiments:
  neuron_analysis:
    num_neurons: 25
    num_random_directions: 33
    num_random_sets: 29
    top_k: 10
    random_seed: 42

  locality_analysis:
    k_neighbors: 10
    num_bins: 50
```

**Load with**:
```python
import yaml
from pathlib import Path

def load_config(config_name: str) -> dict:
    config_path = Path("config") / f"{config_name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
```

#### Model Registry
```python
# src/models/registry.py
class ModelRegistry:
    _models = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get(cls, name: str, **kwargs):
        return cls._models[name](**kwargs)

# Usage
@ModelRegistry.register("bert-base")
class BERTBase:
    def __init__(self):
        self.model = AutoModel.from_pretrained("bert-base-uncased")
```

#### Data Pipeline
```python
# src/data/pipeline.py
class DataPipeline:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.steps = []
    
    def add_step(self, func, **kwargs):
        self.steps.append((func, kwargs))
        return self
    
    def execute(self):
        data = self.load()
        for func, kwargs in self.steps:
            data = func(data, **kwargs)
        return data
```

## 3. Dataset Acquisition and Preparation

### 3.1 Dataset Overview

The paper uses four datasets with specific characteristics:

| Dataset | Size | Source | Access Method |
|---------|------|--------|---------------|
| QQP | 164,246 | Quora | HuggingFace / GLUE |
| QNLI | 58,645 | Wikipedia | HuggingFace / GLUE |
| Wiki | 203,736 | Wikipedia | Manual sampling |
| Books | 198,085 | BookCorpus | Problematic - see below |

### 3.2 Detailed Dataset Acquisition

#### 3.2.1 Quora Question Pairs (QQP)

**Official Source**: Part of the GLUE benchmark

**HuggingFace Access**:
```python
from datasets import load_dataset

# Load QQP from GLUE
qqp = load_dataset("glue", "qqp")

# Extract sentences
# QQP has 'question1' and 'question2' fields
# Paper likely used one or both
questions = []
for split in ['train', 'validation']:
    for example in qqp[split]:
        questions.append(example['question1'])
        questions.append(example['question2'])

# Remove duplicates and sample to 164,246
questions = list(set(questions))
# May need to sample if more than 164,246
```

**Verification**:
- Check if total unique questions ≈ 164,246
- If not exact, document the difference

**Alternative**: Direct download from Quora (deprecated)
```python
# Historical reference - may not work
# https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs
```

#### 3.2.2 Question-Answering NLI (QNLI)

**Official Source**: Also part of GLUE, derived from SQuAD

**HuggingFace Access**:
```python
qnli = load_dataset("glue", "qnli")

# QNLI has 'question' and 'sentence' fields
sentences = []
for split in ['train', 'validation']:
    for example in qnli[split]:
        # Paper used the 'sentence' field (Wikipedia passages)
        sentences.append(example['sentence'])

# Should get ~58,645 from validation/test sets
```

**Notes**:
- QNLI sentences are Wikipedia passages
- Validation set is the most likely source
- Check split sizes to match 58,645

#### 3.2.3 Wikipedia (Wiki)

**Challenge**: Paper says "random subset of English Wikipedia as prepared in (Devlin et al., 2018)"

**Original BERT Pretraining Data**:
- The original BERT paper used BookCorpus + Wikipedia
- Wikipedia dump was from a specific date (likely 2018)
- Preprocessing involved specific cleaning steps

**Acquisition Strategy**:

**Option A: Use the exact preprocessing (if available)**
```python
# Check if HuggingFace has the exact dataset
wiki = load_dataset("wikipedia", "20220301.en")  # Try different dates

# Or try the BookCorpusWiki dataset
bookcorpuswiki = load_dataset("bookcorpuswiki")
```

**Option B: Sample from modern Wikipedia**
```python
# Use current Wikipedia dump
wiki = load_dataset("wikipedia", "20200501.en", split="train")

# Sample 203,736 sentences
import random
random.seed(42)

sentences = []
for article in wiki:
    # Split text into sentences
    text = article['text']
    sents = sent_tokenize(text)  # Using NLTK or spaCy
    sentences.extend(sents)

# Random sample
sampled = random.sample(sentences, 203736)
```

**Option C: Reconstruct from original BERT preprocessing**

Reference the BERT repository:
```bash
# https://github.com/google-research/bert
# Look for create_pretraining_data.py

# Key steps:
# 1. Download Wikipedia dump
# 2. Extract text with wikiextractor
# 3. Split into sentences
# 4. Apply BERT's specific cleaning
```

**Recommended**: Try Option A first, fall back to Option B with clear documentation

#### 3.2.4 Toronto BookCorpus (Books)

**CRITICAL CHALLENGE**: BookCorpus is NO LONGER PUBLICLY AVAILABLE

**Background**:
- Original source: Zhu et al. (2015) - 11,038 books from smashwords.com
- Used in training BERT, GPT, and many models
- Removed due to copyright concerns
- No official mirror exists

**Acquisition Strategies**:

**Strategy 1: Use HuggingFace Cached Version (Deprecated)**
```python
# This will likely FAIL
try:
    books = load_dataset("bookcorpus")
except:
    print("BookCorpus not available")
```

**Strategy 2: Use The Pile's Books3 subset**
```python
# The Pile contains a books subset
# However, it's different from original BookCorpus
pile_books = load_dataset("the_pile", "pile-of-law", split="train")
# Filter for books component
```

**Strategy 3: Approximate with Similar Data**

Alternatives that might serve as substitutes:
1. **Project Gutenberg**: Public domain books
   ```python
   gutenberg = load_dataset("sedthh/gutenberg_english")
   ```

2. **PG-19**: Long-form books dataset
   ```python
   pg19 = load_dataset("pg19")
   ```

3. **Books1/Books2** from The Pile
   ```python
   # If The Pile is accessible
   books = load_dataset("EleutherAI/pile-deduped", "books1")
   ```

4. **OpenWebText / C4**: As fallback, not books but large text
   ```python
   c4 = load_dataset("c4", "en", streaming=True)
   ```

**Strategy 4: Reconstruct from Papers Citing BookCorpus**

Search for papers that:
- Used BookCorpus after 2021
- Might have cached/shared it
- Check supplementary materials

**RECOMMENDED APPROACH**:

1. **Document the limitation** clearly in results
2. **Use PG-19 as primary substitute** (most similar in nature)
3. **Clearly label** as "Books (PG-19 substitute)" in all outputs
4. **Run sensitivity analysis**: Do results change if we use different book datasets?

**Implementation**:
```python
# src/data/loaders.py

def load_books_substitute(substitute="pg19", num_sentences=198085, seed=42):
    """
    Load a substitute for BookCorpus.
    
    Args:
        substitute: Which dataset to use ("pg19", "gutenberg", etc.)
        num_sentences: Target number of sentences
        seed: Random seed
    
    Returns:
        List of sentences
    """
    if substitute == "pg19":
        dataset = load_dataset("pg19", split="train")
        sentences = []
        random.seed(seed)
        
        for book in dataset:
            sents = sent_tokenize(book['text'])
            sentences.extend(sents)
            if len(sentences) >= num_sentences:
                break
        
        return random.sample(sentences, num_sentences)
    
    elif substitute == "gutenberg":
        # Similar logic for Gutenberg
        pass
    
    else:
        raise ValueError(f"Unknown substitute: {substitute}")

# Usage with clear documentation
sentences_books = load_books_substitute(substitute="pg19")

# Save metadata
metadata = {
    "original_dataset": "BookCorpus (Zhu et al., 2015)",
    "substitute_used": "PG-19",
    "reason": "Original BookCorpus no longer publicly available",
    "num_sentences": len(sentences_books),
    "random_seed": 42,
    "date_created": "2025-11-06"
}
```

### 3.3 Dataset Preprocessing

**Unified Preprocessing Pipeline**:

```python
# src/data/preprocessors.py

class TextPreprocessor:
    """
    Standardized preprocessing matching BERT's approach.
    """
    
    def __init__(self, lowercase=True, remove_urls=True, remove_special_chars=False):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_special_chars = remove_special_chars
    
    def clean(self, text: str) -> str:
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters (optional)
        if self.remove_special_chars:
            text = re.sub(r'[^a-zA-Z0-9\s\.,!?\'"-]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def filter_sentence(self, text: str, min_length=10, max_length=512) -> bool:
        """Check if sentence should be kept."""
        words = text.split()
        return min_length <= len(words) <= max_length
    
    def process_dataset(self, sentences: List[str]) -> List[str]:
        """Process entire dataset."""
        processed = []
        for sent in tqdm(sentences, desc="Preprocessing"):
            clean_sent = self.clean(sent)
            if self.filter_sentence(clean_sent):
                processed.append(clean_sent)
        return processed
```

### 3.4 Dataset Storage Format

**Save processed datasets**:

```python
# src/data/storage.py

import json
import pickle
from pathlib import Path

class DatasetStorage:
    def __init__(self, base_path="data/processed"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_dataset(self, name: str, sentences: List[str], metadata: dict):
        """Save dataset with metadata."""
        dataset_path = self.base_path / f"{name}.json"
        
        data = {
            "sentences": sentences,
            "metadata": metadata,
            "num_sentences": len(sentences),
            "created_at": datetime.now().isoformat()
        }
        
        with open(dataset_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Also save as pickle for faster loading
        with open(dataset_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(data, f)
    
    def load_dataset(self, name: str, use_pickle=True):
        """Load dataset."""
        if use_pickle:
            path = self.base_path / f"{name}.pkl"
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            path = self.base_path / f"{name}.json"
            with open(path) as f:
                return json.load(f)
```

### 3.5 Dataset Verification

**Verification Checklist**:

```python
# scripts/verify_datasets.py

def verify_dataset(name: str, expected_size: int, dataset: List[str]):
    """
    Verify dataset matches expectations.
    """
    checks = {
        "size_match": len(dataset) == expected_size,
        "no_empty": all(len(s) > 0 for s in dataset),
        "no_duplicates": len(dataset) == len(set(dataset)),
        "avg_length": sum(len(s.split()) for s in dataset) / len(dataset),
        "charset": "utf-8"  # All should be valid UTF-8
    }
    
    print(f"\nVerification for {name}:")
    print(f"  Expected size: {expected_size}")
    print(f"  Actual size: {len(dataset)}")
    print(f"  Size match: {'✓' if checks['size_match'] else '✗'}")
    print(f"  No empty: {'✓' if checks['no_empty'] else '✗'}")
    print(f"  No duplicates: {'✓' if checks['no_duplicates'] else '✗'}")
    print(f"  Avg words/sentence: {checks['avg_length']:.1f}")
    
    return all([checks['size_match'], checks['no_empty']])

# Expected sizes from paper
EXPECTED_SIZES = {
    "qqp": 164246,
    "qnli": 58645,
    "wiki": 203736,
    "books": 198085
}
```

## 4. Embedding Extraction

### 4.1 BERT Model Setup

```python
# src/models/bert_wrapper.py

import torch
from transformers import AutoModel, AutoTokenizer

class BERTEmbedder:
    """
    Extract sentence embeddings using BERT [CLS] token.
    """
    
    def __init__(
        self,
        model_name="bert-base-uncased",
        device=None,
        batch_size=32,
        max_length=512
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"Loaded {model_name} on {self.device}")
    
    @torch.no_grad()
    def embed_sentences(self, sentences: List[str], show_progress=True) -> np.ndarray:
        """
        Extract embeddings for a list of sentences.
        
        Returns:
            np.ndarray of shape (num_sentences, 768)
        """
        embeddings = []
        
        # Process in batches
        batches = [sentences[i:i+self.batch_size] 
                  for i in range(0, len(sentences), self.batch_size)]
        
        for batch in tqdm(batches, disable=not show_progress, desc="Extracting embeddings"):
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Extract [CLS] token embedding (first token, last layer)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
        
        return np.vstack(embeddings)
    
    def embed_and_cache(self, dataset_name: str, sentences: List[str], cache_dir="data/embeddings"):
        """
        Extract embeddings and cache to disk.
        """
        cache_path = Path(cache_dir) / f"{dataset_name}_embeddings.npy"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}")
            return np.load(cache_path)
        
        print(f"Extracting embeddings for {dataset_name}...")
        embeddings = self.embed_sentences(sentences)
        
        # Save to cache
        np.save(cache_path, embeddings)
        print(f"Cached embeddings to {cache_path}")
        
        return embeddings
```

### 4.2 Embedding Storage

**Format**: NumPy arrays (.npy) for efficiency

```python
# Structure:
# data/embeddings/
#   qqp_embeddings.npy      # (164246, 768)
#   qnli_embeddings.npy     # (58645, 768)
#   wiki_embeddings.npy     # (203736, 768)
#   books_embeddings.npy    # (198085, 768)
#   combined_metadata.json  # Metadata about all embeddings
```

## 5. Core Analysis Implementation

### 5.1 Neuron Analysis

```python
# src/analysis/neurons.py

class NeuronAnalyzer:
    """
    Analyze individual neuron activations.
    """
    
    def __init__(self, embeddings: np.ndarray, sentences: List[str], embedding_dim=768):
        self.embeddings = embeddings
        self.sentences = sentences
        self.embedding_dim = embedding_dim
    
    def get_top_activating(self, neuron_idx: int, k=10) -> List[Tuple[int, float, str]]:
        """
        Get top k activating sentences for a neuron.
        
        Args:
            neuron_idx: Index of neuron (0-767)
            k: Number of top sentences
        
        Returns:
            List of (sentence_idx, activation_score, sentence_text)
        """
        # Neuron activation = dot product with basis vector
        activations = self.embeddings[:, neuron_idx]
        
        # Get top k indices
        top_indices = np.argsort(activations)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((
                idx,
                float(activations[idx]),
                self.sentences[idx]
            ))
        
        return results
    
    def analyze_random_neurons(self, num_neurons=25, k=10, seed=42):
        """
        Analyze a random sample of neurons.
        """
        np.random.seed(seed)
        neuron_indices = np.random.choice(self.embedding_dim, num_neurons, replace=False)
        
        results = {}
        for neuron_idx in neuron_indices:
            results[neuron_idx] = self.get_top_activating(neuron_idx, k=k)
        
        return results
```

### 5.2 Random Directions

```python
# src/analysis/directions.py

class DirectionAnalyzer:
    """
    Analyze random directions in embedding space.
    """
    
    def __init__(self, embeddings: np.ndarray, sentences: List[str], embedding_dim=768):
        self.embeddings = embeddings
        self.sentences = sentences
        self.embedding_dim = embedding_dim
    
    def generate_random_direction(self, seed=None) -> np.ndarray:
        """
        Generate a random direction (unit vector).
        """
        if seed is not None:
            np.random.seed(seed)
        
        direction = np.random.randn(self.embedding_dim)
        direction = direction / np.linalg.norm(direction)  # Normalize
        
        return direction
    
    def get_top_activating_for_direction(self, direction: np.ndarray, k=10):
        """
        Get top k sentences that project most onto direction.
        """
        # Project embeddings onto direction
        projections = self.embeddings @ direction
        
        # Get top k
        top_indices = np.argsort(projections)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((
                idx,
                float(projections[idx]),
                self.sentences[idx]
            ))
        
        return results
    
    def analyze_random_directions(self, num_directions=33, k=10, seed=42):
        """
        Analyze multiple random directions.
        """
        results = {}
        for i in range(num_directions):
            direction = self.generate_random_direction(seed=seed+i)
            results[i] = {
                "direction": direction,
                "top_sentences": self.get_top_activating_for_direction(direction, k=k)
            }
        
        return results
```

### 5.3 Locality Score Computation

```python
# src/analysis/locality.py

from scipy.spatial.distance import cdist

class LocalityAnalyzer:
    """
    Compute locality scores for neuron directions.
    """
    
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
    
    def get_k_nearest_neighbors(self, sentence_idx: int, k=10):
        """
        Find k nearest neighbors of a sentence in embedding space.
        """
        # Compute distances to all other sentences
        distances = np.linalg.norm(
            self.embeddings - self.embeddings[sentence_idx],
            axis=1
        )
        
        # Get k+1 nearest (excluding self)
        nearest_indices = np.argsort(distances)[1:k+1]
        
        return nearest_indices
    
    def compute_pairwise_dots(self, indices: List[int]) -> np.ndarray:
        """
        Compute pairwise dot products for a set of embeddings.
        """
        subset = self.embeddings[indices]
        return subset @ subset.T
    
    def histogram_intersection(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Compute intersection over union (Jaccard) for histograms.
        """
        intersection = np.minimum(hist1, hist2).sum()
        union = np.maximum(hist1, hist2).sum()
        return intersection / union if union > 0 else 0.0
    
    def compute_locality_score(
        self,
        top_activating_indices: List[int],
        k_neighbors=10,
        num_bins=50
    ) -> float:
        """
        Compute locality score for a set of top activating sentences.
        
        Algorithm:
        1. For each top activating sentence, find its k nearest neighbors
        2. Compute distribution of dot products with neighbors (D_nearest)
        3. Compute distribution of dot products between top activating (D_top)
        4. Return histogram intersection between distributions
        """
        # Collect nearest neighbor dot products
        nearest_dots = []
        for idx in top_activating_indices:
            neighbors = self.get_k_nearest_neighbors(idx, k=k_neighbors)
            for neighbor_idx in neighbors:
                dot = self.embeddings[idx] @ self.embeddings[neighbor_idx]
                nearest_dots.append(dot)
        
        # Collect top activating pairwise dot products
        top_dots = []
        for i, idx1 in enumerate(top_activating_indices):
            for idx2 in top_activating_indices[i+1:]:
                dot = self.embeddings[idx1] @ self.embeddings[idx2]
                top_dots.append(dot)
        
        # Create histograms
        min_val = min(min(nearest_dots), min(top_dots))
        max_val = max(max(nearest_dots), max(top_dots))
        bins = np.linspace(min_val, max_val, num_bins)
        
        hist_nearest, _ = np.histogram(nearest_dots, bins=bins, density=True)
        hist_top, _ = np.histogram(top_dots, bins=bins, density=True)
        
        # Compute intersection
        score = self.histogram_intersection(hist_nearest, hist_top)
        
        return score
```

### 5.4 Monotonicity Analysis

```python
# src/analysis/monotonicity.py

class MonotonicityAnalyzer:
    """
    Analyze monotonic relationships between neuron activations and token frequencies.
    """
    
    def __init__(self, embeddings: np.ndarray, sentences: List[str], tokenizer):
        self.embeddings = embeddings
        self.sentences = sentences
        self.tokenizer = tokenizer
    
    def get_token_frequencies_by_quintile(self, neuron_idx: int, token: str) -> List[float]:
        """
        Get token frequency in each activation quintile.
        
        Returns:
            List of 5 frequencies (one per quintile)
        """
        activations = self.embeddings[:, neuron_idx]
        
        # Divide into quintiles
        quintile_bounds = np.percentile(activations, [20, 40, 60, 80])
        
        frequencies = []
        for i in range(5):
            if i == 0:
                mask = activations <= quintile_bounds[0]
            elif i == 4:
                mask = activations > quintile_bounds[3]
            else:
                mask = (activations > quintile_bounds[i-1]) & (activations <= quintile_bounds[i])
            
            quintile_sentences = [self.sentences[j] for j in np.where(mask)[0]]
            
            # Count token occurrences
            count = sum(token in s.lower() for s in quintile_sentences)
            freq = count / len(quintile_sentences) if len(quintile_sentences) > 0 else 0
            frequencies.append(freq)
        
        return frequencies
    
    def is_monotonic(self, values: List[float]) -> Tuple[bool, str]:
        """
        Check if values are monotonically increasing or decreasing.
        
        Returns:
            (is_monotonic, direction) where direction is 'increasing', 'decreasing', or 'none'
        """
        increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
        decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
        
        if increasing:
            return True, 'increasing'
        elif decreasing:
            return True, 'decreasing'
        else:
            return False, 'none'
    
    def analyze_token_monotonicity(
        self,
        tokens: List[str],
        num_neurons=768,
        min_frequency=100
    ) -> pd.DataFrame:
        """
        Analyze which neuron/token pairs show monotonic relationships.
        """
        results = []
        
        for neuron_idx in tqdm(range(num_neurons), desc="Analyzing neurons"):
            for token in tokens:
                frequencies = self.get_token_frequencies_by_quintile(neuron_idx, token)
                is_mono, direction = self.is_monotonic(frequencies)
                
                results.append({
                    'neuron': neuron_idx,
                    'token': token,
                    'is_monotonic': is_mono,
                    'direction': direction,
                    'frequencies': frequencies
                })
        
        return pd.DataFrame(results)
```

## 6. Visualization Pipeline

```python
# src/visualization/embeddings.py

import umap
import matplotlib.pyplot as plt
import seaborn as sns

class EmbeddingVisualizer:
    """
    Create visualizations of embedding space.
    """
    
    def __init__(self, embeddings_dict: Dict[str, np.ndarray]):
        """
        Args:
            embeddings_dict: {dataset_name: embeddings_array}
        """
        self.embeddings_dict = embeddings_dict
    
    def create_umap_plot(self, n_neighbors=15, min_dist=0.1, random_state=42):
        """
        Create UMAP visualization showing dataset clustering.
        """
        # Combine all embeddings
        all_embeddings = []
        all_labels = []
        
        for dataset_name, embeddings in self.embeddings_dict.items():
            all_embeddings.append(embeddings)
            all_labels.extend([dataset_name] * len(embeddings))
        
        all_embeddings = np.vstack(all_embeddings)
        
        # Fit UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            n_components=2
        )
        
        embedding_2d = reducer.fit_transform(all_embeddings)
        
        # Plot
        plt.figure(figsize=(12, 8))
        for dataset_name in self.embeddings_dict.keys():
            mask = np.array(all_labels) == dataset_name
            plt.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                label=dataset_name,
                alpha=0.5,
                s=1
            )
        
        plt.legend()
        plt.title("UMAP Visualization of Sentence Embeddings by Dataset")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        
        return plt.gcf()
```

## 7. Experiment Orchestration

```python
# scripts/07_run_full_pipeline.py

class ExperimentOrchestrator:
    """
    Orchestrate the full replication pipeline.
    """
    
    def __init__(self, config_path="config/experiments.yaml"):
        self.config = load_config("experiments")
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    def run_pipeline(self):
        """
        Execute full experimental pipeline.
        """
        # Step 1: Load datasets
        print("Step 1: Loading datasets...")
        datasets = self.load_all_datasets()
        
        # Step 2: Extract embeddings
        print("\nStep 2: Extracting embeddings...")
        embeddings = self.extract_all_embeddings(datasets)
        
        # Step 3: Neuron analysis
        print("\nStep 3: Analyzing neurons...")
        neuron_results = self.analyze_neurons(embeddings, datasets)
        
        # Step 4: Random direction analysis
        print("\nStep 4: Analyzing random directions...")
        direction_results = self.analyze_directions(embeddings, datasets)
        
        # Step 5: Locality analysis
        print("\nStep 5: Computing locality scores...")
        locality_results = self.compute_locality(embeddings, neuron_results, direction_results)
        
        # Step 6: Dataset separation
        print("\nStep 6: Analyzing dataset separation...")
        separation_results = self.analyze_separation(embeddings)
        
        # Step 7: Monotonicity analysis
        print("\nStep 7: Analyzing token monotonicity...")
        monotonicity_results = self.analyze_monotonicity(embeddings, datasets)
        
        # Step 8: Generate visualizations
        print("\nStep 8: Generating visualizations...")
        self.generate_visualizations(embeddings, all_results)
        
        # Step 9: Compile results
        print("\nStep 9: Compiling results...")
        self.compile_results(all_results)
        
        print("\n✓ Pipeline complete!")
```

## 8. Testing Strategy

### 8.1 Unit Tests

```python
# tests/test_embeddings.py

def test_embedding_extraction():
    """Test that embeddings have correct shape."""
    embedder = BERTEmbedder()
    sentences = ["This is a test.", "Another test sentence."]
    embeddings = embedder.embed_sentences(sentences)
    
    assert embeddings.shape == (2, 768)
    assert embeddings.dtype == np.float32

def test_neuron_analysis():
    """Test neuron analysis returns correct format."""
    embeddings = np.random.randn(100, 768)
    sentences = [f"Sentence {i}" for i in range(100)]
    
    analyzer = NeuronAnalyzer(embeddings, sentences)
    results = analyzer.get_top_activating(neuron_idx=0, k=10)
    
    assert len(results) == 10
    assert all(isinstance(r[2], str) for r in results)
```

### 8.2 Integration Tests

```python
# tests/test_pipeline.py

def test_full_pipeline_small():
    """Test full pipeline on small dataset."""
    # Use tiny subset for fast testing
    config = {
        'num_sentences': 100,
        'num_neurons': 3,
        'batch_size': 16
    }
    
    orchestrator = ExperimentOrchestrator(config)
    results = orchestrator.run_pipeline()
    
    assert 'embeddings' in results
    assert 'neuron_analysis' in results
```

## 9. Computational Requirements & Timeline

### 9.1 Estimated Compute Times

**On RTX 3090 (24GB VRAM)**:

| Task | Time | GPU Memory |
|------|------|------------|
| Extract embeddings (all datasets) | 2-3 hours | 8GB |
| Neuron analysis | 30 min | 4GB |
| Locality computation | 1-2 hours | CPU-bound |
| UMAP visualization | 20 min | CPU-bound |
| Monotonicity analysis | 3-4 hours | CPU-bound |
| **Total** | **~10 hours** | |

**Optimizations**:
- Batch size can be increased with more VRAM
- Embeddings are cached after first run
- Parallel processing for locality scores

### 9.2 Implementation Timeline

**Week 1**: Setup & Data
- Environment setup
- Dataset acquisition
- Preprocessing pipeline
- **Deliverable**: All datasets ready

**Week 2**: Embedding & Core Analysis
- Embedding extraction
- Neuron analysis implementation
- Random direction analysis
- **Deliverable**: Core analysis code working

**Week 3**: Advanced Analysis
- Locality scores
- Monotonicity analysis
- Dataset separation metrics
- **Deliverable**: All metrics computed

**Week 4**: Visualization & Validation
- UMAP plots
- Statistical analysis
- Results compilation
- **Deliverable**: Complete replication

## 10. Results Validation

### 10.1 Key Metrics to Match

From the paper, we should reproduce:

1. **Pattern Detection Rates**:
   - Neurons: ~80% contain patterns
   - Random directions: ~82%
   - Random sentences: ~14%

2. **Locality Scores**:
   - Meaningful neurons: ~0.026
   - Meaningless neurons: ~0.010
   - p-value < 0.001

3. **Monotonicity Rates**:
   - Single dataset: ~27%
   - All datasets: ~2%

4. **Dataset Separation**:
   - SVM classifier accuracy: >90%
   - Clear UMAP clustering visible

### 10.2 Acceptance Criteria

**Success** if:
- ✓ All metrics within ±10% of paper
- ✓ Qualitative patterns match (UMAP clusters, etc.)
- ✓ Statistical significance matches

**Partial Success** if:
- ✓ Core findings replicate (illusion exists)
- ⚠ Some quantitative differences (due to dataset substitutions)

**Failure** if:
- ✗ Cannot reproduce main finding
- ✗ Contradictory results

## 11. Documentation Requirements

For reproducibility, document:

1. **Exact versions** of all libraries
2. **Random seeds** used throughout
3. **Dataset sources** and any substitutions
4. **Hyperparameters** for all experiments
5. **Hardware** specifications
6. **Deviations** from original paper

## 12. Deployment & Sharing

### 12.1 Code Release

```bash
# Create release package
scripts/create_release.sh

# Includes:
# - README with setup instructions
# - requirements.txt with pinned versions
# - Sample data for testing
# - Pre-computed results (optional)
```

### 12.2 Interactive Demo

Consider creating:
- **Streamlit app** for exploring neuron patterns
- **Gradio interface** for analyzing custom text
- **Jupyter notebooks** for educational purposes

## Summary

This technical design provides a complete blueprint for replicating the paper. Key challenges:
- **BookCorpus unavailability** → Use PG-19 or similar substitute
- **Exact dataset matching** → Document any differences
- **Computational resources** → ~10 hours on consumer GPU
- **Reproducibility** → Pin all versions, document all decisions

The modular architecture allows for:
- Easy extension to new models
- Adding new datasets
- Iterative development
- Testing each component independently

**Next Steps**: 
1. Set up environment
2. Implement dataset loaders
3. Test on small subset
4. Scale to full datasets
5. Validate results against paper
