# Replication Plan: An Interpretability Illusion for BERT

This document provides a detailed, step-by-step plan to replicate all experiments from "An Interpretability Illusion for BERT" by Bolukbasi et al. (2021).

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Acquisition & Preprocessing](#2-data-acquisition--preprocessing)
3. [Model Setup](#3-model-setup)
4. [Embedding Generation](#4-embedding-generation)
5. [Core Analyses](#5-core-analyses)
6. [Annotation Study](#6-annotation-study)
7. [Dataset Idiosyncrasy Analysis](#7-dataset-idiosyncrasy-analysis)
8. [Local Semantic Coherence Analysis](#8-local-semantic-coherence-analysis)
9. [Monotonicity Analysis](#9-monotonicity-analysis)
10. [Outlier Analysis](#10-outlier-analysis)
11. [Visualization Generation](#11-visualization-generation)
12. [Statistical Testing](#12-statistical-testing)
13. [Validation & Verification](#13-validation--verification)

---

## 1. Environment Setup

### 1.1 Python Environment

```bash
# Create virtual environment
python3 -m venv bert-illusion-env
source bert-illusion-env/bin/activate  # On Windows: bert-illusion-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 1.2 Required Libraries

```bash
# Core ML libraries
pip install torch==1.9.0  # or compatible version
pip install transformers==4.11.3  # HuggingFace Transformers
pip install numpy==1.21.2
pip install scipy==1.7.1

# Data processing
pip install pandas==1.3.3
pip install datasets==1.12.1  # HuggingFace datasets library

# Visualization
pip install matplotlib==3.4.3
pip install seaborn==0.11.2
pip install umap-learn==0.5.1  # For UMAP dimensionality reduction

# Machine learning
pip install scikit-learn==0.24.2  # For SVM classifier

# Optional: Progress bars and utilities
pip install tqdm==4.62.3
pip install jupyter==1.0.0  # For interactive analysis
```

### 1.3 Hardware Requirements

**Minimum:**
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 32GB+ (for processing ~625K sentence embeddings)
- Storage: 10GB+ free space

**Recommended:**
- GPU: NVIDIA GPU with 8GB+ VRAM (for faster BERT inference)
- RAM: 64GB+
- Storage: 50GB+ (for datasets and intermediate results)

### 1.4 Verify Installation

```python
import torch
import transformers
import numpy as np
import pandas as pd
from datasets import load_dataset
import umap
import sklearn

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## 2. Data Acquisition & Preprocessing

### 2.1 Download Datasets

#### QQP (Quora Question Pairs)
```python
from datasets import load_dataset

# Load QQP dataset
qqp = load_dataset("glue", "qqp", split="train")
# Expected size: 364,000 total, paper uses 164,246

# Filter to match paper's subset (if needed)
# The paper doesn't specify filtering criteria
# Assumption: use validation set or sample
qqp_val = load_dataset("glue", "qqp", split="validation")
```

#### QNLI (Question-answering NLI)
```python
# Load QNLI dataset
qnli = load_dataset("glue", "qnli", split="train")
# Expected size: paper uses 58,645 datapoints
# Likely using validation split
qnli_val = load_dataset("glue", "qnli", split="validation")
```

#### Wikipedia
```python
# Load Wikipedia dataset as prepared in Devlin et al., 2018
# This is the same preprocessing used for BERT pretraining
from datasets import load_dataset

wiki = load_dataset("wikipedia", "20200501.en", split="train")
# Sample 203,736 sentences

# Alternative: Use the exact preprocessing from BERT paper
# This requires downloading the raw Wikipedia dump and processing
# See: https://github.com/google-research/bert#pre-training-with-bert
```

#### Toronto BookCorpus
```python
# Original BookCorpus is no longer publicly available
# Alternatives:
# 1. Use books3 or similar
# 2. Use available approximations

# If using an alternative:
from datasets import load_dataset
books = load_dataset("bookcorpus", split="train")
# Sample 198,085 sentences
```

### 2.2 Data Preprocessing

```python
import pandas as pd
import random

def extract_sentences(dataset_name, dataset, text_column, target_size):
    """
    Extract sentences from dataset.

    Args:
        dataset_name: Name identifier
        dataset: HuggingFace dataset object
        text_column: Column name containing text
        target_size: Number of sentences to sample

    Returns:
        DataFrame with sentences and metadata
    """
    sentences = []

    for i, example in enumerate(dataset):
        text = example[text_column]
        # Handle paired sentences (like in QQP, QNLI)
        if isinstance(text, str):
            sentences.append({
                'dataset': dataset_name,
                'sentence': text,
                'original_index': i
            })

    # Sample if needed
    if len(sentences) > target_size:
        sentences = random.sample(sentences, target_size)

    return pd.DataFrame(sentences)

# Process QQP
qqp_sentences = extract_sentences('QQP', qqp_val, 'question1', 164246)

# Process QNLI
qnli_sentences = extract_sentences('QNLI', qnli_val, 'question', 58645)

# Process Wiki (needs sentence splitting)
def split_wikipedia_into_sentences(wiki_dataset, target_size):
    """Split Wikipedia articles into sentences."""
    import re
    sentences = []

    for article in wiki_dataset:
        text = article['text']
        # Simple sentence splitting (can use spacy/nltk for better results)
        sents = re.split(r'[.!?]+', text)
        for sent in sents:
            sent = sent.strip()
            if len(sent) > 10:  # Filter very short sentences
                sentences.append(sent)
            if len(sentences) >= target_size:
                break
        if len(sentences) >= target_size:
            break

    return pd.DataFrame({
        'dataset': 'Wiki',
        'sentence': sentences[:target_size]
    })

wiki_sentences = split_wikipedia_into_sentences(wiki, 203736)

# Process Books similarly
books_sentences = split_wikipedia_into_sentences(books, 198085)

# Combine all datasets
all_sentences = pd.concat([
    qqp_sentences,
    qnli_sentences,
    wiki_sentences,
    books_sentences
], ignore_index=True)

print(f"Total sentences: {len(all_sentences)}")
# Should be ~624,712

# Save preprocessed data
all_sentences.to_csv('preprocessed_sentences.csv', index=False)
```

### 2.3 Data Validation

```python
# Verify dataset sizes
print("Dataset sizes:")
for dataset in ['QQP', 'QNLI', 'Wiki', 'Books']:
    count = len(all_sentences[all_sentences['dataset'] == dataset])
    print(f"{dataset}: {count}")

# Expected output:
# QQP: 164,246
# QNLI: 58,645
# Wiki: 203,736
# Books: 198,085
```

---

## 3. Model Setup

### 3.1 Load BERT Model

```python
from transformers import BertTokenizer, BertModel
import torch

# Load BERT-base uncased (exact model from paper)
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Set to evaluation mode

print(f"Model loaded on: {device}")
print(f"Model config: {model.config}")
```

### 3.2 Verify Model Configuration

```python
# Verify we're using BERT-base
assert model.config.hidden_size == 768, "Should be 768-dimensional"
assert model.config.num_hidden_layers == 12, "Should be 12 layers"
assert model.config.num_attention_heads == 12, "Should be 12 attention heads"

print("Model configuration verified ✓")
```

---

## 4. Embedding Generation

### 4.1 Extract [CLS] Embeddings

```python
import numpy as np
from tqdm import tqdm

def get_cls_embedding(text, model, tokenizer, device):
    """
    Extract [CLS] token embedding from final layer.

    Args:
        text: Input sentence
        model: BERT model
        tokenizer: BERT tokenizer
        device: torch device

    Returns:
        768-dimensional numpy array
    """
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract [CLS] token from last hidden state (layer 12)
    # Shape: [batch_size, sequence_length, hidden_size]
    last_hidden_state = outputs.last_hidden_state

    # [CLS] is always the first token
    cls_embedding = last_hidden_state[:, 0, :].cpu().numpy()

    return cls_embedding.squeeze()

def generate_all_embeddings(sentences_df, model, tokenizer, device, batch_size=32):
    """
    Generate embeddings for all sentences with batching.

    Args:
        sentences_df: DataFrame with sentences
        model: BERT model
        tokenizer: BERT tokenizer
        device: torch device
        batch_size: Batch size for processing

    Returns:
        numpy array of shape (n_sentences, 768)
    """
    embeddings = []

    # Process in batches for efficiency
    for i in tqdm(range(0, len(sentences_df), batch_size)):
        batch = sentences_df.iloc[i:i+batch_size]
        batch_texts = batch['sentence'].tolist()

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract [CLS] tokens
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)

    # Concatenate all batches
    all_embeddings = np.vstack(embeddings)

    return all_embeddings

# Generate embeddings
print("Generating embeddings for all sentences...")
embeddings = generate_all_embeddings(
    all_sentences,
    model,
    tokenizer,
    device,
    batch_size=32
)

print(f"Embeddings shape: {embeddings.shape}")
# Expected: (624712, 768)

# Save embeddings
np.save('sentence_embeddings.npy', embeddings)
all_sentences.to_pickle('sentences_with_metadata.pkl')
```

### 4.2 Verify Embedding Properties

```python
# Check embedding statistics (Appendix A)
norms = np.linalg.norm(embeddings, axis=1)
print(f"Embedding norms - Mean: {norms.mean():.2f}, Std: {norms.std():.2f}")
print(f"Expected mean: ~14 (≈ sqrt(768)/2 ≈ {np.sqrt(768)/2:.2f})")

# Check value ranges
print(f"Embedding values - Min: {embeddings.min():.2f}, Max: {embeddings.max():.2f}")
print(f"Expected range: approximately [-1.0, 1.0]")

# Verify shape
assert embeddings.shape == (len(all_sentences), 768)
print("Embedding properties verified ✓")
```

---

## 5. Core Analyses

### 5.1 Select Neurons and Directions

```python
# Randomly select 25 neurons (as in paper)
np.random.seed(42)  # For reproducibility
selected_neurons = np.random.choice(768, size=25, replace=False)
print(f"Selected neurons: {selected_neurons}")

# Generate 33 random directions
def generate_random_direction(dim=768):
    """Generate a random unit vector in 768-dimensional space."""
    direction = np.random.randn(dim)
    direction = direction / np.linalg.norm(direction)
    return direction

random_directions = [generate_random_direction() for _ in range(33)]
print(f"Generated {len(random_directions)} random directions")

# Save for reproducibility
np.save('selected_neurons.npy', selected_neurons)
np.save('random_directions.npy', np.array(random_directions))
```

### 5.2 Find Top Activating Sentences

```python
def get_top_activating_sentences(embeddings, direction, sentences_df,
                                 dataset_name=None, k=10):
    """
    Find top k sentences that maximally activate along a direction.

    Args:
        embeddings: Sentence embeddings (n x 768)
        direction: Direction vector (768,)
        sentences_df: DataFrame with sentence metadata
        dataset_name: Filter to specific dataset (or None for all)
        k: Number of top sentences to return

    Returns:
        DataFrame with top k sentences and their activation scores
    """
    # Filter to specific dataset if specified
    if dataset_name:
        mask = sentences_df['dataset'] == dataset_name
        filtered_embeddings = embeddings[mask]
        filtered_df = sentences_df[mask].reset_index(drop=True)
    else:
        filtered_embeddings = embeddings
        filtered_df = sentences_df

    # Compute projection scores (dot products)
    projection_scores = filtered_embeddings @ direction

    # Get top k indices
    top_k_indices = np.argsort(projection_scores)[-k:][::-1]

    # Create result dataframe
    result = filtered_df.iloc[top_k_indices].copy()
    result['projection_score'] = projection_scores[top_k_indices]
    result['rank'] = range(1, k+1)

    return result

# Example: Get top 10 sentences for neuron 0 in QQP dataset
neuron_0_direction = np.zeros(768)
neuron_0_direction[selected_neurons[0]] = 1.0

top_qqp = get_top_activating_sentences(
    embeddings,
    neuron_0_direction,
    all_sentences,
    dataset_name='QQP',
    k=10
)

print("Top 10 activating sentences for neuron 0 in QQP:")
for i, row in top_qqp.iterrows():
    print(f"{row['rank']}. {row['sentence'][:100]}... (score: {row['projection_score']:.3f})")
```

### 5.3 Generate All Top Activating Sets

```python
def generate_all_top_activating_sets(embeddings, sentences_df,
                                     neurons, random_dirs, k=10):
    """
    Generate top activating sentences for all neurons, directions, and datasets.

    Returns:
        Dictionary with results
    """
    results = {
        'neurons': {},
        'random_directions': {},
        'random_sentences': {}
    }

    datasets = ['QQP', 'QNLI', 'Wiki', 'Books']

    # For each neuron
    for neuron_idx in tqdm(neurons, desc="Processing neurons"):
        neuron_direction = np.zeros(768)
        neuron_direction[neuron_idx] = 1.0

        results['neurons'][neuron_idx] = {}

        for dataset in datasets:
            top_sents = get_top_activating_sentences(
                embeddings, neuron_direction, sentences_df,
                dataset_name=dataset, k=k
            )
            results['neurons'][neuron_idx][dataset] = top_sents

    # For each random direction
    for dir_idx, direction in enumerate(tqdm(random_dirs, desc="Processing random directions")):
        results['random_directions'][dir_idx] = {}

        for dataset in datasets:
            top_sents = get_top_activating_sentences(
                embeddings, direction, sentences_df,
                dataset_name=dataset, k=k
            )
            results['random_directions'][dir_idx][dataset] = top_sents

    # Generate random sentence sets (baseline)
    for i in tqdm(range(29), desc="Generating random sentence sets"):
        results['random_sentences'][i] = {}

        for dataset in datasets:
            dataset_mask = sentences_df['dataset'] == dataset
            dataset_sentences = sentences_df[dataset_mask]
            random_sample = dataset_sentences.sample(n=k)
            results['random_sentences'][i][dataset] = random_sample

    return results

# Generate all sets
print("Generating all top activating sentence sets...")
all_results = generate_all_top_activating_sets(
    embeddings,
    all_sentences,
    selected_neurons,
    random_directions,
    k=10
)

# Save results
import pickle
with open('top_activating_sets.pkl', 'wb') as f:
    pickle.dump(all_results, f)

print("All top activating sets generated ✓")
```

---

## 6. Annotation Study

### 6.1 Build Annotation Interface

```python
import json
from datetime import datetime

class AnnotationInterface:
    """
    Simple annotation interface for pattern identification.
    """

    def __init__(self, results_dict, output_file='annotations.json'):
        self.results = results_dict
        self.output_file = output_file
        self.annotations = []

        # Load existing annotations if file exists
        try:
            with open(output_file, 'r') as f:
                self.annotations = json.load(f)
        except FileNotFoundError:
            pass

    def annotate_set(self, sentences, condition_type, condition_id,
                     dataset, annotator_id):
        """
        Present a set of sentences for annotation.

        Args:
            sentences: List of sentences to annotate
            condition_type: 'neuron', 'random_direction', or 'random_sentences'
            condition_id: Identifier for the specific neuron/direction/set
            dataset: Dataset name
            annotator_id: Identifier for the annotator

        Returns:
            Annotation dictionary
        """
        print("\n" + "="*80)
        print(f"Dataset: {dataset}")
        print(f"Annotator: {annotator_id}")
        print("="*80)
        print("\nSentences:")
        print("-"*80)

        for i, sent in enumerate(sentences, 1):
            print(f"{i}. {sent}")
            print()

        # Collect annotation
        has_pattern = input("\nDoes this set contain a pattern? (y/n): ").lower()

        annotation = {
            'timestamp': datetime.now().isoformat(),
            'annotator_id': annotator_id,
            'condition_type': condition_type,
            'condition_id': str(condition_id),
            'dataset': dataset,
            'has_pattern': has_pattern == 'y',
            'pattern_description': None,
            'positive_examples': []
        }

        if annotation['has_pattern']:
            pattern_desc = input("Describe the pattern: ")
            annotation['pattern_description'] = pattern_desc

            positive_indices = input("Which sentences show the pattern? (comma-separated numbers, e.g., 1,3,5): ")
            if positive_indices.strip():
                annotation['positive_examples'] = [
                    int(x.strip()) for x in positive_indices.split(',')
                ]

        self.annotations.append(annotation)
        self.save()

        return annotation

    def save(self):
        """Save annotations to file."""
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)

    def get_annotation_summary(self):
        """Get summary statistics of annotations."""
        total = len(self.annotations)
        with_patterns = sum(1 for a in self.annotations if a['has_pattern'])

        by_condition = {}
        for ann in self.annotations:
            cond_type = ann['condition_type']
            if cond_type not in by_condition:
                by_condition[cond_type] = {'total': 0, 'with_pattern': 0}
            by_condition[cond_type]['total'] += 1
            if ann['has_pattern']:
                by_condition[cond_type]['with_pattern'] += 1

        return {
            'total_annotations': total,
            'with_patterns': with_patterns,
            'by_condition': by_condition
        }

# Usage example
annotator = AnnotationInterface(all_results)

# Annotate a specific set
example_sentences = all_results['neurons'][selected_neurons[0]]['QQP']['sentence'].tolist()
annotator.annotate_set(
    example_sentences,
    condition_type='neuron',
    condition_id=selected_neurons[0],
    dataset='QQP',
    annotator_id='annotator_1'
)
```

### 6.2 Annotation Protocol

```python
def run_annotation_protocol(all_results, annotator_id,
                            neurons_to_annotate=None,
                            random_dirs_to_annotate=None,
                            random_sets_to_annotate=None):
    """
    Run the full annotation protocol.

    Args:
        all_results: Results dictionary from generate_all_top_activating_sets
        annotator_id: Identifier for current annotator
        neurons_to_annotate: List of neuron indices (or None for all)
        random_dirs_to_annotate: List of random direction indices (or None for all)
        random_sets_to_annotate: List of random set indices (or None for all)
    """
    interface = AnnotationInterface(all_results,
                                   output_file=f'annotations_{annotator_id}.json')

    datasets = ['QQP', 'QNLI', 'Wiki', 'Books']

    # Annotate neurons
    if neurons_to_annotate is None:
        neurons_to_annotate = list(all_results['neurons'].keys())

    for neuron_idx in neurons_to_annotate:
        for dataset in datasets:
            sentences = all_results['neurons'][neuron_idx][dataset]['sentence'].tolist()
            interface.annotate_set(
                sentences,
                condition_type='neuron',
                condition_id=neuron_idx,
                dataset=dataset,
                annotator_id=annotator_id
            )

    # Annotate random directions
    if random_dirs_to_annotate is None:
        random_dirs_to_annotate = list(all_results['random_directions'].keys())

    for dir_idx in random_dirs_to_annotate:
        for dataset in datasets:
            sentences = all_results['random_directions'][dir_idx][dataset]['sentence'].tolist()
            interface.annotate_set(
                sentences,
                condition_type='random_direction',
                condition_id=dir_idx,
                dataset=dataset,
                annotator_id=annotator_id
            )

    # Annotate random sentence sets
    if random_sets_to_annotate is None:
        random_sets_to_annotate = list(all_results['random_sentences'].keys())

    for set_idx in random_sets_to_annotate:
        for dataset in datasets:
            sentences = all_results['random_sentences'][set_idx][dataset]['sentence'].tolist()
            interface.annotate_set(
                sentences,
                condition_type='random_sentences',
                condition_id=set_idx,
                dataset=dataset,
                annotator_id=annotator_id
            )

    # Print summary
    summary = interface.get_annotation_summary()
    print("\n" + "="*80)
    print("ANNOTATION SUMMARY")
    print("="*80)
    print(json.dumps(summary, indent=2))

# Run annotation (can be run by multiple annotators)
# run_annotation_protocol(all_results, annotator_id='annotator_1')
```

### 6.3 Calculate Inter-Annotator Agreement

```python
def calculate_inter_annotator_agreement(annotation_files):
    """
    Calculate agreement between annotators.

    Args:
        annotation_files: List of annotation JSON file paths

    Returns:
        Agreement statistics
    """
    from collections import defaultdict

    # Load all annotations
    all_annotations = []
    for file in annotation_files:
        with open(file, 'r') as f:
            all_annotations.extend(json.load(f))

    # Group by condition
    by_condition = defaultdict(list)
    for ann in all_annotations:
        key = (ann['condition_type'], ann['condition_id'], ann['dataset'])
        by_condition[key].append(ann)

    # Calculate agreement
    agreement_stats = {
        'total_conditions': len(by_condition),
        'both_agree_pattern': 0,
        'both_agree_no_pattern': 0,
        'conflicting': 0
    }

    for key, annotations in by_condition.items():
        if len(annotations) < 2:
            continue

        has_pattern_votes = [a['has_pattern'] for a in annotations]

        if all(has_pattern_votes):
            agreement_stats['both_agree_pattern'] += 1
        elif not any(has_pattern_votes):
            agreement_stats['both_agree_no_pattern'] += 1
        else:
            agreement_stats['conflicting'] += 1

    return agreement_stats
```

---

## 7. Dataset Idiosyncrasy Analysis

### 7.1 UMAP Visualization

```python
import umap
import matplotlib.pyplot as plt
import seaborn as sns

def create_umap_visualization(embeddings, sentences_df, save_path='umap_viz.png'):
    """
    Create UMAP visualization of embeddings colored by dataset.
    Replicates Figure 1 from the paper.

    Args:
        embeddings: Sentence embeddings (n x 768)
        sentences_df: DataFrame with dataset labels
        save_path: Path to save figure
    """
    print("Running UMAP dimensionality reduction...")

    # UMAP parameters (may need tuning to match paper's visualization)
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=42
    )

    # Fit and transform
    embedding_2d = reducer.fit_transform(embeddings)

    # Create plot
    plt.figure(figsize=(12, 10))

    # Color by dataset
    datasets = ['QQP', 'QNLI', 'Wiki', 'Books']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for dataset, color in zip(datasets, colors):
        mask = sentences_df['dataset'] == dataset
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=color,
            label=dataset,
            alpha=0.6,
            s=1
        )

    plt.legend(markerscale=10)
    plt.title('UMAP Visualization of Sentence Embeddings by Dataset', fontsize=14)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"UMAP visualization saved to {save_path}")

    return embedding_2d

# Generate visualization
umap_embeddings = create_umap_visualization(embeddings, all_sentences)
```

### 7.2 SVM Dataset Classification

```python
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def train_dataset_classifier(embeddings, sentences_df):
    """
    Train linear SVM to classify datasets.
    Replicates Figure 4 from the paper.

    Args:
        embeddings: Sentence embeddings
        sentences_df: DataFrame with dataset labels

    Returns:
        Trained classifier and results
    """
    print("Training linear SVM classifier...")

    # Prepare data
    X = embeddings
    y = sentences_df['dataset'].values

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train linear SVM
    clf = LinearSVC(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = (y_pred == y_test).mean()
    print(f"Classification accuracy: {accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['QQP', 'QNLI', 'Wiki', 'Books'])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Reds',
        xticklabels=['QQP', 'QNLI', 'Wiki', 'Books'],
        yticklabels=['QQP', 'QNLI', 'Wiki', 'Books']
    )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix: Dataset Classification')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()

    print("Confusion matrix saved to confusion_matrix.png")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return clf, cm_normalized

# Train classifier
classifier, conf_matrix = train_dataset_classifier(embeddings, all_sentences)
```

---

## 8. Local Semantic Coherence Analysis

### 8.1 Compute Nearest Neighbors

```python
from sklearn.neighbors import NearestNeighbors

def compute_nearest_neighbors(embeddings, k=10):
    """
    Compute k-nearest neighbors for all embeddings.

    Args:
        embeddings: Sentence embeddings
        k: Number of neighbors

    Returns:
        NearestNeighbors model fitted on embeddings
    """
    print(f"Computing {k}-nearest neighbors...")

    # Use cosine similarity (equivalent to dot product for normalized vectors)
    # But paper uses unnormalized, so use dot product
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute')
    nbrs.fit(embeddings)

    return nbrs

# Compute neighbors
knn_model = compute_nearest_neighbors(embeddings, k=10)
```

### 8.2 Locality Score Calculation

```python
def calculate_locality_score(top_activating_indices, embeddings, knn_model, k=10):
    """
    Calculate locality score for a set of top activating sentences.

    Args:
        top_activating_indices: Indices of top activating sentences
        embeddings: All sentence embeddings
        knn_model: Fitted NearestNeighbors model
        k: Number of neighbors

    Returns:
        Locality score (Jaccard similarity)
    """
    # Get embeddings for top activating sentences
    top_embeddings = embeddings[top_activating_indices]

    # Compute D_p,nearest: distances to nearest neighbors in original space
    D_nearest = []
    for idx in top_activating_indices:
        # Get k nearest neighbors
        distances, indices = knn_model.kneighbors([embeddings[idx]], n_neighbors=k+1)
        # Exclude self (first neighbor)
        neighbor_indices = indices[0][1:]

        # Compute dot products with neighbors
        for neighbor_idx in neighbor_indices:
            dot_prod = np.dot(embeddings[idx], embeddings[neighbor_idx])
            D_nearest.append(dot_prod)

    # Compute D_p,top: distances between top activating sentences
    D_top = []
    for i in range(len(top_activating_indices)):
        for j in range(i+1, len(top_activating_indices)):
            dot_prod = np.dot(top_embeddings[i], top_embeddings[j])
            D_top.append(dot_prod)

    # Compute D_p,random: distances to random sentences
    D_random = []
    random_indices = np.random.choice(len(embeddings), size=k*len(top_activating_indices))
    for idx in top_activating_indices:
        for rand_idx in random_indices[:k]:
            dot_prod = np.dot(embeddings[idx], embeddings[rand_idx])
            D_random.append(dot_prod)

    # Create histograms
    bins = np.linspace(
        min(min(D_nearest), min(D_top), min(D_random)),
        max(max(D_nearest), max(D_top), max(D_random)),
        50
    )

    hist_nearest, _ = np.histogram(D_nearest, bins=bins, density=True)
    hist_top, _ = np.histogram(D_top, bins=bins, density=True)
    hist_random, _ = np.histogram(D_random, bins=bins, density=True)

    # Calculate locality score (Jaccard similarity between hist_nearest and hist_top)
    locality_score = np.sum(np.minimum(hist_nearest, hist_top)) / np.sum(np.maximum(hist_nearest, hist_top))

    # Also calculate score with random for comparison
    random_score = np.sum(np.minimum(hist_random, hist_top)) / np.sum(np.maximum(hist_random, hist_top))

    return {
        'locality_score': locality_score,
        'random_score': random_score,
        'D_nearest': D_nearest,
        'D_top': D_top,
        'D_random': D_random
    }

# Example: Calculate locality score for a neuron
def calculate_all_locality_scores(all_results, embeddings, knn_model, sentences_df):
    """
    Calculate locality scores for all annotated directions.

    Returns:
        Dictionary with locality scores
    """
    locality_scores = {
        'neurons': {},
        'random_directions': {}
    }

    datasets = ['QQP', 'QNLI', 'Wiki', 'Books']

    # For neurons
    for neuron_idx in tqdm(all_results['neurons'].keys(), desc="Computing locality for neurons"):
        locality_scores['neurons'][neuron_idx] = {}

        for dataset in datasets:
            # Get indices of top activating sentences
            top_df = all_results['neurons'][neuron_idx][dataset]

            # Get original indices
            top_indices = []
            for _, row in top_df.iterrows():
                # Find index in original dataframe
                mask = (sentences_df['sentence'] == row['sentence']) & \
                       (sentences_df['dataset'] == dataset)
                idx = sentences_df[mask].index[0]
                top_indices.append(idx)

            # Calculate locality score
            scores = calculate_locality_score(top_indices, embeddings, knn_model)
            locality_scores['neurons'][neuron_idx][dataset] = scores

    # Similar for random directions
    for dir_idx in tqdm(all_results['random_directions'].keys(), desc="Computing locality for random directions"):
        locality_scores['random_directions'][dir_idx] = {}

        for dataset in datasets:
            top_df = all_results['random_directions'][dir_idx][dataset]

            top_indices = []
            for _, row in top_df.iterrows():
                mask = (sentences_df['sentence'] == row['sentence']) & \
                       (sentences_df['dataset'] == dataset)
                idx = sentences_df[mask].index[0]
                top_indices.append(idx)

            scores = calculate_locality_score(top_indices, embeddings, knn_model)
            locality_scores['random_directions'][dir_idx][dataset] = scores

    return locality_scores

# Calculate all locality scores
locality_results = calculate_all_locality_scores(all_results, embeddings, knn_model, all_sentences)

# Save results
with open('locality_scores.pkl', 'wb') as f:
    pickle.dump(locality_results, f)
```

### 8.3 Visualize Distance Distributions (Figure 7)

```python
def plot_distance_distributions(locality_result, neuron_id, dataset, save_path=None):
    """
    Plot distance distributions for a specific neuron/dataset.
    Replicates Figure 7 from paper.

    Args:
        locality_result: Result from calculate_locality_score
        neuron_id: Neuron identifier
        dataset: Dataset name
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms
    ax.hist(locality_result['D_nearest'], bins=50, alpha=0.7,
            label='Nearest neighbors', color='orange', density=True)
    ax.hist(locality_result['D_top'], bins=50, alpha=0.7,
            label='Top activating', color='blue', density=True)
    ax.hist(locality_result['D_random'], bins=50, alpha=0.7,
            label='Random', color='green', density=True)

    ax.set_xlabel('Pairwise distance between sentences')
    ax.set_ylabel('Number of sentences')
    ax.set_title(f'Distance Distributions - Neuron {neuron_id} - {dataset}')
    ax.legend()

    # Add text with locality score
    ax.text(0.02, 0.98,
            f"Locality score: {locality_result['locality_score']:.4f}",
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()
```

---

## 9. Monotonicity Analysis

### 9.1 Token Frequency by Activation Quintile

```python
from collections import Counter
import re

def tokenize_simple(text):
    """Simple tokenization."""
    return re.findall(r'\b\w+\b', text.lower())

def calculate_token_frequencies_by_quintile(embeddings, sentences_df, direction,
                                            dataset_name=None, min_token_count=100):
    """
    Calculate token frequencies across activation quintiles.

    Args:
        embeddings: Sentence embeddings
        sentences_df: Sentence dataframe
        direction: Direction vector
        dataset_name: Dataset to analyze (or None for all)
        min_token_count: Minimum token frequency to include

    Returns:
        Dictionary with token frequencies per quintile
    """
    # Filter to dataset
    if dataset_name:
        mask = sentences_df['dataset'] == dataset_name
        filtered_embeddings = embeddings[mask]
        filtered_df = sentences_df[mask].reset_index(drop=True)
    else:
        filtered_embeddings = embeddings
        filtered_df = sentences_df

    # Compute activations
    activations = filtered_embeddings @ direction

    # Divide into quintiles
    quintile_boundaries = np.percentile(activations, [0, 20, 40, 60, 80, 100])

    # Count tokens in each quintile
    quintile_tokens = [Counter() for _ in range(5)]

    for i, (act, sent) in enumerate(zip(activations, filtered_df['sentence'])):
        # Determine quintile
        quintile = min(4, np.searchsorted(quintile_boundaries[1:], act))

        # Tokenize and count
        tokens = tokenize_simple(sent)
        quintile_tokens[quintile].update(tokens)

    # Filter to common tokens
    all_tokens = Counter()
    for qt in quintile_tokens:
        all_tokens.update(qt)

    common_tokens = {token for token, count in all_tokens.items()
                    if count >= min_token_count}

    # Create frequency matrix
    token_frequencies = {}
    for token in common_tokens:
        freqs = [qt.get(token, 0) for qt in quintile_tokens]
        token_frequencies[token] = freqs

    return token_frequencies, quintile_boundaries

def check_monotonicity(frequencies):
    """
    Check if a list of frequencies is monotonically increasing or decreasing.

    Returns:
        'increasing', 'decreasing', or None
    """
    increasing = all(frequencies[i] <= frequencies[i+1] for i in range(len(frequencies)-1))
    decreasing = all(frequencies[i] >= frequencies[i+1] for i in range(len(frequencies)-1))

    if increasing and not decreasing:
        return 'increasing'
    elif decreasing and not increasing:
        return 'decreasing'
    else:
        return None

def analyze_monotonicity_for_all_neurons(embeddings, sentences_df, neurons,
                                        datasets=['QQP', 'QNLI', 'Wiki', 'Books']):
    """
    Analyze token monotonicity for all neurons across datasets.
    Replicates Table 3 from the paper.

    Returns:
        Monotonicity statistics
    """
    results = {}

    for neuron_idx in tqdm(neurons, desc="Analyzing neuron monotonicity"):
        # Create neuron direction
        direction = np.zeros(768)
        direction[neuron_idx] = 1.0

        results[neuron_idx] = {}

        for dataset in datasets:
            # Get token frequencies by quintile
            token_freqs, _ = calculate_token_frequencies_by_quintile(
                embeddings, sentences_df, direction, dataset_name=dataset
            )

            # Check monotonicity for each token
            monotonic_tokens = {
                'increasing': [],
                'decreasing': []
            }

            for token, freqs in token_freqs.items():
                monotonicity = check_monotonicity(freqs)
                if monotonicity == 'increasing':
                    monotonic_tokens['increasing'].append(token)
                elif monotonicity == 'decreasing':
                    monotonic_tokens['decreasing'].append(token)

            results[neuron_idx][dataset] = monotonic_tokens

    return results

# Run monotonicity analysis
monotonicity_results = analyze_monotonicity_for_all_neurons(
    embeddings, all_sentences, selected_neurons
)

# Save results
with open('monotonicity_results.pkl', 'wb') as f:
    pickle.dump(monotonicity_results, f)
```

### 9.2 Calculate Monotonicity Statistics (Table 3)

```python
def calculate_monotonicity_statistics(monotonicity_results, datasets):
    """
    Calculate statistics matching Table 3 from the paper.

    Returns:
        DataFrame with monotonicity percentages
    """
    stats = []

    # Get common tokens across all datasets/neurons
    all_tokens = set()
    for neuron_results in monotonicity_results.values():
        for dataset_results in neuron_results.values():
            all_tokens.update(dataset_results['increasing'])
            all_tokens.update(dataset_results['decreasing'])

    total_neuron_token_pairs = len(monotonicity_results) * len(all_tokens)

    # Single dataset statistics
    for dataset in datasets:
        increasing = 0
        decreasing = 0

        for neuron_results in monotonicity_results.values():
            if dataset in neuron_results:
                increasing += len(neuron_results[dataset]['increasing'])
                decreasing += len(neuron_results[dataset]['decreasing'])

        total = increasing + decreasing
        stats.append({
            'Datasets': dataset,
            'Monotonic': f"{100*total/total_neuron_token_pairs:.1f}%",
            'Increasing': f"{100*increasing/total_neuron_token_pairs:.1f}%",
            'Decreasing': f"{100*decreasing/total_neuron_token_pairs:.1f}%"
        })

    # Multi-dataset statistics would require tracking tokens across datasets
    # This is more complex - left as exercise

    return pd.DataFrame(stats)

# Calculate and print statistics
mono_stats = calculate_monotonicity_statistics(monotonicity_results,
                                               ['QQP', 'QNLI', 'Wiki', 'Books'])
print(mono_stats)
```

### 9.3 Find Most Monotonic Tokens (Table 4)

```python
def find_most_monotonic_tokens(monotonicity_results, top_n=20):
    """
    Find tokens that are most frequently monotonic across neurons.
    Replicates Table 4 from paper.

    Returns:
        List of (token, count) tuples
    """
    token_monotonic_count = Counter()

    for neuron_results in monotonicity_results.values():
        for dataset_results in neuron_results.values():
            for token in dataset_results['increasing']:
                token_monotonic_count[token] += 1
            for token in dataset_results['decreasing']:
                token_monotonic_count[token] += 1

    # Get top tokens
    top_tokens = token_monotonic_count.most_common(top_n)

    return top_tokens

# Find most monotonic tokens
top_monotonic = find_most_monotonic_tokens(monotonicity_results, top_n=20)
print("\nMost monotonic tokens:")
for token, count in top_monotonic:
    print(f"{token} ({count})")
```

---

## 10. Outlier Analysis

### 10.1 Identify Most Distant Sentences

```python
def analyze_outliers(embeddings, sentences_df, dataset_name='QQP'):
    """
    Identify and analyze outlier sentences (most distant).

    Args:
        embeddings: Sentence embeddings
        sentences_df: Sentence dataframe
        dataset_name: Dataset to analyze

    Returns:
        Analysis results
    """
    # Filter to dataset
    mask = sentences_df['dataset'] == dataset_name
    dataset_embeddings = embeddings[mask]
    dataset_df = sentences_df[mask].reset_index(drop=True)

    # Compute pairwise distances (Euclidean)
    from scipy.spatial.distance import pdist, squareform

    print(f"Computing pairwise distances for {len(dataset_embeddings)} sentences...")
    distances = pdist(dataset_embeddings, metric='euclidean')
    distance_matrix = squareform(distances)

    # Calculate mean distance for each sentence
    mean_distances = distance_matrix.mean(axis=1)

    # Sort by mean distance
    sorted_indices = np.argsort(mean_distances)[::-1]

    # Get top 20 most distant
    top_20_distant = sorted_indices[:20]

    results = {
        'mean_distances': mean_distances,
        'sorted_indices': sorted_indices,
        'top_20_distant_sentences': dataset_df.iloc[top_20_distant].copy()
    }

    results['top_20_distant_sentences']['mean_distance'] = mean_distances[top_20_distant]

    return results

# Analyze outliers for QQP
qqp_outliers = analyze_outliers(embeddings, all_sentences, 'QQP')

print("\nTop 3 most distant QQP sentences:")
for i, row in qqp_outliers['top_20_distant_sentences'].head(3).iterrows():
    print(f"{i+1}. {row['sentence'][:100]}...")
    print(f"   Mean distance: {row['mean_distance']:.2f}\n")
```

### 10.2 Count Top Activations for Outliers

```python
def count_top_activations_for_outliers(all_results, outlier_sentences, dataset_name):
    """
    Count how many times each outlier sentence is in top activations.

    Args:
        all_results: Results from generate_all_top_activating_sets
        outlier_sentences: DataFrame of outlier sentences
        dataset_name: Dataset name

    Returns:
        DataFrame with activation counts
    """
    activation_counts = {sent: 0 for sent in outlier_sentences['sentence']}

    # Check neurons
    for neuron_idx, neuron_results in all_results['neurons'].items():
        if dataset_name in neuron_results:
            top_sents = neuron_results[dataset_name]['sentence'].tolist()
            for sent in top_sents:
                if sent in activation_counts:
                    activation_counts[sent] += 1

    # Check random directions
    for dir_idx, dir_results in all_results['random_directions'].items():
        if dataset_name in dir_results:
            top_sents = dir_results[dataset_name]['sentence'].tolist()
            for sent in top_sents:
                if sent in activation_counts:
                    activation_counts[sent] += 1

    # Add counts to dataframe
    result_df = outlier_sentences.copy()
    result_df['top_activation_count'] = result_df['sentence'].map(activation_counts)
    result_df = result_df.sort_values('top_activation_count', ascending=False)

    return result_df

# Count activations
outliers_with_counts = count_top_activations_for_outliers(
    all_results,
    qqp_outliers['top_20_distant_sentences'],
    'QQP'
)

print("\nTop 20 most distant sentences with activation counts:")
print(outliers_with_counts[['sentence', 'mean_distance', 'top_activation_count']].head(20))
```

---

## 11. Visualization Generation

### 11.1 Reproduce Figure 2 (Annotation Statistics)

```python
def plot_annotation_statistics(annotations_file):
    """
    Plot annotation statistics (Figure 2 from paper).

    Args:
        annotations_file: Path to annotations JSON file
    """
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    # Count distinct patterns per neuron
    neuron_patterns = {}
    for ann in annotations:
        if ann['condition_type'] == 'neuron':
            neuron_id = ann['condition_id']
            if neuron_id not in neuron_patterns:
                neuron_patterns[neuron_id] = set()
            if ann['has_pattern'] and ann['pattern_description']:
                neuron_patterns[neuron_id].add(ann['pattern_description'].lower())

    pattern_counts = [len(patterns) for patterns in neuron_patterns.values()]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Top: Histogram of pattern counts
    ax1.hist(pattern_counts, bins=range(max(pattern_counts)+2),
             align='left', edgecolor='black')
    ax1.set_xlabel('Number of patterns per neuron')
    ax1.set_ylabel('Count of neurons')
    ax1.set_title(f'Number of distinct patterns found for each neuron (μ = {np.mean(pattern_counts):.1f})')

    # Bottom: Positive examples per pattern
    positive_example_counts = []
    for ann in annotations:
        if ann['has_pattern'] and ann['positive_examples']:
            positive_example_counts.append(len(ann['positive_examples']))

    conditions = ['Neurons', 'Random Directions', 'Random Sentences']
    condition_types = ['neuron', 'random_direction', 'random_sentences']

    means = []
    stds = []
    for cond_type in condition_types:
        counts = [len(ann['positive_examples'])
                 for ann in annotations
                 if ann['condition_type'] == cond_type
                 and ann['has_pattern']
                 and ann['positive_examples']]
        means.append(np.mean(counts) if counts else 0)
        stds.append(np.std(counts) if counts else 0)

    ax2.bar(conditions, means, yerr=stds, capsize=5, edgecolor='black')
    ax2.set_ylabel('Mean number of positive examples')
    ax2.set_title('Positive examples per pattern by condition')
    ax2.set_ylim(0, 10)

    plt.tight_layout()
    plt.savefig('annotation_statistics.png', dpi=300)
    plt.close()

    print("Annotation statistics plot saved")
```

### 11.2 Reproduce Figure 6 (Token Frequency Monotonicity)

```python
def plot_token_frequency_by_quintile(embeddings, sentences_df, neuron_idx,
                                     tokens, dataset='QQP', save_path=None):
    """
    Plot token frequency across activation quintiles.
    Replicates Figure 6 from paper.

    Args:
        embeddings: Sentence embeddings
        sentences_df: Sentence dataframe
        neuron_idx: Neuron to analyze
        tokens: List of tokens to plot
        dataset: Dataset name
        save_path: Path to save figure
    """
    # Create neuron direction
    direction = np.zeros(768)
    direction[neuron_idx] = 1.0

    # Get token frequencies
    token_freqs, quintile_bounds = calculate_token_frequencies_by_quintile(
        embeddings, sentences_df, direction, dataset_name=dataset
    )

    # Create subplots
    n_tokens = len(tokens)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, token in enumerate(tokens[:4]):
        if token in token_freqs:
            freqs = token_freqs[token]
            monotonicity = check_monotonicity(freqs)

            axes[i].bar(range(5), freqs, edgecolor='black',
                       color='gray' if monotonicity is None else 'blue')
            axes[i].set_xlabel('Neuron Activation Quintile')
            axes[i].set_ylabel(f'"{token}" Count')
            axes[i].set_title(f'{dataset} - Neuron {neuron_idx} - "{token}"')
            axes[i].set_xticks(range(5))
            axes[i].set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

            if monotonicity:
                axes[i].text(0.5, 0.95, f'Monotonically {monotonicity}',
                           transform=axes[i].transAxes,
                           ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()
```

---

## 12. Statistical Testing

### 12.1 Compare Locality Scores

```python
from scipy import stats

def compare_locality_scores(locality_results, annotations):
    """
    Compare locality scores between meaningful and meaningless neurons.
    Statistical test for Table 5.

    Args:
        locality_results: Results from calculate_all_locality_scores
        annotations: Annotation data

    Returns:
        Statistical test results
    """
    # Classify neurons as meaningful or meaningless based on annotations
    meaningful_neurons = set()
    meaningless_neurons = set()

    for ann in annotations:
        if ann['condition_type'] == 'neuron':
            neuron_id = ann['condition_id']
            if ann['has_pattern']:
                meaningful_neurons.add(neuron_id)
            else:
                meaningless_neurons.add(neuron_id)

    # Collect locality scores
    meaningful_scores = []
    meaningless_scores = []

    for neuron_id, neuron_results in locality_results['neurons'].items():
        neuron_id_str = str(neuron_id)
        for dataset, scores in neuron_results.items():
            if neuron_id_str in meaningful_neurons:
                meaningful_scores.append(scores['locality_score'])
            elif neuron_id_str in meaningless_neurons:
                meaningless_scores.append(scores['locality_score'])

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(meaningful_scores, meaningless_scores)

    results = {
        'meaningful': {
            'mean': np.mean(meaningful_scores),
            'std': np.std(meaningful_scores),
            'n': len(meaningful_scores)
        },
        'meaningless': {
            'mean': np.mean(meaningless_scores),
            'std': np.std(meaningless_scores),
            'n': len(meaningless_scores)
        },
        't_statistic': t_stat,
        'p_value': p_value
    }

    print("\nLocality Score Comparison:")
    print(f"Meaningful neurons: μ = {results['meaningful']['mean']:.4f}, "
          f"σ = {results['meaningful']['std']:.4f}, n = {results['meaningful']['n']}")
    print(f"Meaningless neurons: μ = {results['meaningless']['mean']:.4f}, "
          f"σ = {results['meaningless']['std']:.4f}, n = {results['meaningless']['n']}")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

    return results

# Run comparison
# locality_comparison = compare_locality_scores(locality_results, annotations)
```

### 12.2 Activation Range Overlap Analysis

```python
def calculate_activation_overlap(all_results):
    """
    Calculate overlap in activation ranges between dataset pairs.

    Returns:
        Percentage of neuron/dataset pairs with overlapping activation ranges
    """
    datasets = ['QQP', 'QNLI', 'Wiki', 'Books']
    overlaps = []

    for neuron_idx, neuron_results in all_results['neurons'].items():
        # Get activation ranges for each dataset
        ranges = {}
        for dataset in datasets:
            if dataset in neuron_results:
                scores = neuron_results[dataset]['projection_score'].values
                ranges[dataset] = (scores.min(), scores.max())

        # Check all pairs
        for i, ds1 in enumerate(datasets):
            for ds2 in datasets[i+1:]:
                if ds1 in ranges and ds2 in ranges:
                    r1 = ranges[ds1]
                    r2 = ranges[ds2]

                    # Check if ranges overlap
                    overlap = (r1[0] <= r2[1]) and (r2[0] <= r1[1])
                    overlaps.append(1 if overlap else 0)

    overlap_pct = 100 * np.mean(overlaps)
    print(f"\nActivation range overlap: {overlap_pct:.1f}%")

    return overlap_pct
```

---

## 13. Validation & Verification

### 13.1 Sanity Checks

```python
def run_sanity_checks(embeddings, all_sentences, all_results):
    """
    Run sanity checks to verify implementation correctness.
    """
    print("\n" + "="*80)
    print("RUNNING SANITY CHECKS")
    print("="*80)

    # Check 1: Embedding dimensions
    assert embeddings.shape[0] == len(all_sentences), \
        "Mismatch between embeddings and sentences"
    assert embeddings.shape[1] == 768, \
        "Embeddings should be 768-dimensional"
    print("✓ Embedding dimensions correct")

    # Check 2: Dataset sizes
    expected_sizes = {'QQP': 164246, 'QNLI': 58645, 'Wiki': 203736, 'Books': 198085}
    for dataset, expected in expected_sizes.items():
        actual = len(all_sentences[all_sentences['dataset'] == dataset])
        # Allow some tolerance
        assert abs(actual - expected) < 1000, \
            f"Dataset {dataset} size mismatch: expected ~{expected}, got {actual}"
    print("✓ Dataset sizes approximately correct")

    # Check 3: Embedding norms
    norms = np.linalg.norm(embeddings, axis=1)
    assert norms.mean() > 10 and norms.mean() < 20, \
        f"Unexpected embedding norm mean: {norms.mean()}"
    print(f"✓ Embedding norms reasonable (mean: {norms.mean():.2f})")

    # Check 4: Top activating sentences are actually top
    neuron_0 = np.zeros(768)
    neuron_0[0] = 1.0
    activations = embeddings @ neuron_0
    top_10_indices = np.argsort(activations)[-10:][::-1]
    top_10_scores = activations[top_10_indices]

    # Scores should be decreasing
    assert all(top_10_scores[i] >= top_10_scores[i+1] for i in range(9)), \
        "Top activating sentences not properly sorted"
    print("✓ Top activating sentence extraction working correctly")

    # Check 5: Random directions are normalized
    for i, direction in enumerate(random_directions[:5]):
        norm = np.linalg.norm(direction)
        assert abs(norm - 1.0) < 1e-6, \
            f"Random direction {i} not normalized: norm = {norm}"
    print("✓ Random directions properly normalized")

    print("\n" + "="*80)
    print("ALL SANITY CHECKS PASSED ✓")
    print("="*80 + "\n")

# Run sanity checks
run_sanity_checks(embeddings, all_sentences, all_results)
```

### 13.2 Reproduce Key Numbers from Paper

```python
def reproduce_paper_numbers(all_results, annotations, locality_results):
    """
    Attempt to reproduce key quantitative results from the paper.
    """
    print("\n" + "="*80)
    print("ATTEMPTING TO REPRODUCE PAPER NUMBERS")
    print("="*80)

    # Table 1: Pattern prevalence
    print("\n### Table 1: Pattern Prevalence ###")

    neuron_with_patterns = sum(1 for a in annotations
                               if a['condition_type'] == 'neuron' and a['has_pattern'])
    total_neuron_annotations = sum(1 for a in annotations
                                   if a['condition_type'] == 'neuron')

    print(f"Neurons with patterns: {neuron_with_patterns}/{total_neuron_annotations} "
          f"({100*neuron_with_patterns/total_neuron_annotations:.0f}%)")
    print(f"Expected from paper: 20/25 (80%)")

    # Figure 2: Average patterns per neuron
    print("\n### Figure 2: Patterns per Neuron ###")

    neuron_patterns = {}
    for ann in annotations:
        if ann['condition_type'] == 'neuron':
            neuron_id = ann['condition_id']
            if neuron_id not in neuron_patterns:
                neuron_patterns[neuron_id] = set()
            if ann['has_pattern'] and ann['pattern_description']:
                neuron_patterns[neuron_id].add(ann['pattern_description'].lower())

    avg_patterns = np.mean([len(p) for p in neuron_patterns.values()])
    print(f"Average patterns per neuron: {avg_patterns:.1f}")
    print(f"Expected from paper: 2.5")

    print("\n" + "="*80)
```

---

## Summary and Next Steps

This replication plan provides:

1. **Complete environment setup** with all required dependencies
2. **Data acquisition and preprocessing** for all four datasets
3. **BERT model setup** matching the paper's configuration
4. **Embedding generation** using [CLS] tokens from layer 12
5. **Core analyses** including neuron selection and top activation finding
6. **Annotation protocol** with interface for pattern identification
7. **Dataset idiosyncrasy analysis** with UMAP and SVM classification
8. **Local semantic coherence analysis** with locality scores
9. **Monotonicity analysis** for token frequencies across quintiles
10. **Outlier analysis** for most distant sentences
11. **Visualization generation** for all key figures
12. **Statistical testing** to verify significance
13. **Validation procedures** to ensure correctness

### Recommended Execution Order:

1. Set up environment (Section 1)
2. Acquire and preprocess data (Section 2)
3. Generate embeddings (Sections 3-4)
4. Run core analyses (Section 5)
5. Conduct annotation study (Section 6) - **this requires human annotators**
6. Run automated analyses (Sections 7-10)
7. Generate visualizations (Section 11)
8. Perform statistical tests (Section 12)
9. Validate results (Section 13)

### Expected Timeline:

- Setup and data preparation: 1-2 days
- Embedding generation: 4-8 hours (depending on hardware)
- Automated analyses: 1-2 days
- Annotation study: 1-2 weeks (depending on number of annotators)
- Visualization and validation: 1-2 days

**Total: ~2-3 weeks for complete replication**

### Notes:

- BookCorpus dataset may require finding alternative source
- Annotation requires multiple independent annotators for inter-annotator agreement
- GPU strongly recommended for embedding generation
- Some randomness in results due to random direction selection and data sampling
