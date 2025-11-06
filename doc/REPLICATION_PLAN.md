# Replication Plan: An Interpretability Illusion for BERT

This document provides a step-by-step plan to replicate the experiments from "An Interpretability Illusion for BERT" (Caciularu et al., 2021).

## Overview

The replication involves three main experimental components:
1. **Adversarial Attention Manipulation** - Modify attention patterns while preserving predictions
2. **Attention-Gradient Correlation Analysis** - Compare attention weights with gradient-based importance
3. **Erasure Studies** - Measure prediction changes when removing tokens

## Prerequisites

### Environment Setup

#### 1. Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install torch>=1.10.0
pip install transformers>=4.15.0
pip install datasets>=1.18.0
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install pandas>=1.3.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install tqdm>=4.62.0
pip install scikit-learn>=1.0.0
```

#### 2. Hardware Requirements
- **Minimum**: GPU with 8GB VRAM (for BERT-base)
- **Recommended**: GPU with 16GB+ VRAM (for RoBERTa-large)
- **Alternative**: Can use CPU but will be significantly slower
- **Storage**: ~10GB for models and datasets

#### 3. Dataset Access
All datasets available through HuggingFace `datasets` library:
- MNLI (Multi-Genre Natural Language Inference)
- SST-2 (Stanford Sentiment Treebank)
- QQP (Quora Question Pairs)
- FEVER (Fact Extraction and VERification)

### Model Setup

#### 4. Pre-trained Models
Download from HuggingFace:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# BERT-base
bert_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# RoBERTa-base
roberta_base = AutoModelForSequenceClassification.from_pretrained('roberta-base')
roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# RoBERTa-large
roberta_large = AutoModelForSequenceClassification.from_pretrained('roberta-large')
roberta_large_tokenizer = AutoTokenizer.from_pretrained('roberta-large')
```

#### 5. Fine-tuned Models
Option A: Fine-tune models yourself (recommended for full replication)
Option B: Use pre-fine-tuned models from HuggingFace model hub

For Option A, fine-tune on each task:
```python
# Example for MNLI
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

dataset = load_dataset('glue', 'mnli')

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
)

# Similar for SST-2, QQP, FEVER
```

## Phase 1: Baseline Analysis

### Step 1: Extract Baseline Attention Patterns

**Goal**: Understand normal attention behavior on test sets

**Implementation**:
```python
def extract_attention_patterns(model, tokenizer, dataset, num_samples=1000):
    """
    Extract attention weights from model on dataset.
    
    Returns:
        attention_patterns: dict with keys [layer][head][sample_id]
        predictions: model predictions for each sample
        labels: ground truth labels
    """
    model.eval()
    attention_patterns = {}
    predictions = []
    labels = []
    
    for i, example in enumerate(dataset[:num_samples]):
        inputs = tokenizer(example['text'], return_tensors='pt', 
                          padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            
        # outputs.attentions: tuple of (batch, heads, seq_len, seq_len)
        for layer_idx, layer_attn in enumerate(outputs.attentions):
            if layer_idx not in attention_patterns:
                attention_patterns[layer_idx] = {}
            for head_idx in range(layer_attn.shape[1]):
                if head_idx not in attention_patterns[layer_idx]:
                    attention_patterns[layer_idx][head_idx] = []
                attention_patterns[layer_idx][head_idx].append(
                    layer_attn[0, head_idx].cpu().numpy()
                )
        
        predictions.append(outputs.logits.argmax(-1).item())
        labels.append(example['label'])
    
    return attention_patterns, predictions, labels
```

**Tasks**:
- [ ] Extract attention for MNLI validation set (RoBERTa-base, RoBERTa-large)
- [ ] Extract attention for SST-2 validation set (BERT-base, RoBERTa-base)
- [ ] Extract attention for QQP validation set (RoBERTa-base)
- [ ] Extract attention for FEVER validation set (RoBERTa-base)
- [ ] Compute statistics: mean, std, entropy per head/layer
- [ ] Visualize attention distributions

**Expected Output**: 
- Attention patterns saved as NumPy arrays
- Summary statistics per model/task
- Visualization of typical attention patterns

### Step 2: Compute Gradient-Based Importance

**Goal**: Calculate "ground truth" importance scores using gradients

**Implementation**:
```python
def compute_input_gradients(model, tokenizer, text, label):
    """
    Compute Input × Gradient importance scores.
    
    Returns:
        importance_scores: array of shape (seq_len,)
    """
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    # Get embeddings
    embeddings = model.get_input_embeddings()(inputs['input_ids'])
    embeddings.requires_grad = True
    
    # Forward pass
    outputs = model(inputs_embeds=embeddings, attention_mask=inputs['attention_mask'])
    loss = F.cross_entropy(outputs.logits, torch.tensor([label]))
    
    # Backward pass
    loss.backward()
    
    # Compute importance: ||embedding × gradient||
    importance = (embeddings * embeddings.grad).abs().sum(dim=-1).squeeze()
    
    return importance.detach().cpu().numpy()

def compute_integrated_gradients(model, tokenizer, text, label, steps=50):
    """
    Compute Integrated Gradients importance scores.
    
    Returns:
        importance_scores: array of shape (seq_len,)
    """
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    # Baseline: zero embeddings
    embeddings = model.get_input_embeddings()(inputs['input_ids'])
    baseline = torch.zeros_like(embeddings)
    
    # Accumulate gradients along path
    integrated_grads = torch.zeros_like(embeddings)
    
    for alpha in torch.linspace(0, 1, steps):
        interp_embeds = baseline + alpha * (embeddings - baseline)
        interp_embeds.requires_grad = True
        
        outputs = model(inputs_embeds=interp_embeds, 
                       attention_mask=inputs['attention_mask'])
        loss = F.cross_entropy(outputs.logits, torch.tensor([label]))
        loss.backward()
        
        integrated_grads += interp_embeds.grad / steps
    
    # Final importance
    importance = (embeddings * integrated_grads).abs().sum(dim=-1).squeeze()
    
    return importance.detach().cpu().numpy()
```

**Tasks**:
- [ ] Compute Input×Gradient for all validation samples
- [ ] Compute Integrated Gradients for subset (computationally expensive)
- [ ] Save gradient importance scores
- [ ] Compare I×G vs. IG correlation (sanity check)

**Expected Output**:
- Gradient importance scores for each sample
- Correlation analysis between I×G and IG

### Step 3: Erasure Analysis

**Goal**: Measure importance via counterfactual token removal

**Implementation**:
```python
def erasure_importance(model, tokenizer, text, label):
    """
    Compute importance by removing each token and measuring prediction change.
    
    Returns:
        importance_scores: array of shape (seq_len,)
    """
    model.eval()
    
    # Baseline prediction
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        baseline_logits = model(**inputs).logits
        baseline_pred = baseline_logits.argmax(-1).item()
        baseline_conf = F.softmax(baseline_logits, dim=-1)[0, label].item()
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    importance = []
    
    # Try removing each token
    for i in range(len(tokens)):
        # Skip special tokens
        if tokens[i] in [tokenizer.cls_token, tokenizer.sep_token, 
                        tokenizer.pad_token]:
            importance.append(0.0)
            continue
        
        # Create text with token i removed
        masked_tokens = tokens[:i] + tokens[i+1:]
        masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
        
        masked_inputs = tokenizer(masked_text, return_tensors='pt', 
                                  padding=True, truncation=True)
        with torch.no_grad():
            masked_logits = model(**masked_inputs).logits
            masked_conf = F.softmax(masked_logits, dim=-1)[0, label].item()
        
        # Importance = change in confidence
        importance.append(abs(baseline_conf - masked_conf))
    
    return np.array(importance)
```

**Tasks**:
- [ ] Compute erasure importance for validation samples (sample subset due to compute)
- [ ] Save erasure importance scores
- [ ] Analyze which tokens are most important via erasure

**Expected Output**:
- Erasure importance scores
- Analysis of special token patterns

## Phase 2: Correlation Analysis

### Step 4: Compare Attention vs. Gradients

**Goal**: Quantify correlation between attention and gradient-based importance

**Implementation**:
```python
from scipy.stats import kendalltau, pearsonr

def compute_correlations(attention_weights, gradient_importance):
    """
    Compute correlation between attention and gradient importance.
    
    Args:
        attention_weights: shape (seq_len, seq_len) - focus on [:, i] for token i
        gradient_importance: shape (seq_len,)
    
    Returns:
        kendall_tau: rank correlation
        pearson_r: linear correlation
    """
    # Aggregate attention: how much each token is attended TO
    attn_importance = attention_weights.mean(axis=0)  # Average over all source tokens
    
    # Ensure same length (exclude special tokens if needed)
    min_len = min(len(attn_importance), len(gradient_importance))
    attn_importance = attn_importance[:min_len]
    gradient_importance = gradient_importance[:min_len]
    
    kendall, _ = kendalltau(attn_importance, gradient_importance)
    pearson, _ = pearsonr(attn_importance, gradient_importance)
    
    return kendall, pearson

def analyze_all_correlations(attention_patterns, gradient_scores):
    """
    Compute correlations across all samples, layers, heads.
    """
    results = {
        'layer': [],
        'head': [],
        'sample_id': [],
        'kendall_tau': [],
        'pearson_r': []
    }
    
    for sample_id in range(len(gradient_scores)):
        for layer in attention_patterns.keys():
            for head in attention_patterns[layer].keys():
                attn = attention_patterns[layer][head][sample_id]
                grad = gradient_scores[sample_id]
                
                kendall, pearson = compute_correlations(attn, grad)
                
                results['layer'].append(layer)
                results['head'].append(head)
                results['sample_id'].append(sample_id)
                results['kendall_tau'].append(kendall)
                results['pearson_r'].append(pearson)
    
    return pd.DataFrame(results)
```

**Tasks**:
- [ ] Compute correlations for all (layer, head, sample) combinations
- [ ] Aggregate statistics: mean, median, std per layer/head
- [ ] Identify heads with highest/lowest correlation
- [ ] Create visualizations: heatmaps, distributions
- [ ] Test statistical significance

**Expected Results**:
- Mean Kendall's tau: 0.15-0.35 (as reported in paper)
- Mean Pearson's r: 0.20-0.45
- Higher correlation in later layers
- Most heads show poor correlation

**Deliverables**:
- CSV with correlation scores
- Heatmap of correlation by layer/head
- Distribution plots
- Statistical test results

### Step 5: Compare Attention vs. Erasure

**Goal**: Measure agreement between attention and erasure importance

**Implementation**:
```python
def top_k_agreement(attention_scores, erasure_scores, k=5):
    """
    Compute agreement in top-k important tokens.
    
    Returns:
        agreement: fraction of overlap in top-k tokens
    """
    attn_top_k = set(np.argsort(attention_scores)[-k:])
    erasure_top_k = set(np.argsort(erasure_scores)[-k:])
    
    agreement = len(attn_top_k & erasure_top_k) / k
    return agreement
```

**Tasks**:
- [ ] Compute top-k agreement for k=3, k=5, k=10
- [ ] Compare to random baseline
- [ ] Analyze disagreement patterns

**Expected Results**:
- Top-3 agreement: ~30-40%
- Top-5 agreement: ~35-45%
- Better than random but far from perfect

## Phase 3: Adversarial Attention Manipulation

This is the core contribution of the paper.

### Step 6: Implement Adversarial Framework

**Goal**: Modify attention patterns while preserving predictions

**Implementation**:

```python
class AdversarialAttentionModel(nn.Module):
    """
    Wrapper around pre-trained model that allows attention manipulation.
    """
    def __init__(self, base_model, layer_idx, head_idx):
        super().__init__()
        self.base_model = base_model
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        
        # Get original Q and K projection matrices
        attention_layer = base_model.encoder.layer[layer_idx].attention.self
        self.W_Q = nn.Parameter(attention_layer.query.weight.clone())
        self.W_K = nn.Parameter(attention_layer.key.weight.clone())
        
        # Freeze all other parameters
        for param in base_model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get embeddings
        embeddings = self.base_model.embeddings(input_ids)
        
        # Compute new attention for target layer/head
        Q = F.linear(embeddings, self.W_Q)
        K = F.linear(embeddings, self.W_K)
        
        # Replace attention in the model
        # (This requires monkey-patching the model's attention computation)
        # Implementation details depend on model architecture
        
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        
        return outputs

def adversarial_attention_loss(model, inputs, labels, target_attention, alpha=0.5):
    """
    Loss function for adversarial attention training.
    
    Args:
        model: AdversarialAttentionModel
        inputs: input tensors
        labels: ground truth labels
        target_attention: desired attention distribution
        alpha: balance between KL divergence and task loss
    
    Returns:
        total_loss: weighted combination of KL and task loss
    """
    outputs = model(**inputs, labels=labels)
    
    # Task loss (preserve predictions)
    task_loss = outputs.loss
    
    # KL divergence loss (modify attention)
    current_attention = outputs.attentions[model.layer_idx][:, model.head_idx]
    kl_loss = F.kl_div(
        F.log_softmax(current_attention, dim=-1),
        target_attention,
        reduction='batchmean'
    )
    
    total_loss = alpha * kl_loss + (1 - alpha) * task_loss
    
    return total_loss, kl_loss.item(), task_loss.item()
```

**Alternative Simpler Approach**:
Instead of full adversarial training, use attention manipulation during inference:

```python
def manipulate_attention_inference(model, inputs, target_attention_dist, layer_idx, head_idx):
    """
    Simpler approach: directly replace attention during forward pass.
    Requires model surgery / hooks.
    """
    # Register hook to replace attention
    def attention_hook(module, input, output):
        # output is (batch, heads, seq, seq)
        modified_output = output.clone()
        modified_output[:, head_idx] = target_attention_dist
        return modified_output
    
    handle = model.encoder.layer[layer_idx].attention.register_forward_hook(attention_hook)
    
    outputs = model(**inputs)
    
    handle.remove()
    
    return outputs
```

**Tasks**:
- [ ] Implement adversarial attention model wrapper
- [ ] Implement loss function with KL divergence + task loss
- [ ] Set up training loop with Adam optimizer

### Step 7: Generate Target Attention Distributions

**Goal**: Create different target attention patterns to test

**Implementation**:
```python
def generate_target_attention(current_attention, target_type, seq_len):
    """
    Generate target attention distribution.
    
    Args:
        current_attention: current attention weights (seq_len, seq_len)
        target_type: 'uniform', 'reversed', 'random', 'shuffled'
        seq_len: sequence length
    
    Returns:
        target_attention: new attention distribution
    """
    if target_type == 'uniform':
        # Equal attention to all tokens
        target = torch.ones(seq_len, seq_len) / seq_len
    
    elif target_type == 'reversed':
        # Flip attention: high -> low, low -> high
        max_attn = current_attention.max()
        target = max_attn - current_attention
        target = F.softmax(target, dim=-1)
    
    elif target_type == 'random':
        # Random attention distribution
        target = torch.rand(seq_len, seq_len)
        target = F.softmax(target, dim=-1)
    
    elif target_type == 'shuffled':
        # Shuffle attention weights
        target = current_attention.clone()
        for i in range(seq_len):
            perm = torch.randperm(seq_len)
            target[i] = target[i, perm]
    
    return target
```

**Tasks**:
- [ ] Implement all target distribution types
- [ ] Verify distributions are valid (sum to 1, non-negative)
- [ ] Visualize example targets

### Step 8: Run Adversarial Experiments

**Goal**: Manipulate attention and measure impact on predictions

**Training Protocol**:
```python
def run_adversarial_experiment(model, dataset, layer_idx, head_idx, 
                               target_type, alpha=0.5, num_steps=100):
    """
    Run one adversarial manipulation experiment.
    
    Returns:
        results: dict with metrics before/after manipulation
    """
    adv_model = AdversarialAttentionModel(model, layer_idx, head_idx)
    optimizer = torch.optim.Adam([adv_model.W_Q, adv_model.W_K], lr=1e-4)
    
    # Collect metrics
    metrics = {
        'kl_divergence': [],
        'task_loss': [],
        'accuracy': []
    }
    
    for step in range(num_steps):
        batch = next(iter(dataset))
        
        # Generate target attention
        current_attn = get_current_attention(model, batch, layer_idx, head_idx)
        target_attn = generate_target_attention(current_attn, target_type)
        
        # Optimization step
        optimizer.zero_grad()
        loss, kl, task = adversarial_attention_loss(
            adv_model, batch, batch['labels'], target_attn, alpha
        )
        loss.backward()
        optimizer.step()
        
        # Track metrics
        metrics['kl_divergence'].append(kl)
        metrics['task_loss'].append(task)
        
        if step % 10 == 0:
            acc = evaluate_accuracy(adv_model, dataset)
            metrics['accuracy'].append(acc)
    
    return metrics, adv_model
```

**Experimental Matrix**:

For each combination, run experiment:
- **Models**: BERT-base, RoBERTa-base, RoBERTa-large
- **Tasks**: MNLI, SST-2, QQP, FEVER
- **Layers**: 6, 11 (middle and late layers)
- **Heads**: 0-11 (all heads, or subset)
- **Target Types**: uniform, reversed, random

**Tasks**:
- [ ] Run experiments for all combinations (prioritize key results first)
- [ ] Monitor: KL divergence, task loss, accuracy, F1
- [ ] Early stopping when KL < threshold or accuracy drops > 2%
- [ ] Save manipulated model checkpoints

**Expected Results** (per paper):
- Can achieve KL divergence reduction of 50%+ 
- While maintaining accuracy within 1-2%
- Demonstrates attention can be changed without affecting predictions

### Step 9: Evaluate Manipulated Models

**Goal**: Thoroughly test that predictions haven't changed

**Evaluation Protocol**:
```python
def evaluate_manipulation_success(original_model, manipulated_model, test_set):
    """
    Compare original vs. manipulated model predictions.
    
    Returns:
        metrics: dict with evaluation results
    """
    metrics = {
        'accuracy_original': 0.0,
        'accuracy_manipulated': 0.0,
        'accuracy_difference': 0.0,
        'prediction_agreement': 0.0,
        'confidence_correlation': 0.0,
        'attention_kl_divergence': 0.0
    }
    
    original_preds = []
    manipulated_preds = []
    original_confs = []
    manipulated_confs = []
    
    for example in test_set:
        # Original model
        with torch.no_grad():
            orig_out = original_model(**example)
            orig_pred = orig_out.logits.argmax(-1).item()
            orig_conf = F.softmax(orig_out.logits, dim=-1).max().item()
        
        # Manipulated model
        with torch.no_grad():
            manip_out = manipulated_model(**example)
            manip_pred = manip_out.logits.argmax(-1).item()
            manip_conf = F.softmax(manip_out.logits, dim=-1).max().item()
        
        original_preds.append(orig_pred)
        manipulated_preds.append(manip_pred)
        original_confs.append(orig_conf)
        manipulated_confs.append(manip_conf)
    
    # Compute metrics
    metrics['accuracy_original'] = accuracy_score(test_set.labels, original_preds)
    metrics['accuracy_manipulated'] = accuracy_score(test_set.labels, manipulated_preds)
    metrics['accuracy_difference'] = abs(
        metrics['accuracy_original'] - metrics['accuracy_manipulated']
    )
    metrics['prediction_agreement'] = accuracy_score(original_preds, manipulated_preds)
    metrics['confidence_correlation'] = pearsonr(original_confs, manipulated_confs)[0]
    
    return metrics
```

**Tasks**:
- [ ] Evaluate all manipulated models on test sets
- [ ] Create comparison tables: original vs. manipulated
- [ ] Visualize attention changes
- [ ] Test on held-out examples
- [ ] Statistical significance tests

**Expected Metrics**:
- Prediction agreement: >98%
- Accuracy drop: <2%
- Attention KL divergence: 0.5-2.0 (substantial change)

## Phase 4: Analysis and Visualization

### Step 10: Create Visualizations

**Attention Heatmaps**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_comparison(original_attn, manipulated_attn, tokens):
    """
    Side-by-side attention heatmaps.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(original_attn, xticklabels=tokens, yticklabels=tokens,
                cmap='Blues', ax=ax1, cbar=True, vmin=0, vmax=1)
    ax1.set_title('Original Attention')
    
    sns.heatmap(manipulated_attn, xticklabels=tokens, yticklabels=tokens,
                cmap='Reds', ax=ax2, cbar=True, vmin=0, vmax=1)
    ax2.set_title('Manipulated Attention')
    
    plt.tight_layout()
    return fig
```

**Correlation Plots**:
```python
def plot_correlation_distributions(correlation_df):
    """
    Plot distribution of correlations across layers/heads.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Overall distribution
    sns.histplot(correlation_df['kendall_tau'], bins=30, ax=axes[0, 0])
    axes[0, 0].set_title('Kendall Tau Distribution')
    axes[0, 0].axvline(0, color='red', linestyle='--')
    
    # By layer
    sns.boxplot(data=correlation_df, x='layer', y='kendall_tau', ax=axes[0, 1])
    axes[0, 1].set_title('Kendall Tau by Layer')
    
    # Heatmap: layer × head
    pivot = correlation_df.pivot_table(
        values='kendall_tau', index='layer', columns='head', aggfunc='mean'
    )
    sns.heatmap(pivot, cmap='RdYlGn', center=0, ax=axes[1, 0])
    axes[1, 0].set_title('Mean Kendall Tau: Layer × Head')
    
    # Scatter: Kendall vs. Pearson
    sns.scatterplot(data=correlation_df, x='kendall_tau', y='pearson_r',
                   alpha=0.3, ax=axes[1, 1])
    axes[1, 1].set_title('Kendall vs. Pearson Correlation')
    
    plt.tight_layout()
    return fig
```

**Tasks**:
- [ ] Create attention heatmap visualizations
- [ ] Plot correlation distributions
- [ ] Visualize top-k agreement
- [ ] Create summary figures for paper/presentation
- [ ] Generate example case studies

### Step 11: Statistical Analysis

**Hypothesis Testing**:
```python
from scipy.stats import ttest_rel, wilcoxon

def statistical_tests(attention_importance, gradient_importance):
    """
    Test if attention and gradients are significantly different.
    """
    # Paired t-test
    t_stat, t_pval = ttest_rel(attention_importance, gradient_importance)
    
    # Wilcoxon signed-rank test (non-parametric)
    w_stat, w_pval = wilcoxon(attention_importance, gradient_importance)
    
    return {
        't_statistic': t_stat,
        't_pvalue': t_pval,
        'wilcoxon_statistic': w_stat,
        'wilcoxon_pvalue': w_pval
    }
```

**Tasks**:
- [ ] Test significance of correlation differences
- [ ] Test significance of accuracy changes
- [ ] Compute confidence intervals
- [ ] Report effect sizes

### Step 12: Compile Results

**Create Result Tables**:
```python
def create_results_table(all_experiments):
    """
    Compile all experimental results into summary tables.
    """
    results = []
    
    for exp in all_experiments:
        results.append({
            'Model': exp['model_name'],
            'Task': exp['task_name'],
            'Layer': exp['layer_idx'],
            'Head': exp['head_idx'],
            'Target': exp['target_type'],
            'Attention KL': exp['final_kl'],
            'Accuracy Drop': exp['accuracy_drop'],
            'Kendall Tau': exp['mean_kendall_tau'],
            'Pearson r': exp['mean_pearson_r'],
        })
    
    return pd.DataFrame(results)
```

**Tasks**:
- [ ] Create summary tables for all experiments
- [ ] Compute aggregate statistics
- [ ] Compare results to paper's reported values
- [ ] Document any discrepancies

## Phase 5: Validation and Reproducibility

### Step 13: Verify Replication Accuracy

**Checklist**:
- [ ] Attention-gradient correlation matches paper (±0.05)
- [ ] Adversarial manipulation success rate matches paper
- [ ] Accuracy preservation matches paper (within 1%)
- [ ] Top-k agreement matches paper (±5%)

**If results don't match**:
1. Check hyperparameters (learning rate, alpha, num_steps)
2. Verify dataset splits (train/val/test)
3. Check model versions (exact checkpoint used)
4. Review implementation details
5. Consider reaching out to authors

### Step 14: Document Methodology

**Create Documentation**:
- [ ] README with setup instructions
- [ ] Requirements.txt with exact versions
- [ ] Scripts for each experiment
- [ ] Jupyter notebooks with analysis
- [ ] Instructions for reproducing each figure

**Code Organization**:
```
bert-interpretability-illusion/
├── data/
│   └── download_datasets.py
├── models/
│   ├── load_models.py
│   └── adversarial_attention.py
├── experiments/
│   ├── baseline_analysis.py
│   ├── correlation_study.py
│   ├── adversarial_experiments.py
│   └── erasure_analysis.py
├── analysis/
│   ├── compute_statistics.py
│   └── generate_plots.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_attention_analysis.ipynb
│   ├── 03_correlation_study.ipynb
│   └── 04_adversarial_results.ipynb
├── results/
│   ├── figures/
│   ├── tables/
│   └── checkpoints/
└── README.md
```

### Step 15: Create Reproducibility Package

**Package Contents**:
- [ ] All code with comments
- [ ] Exact package versions
- [ ] Random seeds for reproducibility
- [ ] Instructions for each experiment
- [ ] Expected runtimes
- [ ] Hardware requirements
- [ ] Troubleshooting guide

**Reproducibility Checklist**:
```python
# Set seeds
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Timeline Estimate

**Phase 1: Baseline Analysis** (1-2 weeks)
- Setup: 1-2 days
- Extract attention: 2-3 days
- Compute gradients: 2-3 days
- Erasure analysis: 2-3 days

**Phase 2: Correlation Analysis** (1 week)
- Implement metrics: 1-2 days
- Run analysis: 2-3 days
- Create visualizations: 1-2 days

**Phase 3: Adversarial Manipulation** (2-3 weeks)
- Implement framework: 3-5 days
- Run experiments: 5-7 days (compute-intensive)
- Evaluate results: 2-3 days

**Phase 4: Analysis and Visualization** (1 week)
- Create all plots: 2-3 days
- Statistical analysis: 1-2 days
- Compile results: 1-2 days

**Phase 5: Validation** (1 week)
- Verify results: 2-3 days
- Document methodology: 2-3 days
- Create package: 1-2 days

**Total: 6-9 weeks**

## Success Criteria

The replication is successful if:

1. **Correlation Results Match**
   - Kendall's tau between attention and gradients: 0.15-0.35 ✓
   - Pearson's r: 0.20-0.45 ✓
   - Variation by layer/head similar to paper ✓

2. **Adversarial Manipulation Works**
   - Can change attention distribution significantly (KL > 0.5) ✓
   - While preserving accuracy (drop < 2%) ✓
   - Across multiple tasks and models ✓

3. **Erasure Results Agree**
   - Top-k agreement ~30-45% ✓
   - Better than random but imperfect ✓

4. **Statistical Significance**
   - All main results statistically significant ✓
   - Effect sizes reported ✓

## Troubleshooting Common Issues

### Issue 1: Low Correlation
**Symptoms**: Kendall's tau < 0.1
**Solutions**:
- Check attention aggregation (average vs. max)
- Verify gradient computation is correct
- Try different layers/heads
- Ensure using validation set (not training set)

### Issue 2: Adversarial Training Fails
**Symptoms**: Can't change attention or accuracy drops too much
**Solutions**:
- Adjust alpha parameter (try 0.3, 0.5, 0.7)
- Change learning rate (try 1e-5 to 1e-3)
- Increase num_steps
- Initialize from original Q/K matrices
- Check gradient flow

### Issue 3: Out of Memory
**Solutions**:
- Reduce batch size
- Use gradient accumulation
- Process samples one at a time
- Use smaller model (BERT-base instead of RoBERTa-large)
- Use mixed precision training (fp16)

### Issue 4: Different Results
**Solutions**:
- Check model checkpoint (base vs. large)
- Verify dataset version
- Check random seed
- Review preprocessing steps
- Contact authors for clarification

## References

- Original Paper: https://arxiv.org/abs/2104.07143
- HuggingFace Transformers: https://huggingface.co/docs/transformers/
- GLUE Benchmark: https://gluebenchmark.com/
- Integrated Gradients: https://arxiv.org/abs/1703.01365

## Additional Resources

- Attention visualization tools: BertViz, exBERT
- Interpretability libraries: Captum, InterpretML
- Related papers: "Attention is not Explanation", "Attention is not not Explanation"
