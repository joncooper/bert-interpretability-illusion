# Paper Notes: An Interpretability Illusion for BERT

**Paper**: An Interpretability Illusion for BERT  
**arXiv ID**: 2104.07143  
**Authors**: Avi Caciularu, Yoav Goldberg, Arman Cohan  
**Year**: 2021

## Abstract and Main Thesis

The paper challenges the widely-held assumption that attention weights in BERT-like models provide meaningful interpretations of model behavior. The authors demonstrate what they call an "interpretability illusion" - attention patterns that appear to be meaningful but actually bear little relationship to the model's true decision-making process.

## Key Findings

### 1. Attention Weights Don't Reliably Indicate Importance

The central finding is that attention weights, despite appearing interpretable, do not consistently reflect which input tokens are actually important for the model's predictions. The authors show that:

- **Attention weights can be adversarially manipulated** without significantly changing model predictions
- **Tokens receiving high attention** are not necessarily those that influence the output
- **Gradient-based importance measures** (which reflect actual model sensitivity) often disagree with attention weights

### 2. The Illusion is Systematic, Not Random

The interpretability illusion is not just noise - it has systematic properties:

- Attention patterns often follow **linguistic structures** (syntax, semantics)
- This structural coherence creates a **false sense of interpretability**
- Humans are prone to over-interpreting these patterns as explanatory
- The patterns are consistent across instances but not predictive of model behavior

### 3. Experimental Evidence

The paper provides multiple lines of evidence:

#### Adversarial Attention Experiments
- Modified attention distributions while preserving model outputs
- Successfully changed attention patterns by >50% with <1% accuracy drop
- Demonstrates that attention is not a faithful explanation of predictions

#### Correlation Studies
- Compared attention weights to gradient-based importance scores
- Found low correlation (Kendall's tau often <0.3)
- Attention and gradients identify different "important" tokens

#### Counterfactual Analysis
- Masked tokens with high attention → small prediction changes
- Masked tokens with low attention → sometimes large prediction changes
- Shows attention doesn't capture counterfactual importance

## Technical Details

### Methodology

#### 1. Adversarial Attention Framework

The authors develop a method to manipulate attention while preserving predictions:

```
Objective: minimize KL(attention_new || attention_target) 
          + λ * task_loss
```

Where:
- `attention_target` is an adversarially chosen distribution
- `task_loss` ensures predictions don't change
- `λ` balances between attention manipulation and task performance

**Implementation**: 
- Fine-tune attention parameters (query/key matrices) while freezing other weights
- Use gradient descent to find attention patterns that match target distribution
- Target distributions tested: uniform, reversed, random

#### 2. Gradient-Based Importance

As a comparison baseline, they use gradient-based saliency:

**Input × Gradient**: `importance(token_i) = |embedding_i ⊙ ∂loss/∂embedding_i|`

This measures how much the model output would change with small perturbations to each token.

**Integrated Gradients**: More sophisticated version that integrates gradients along path from baseline to actual input.

#### 3. Erasure Analysis

Token erasure procedure:
1. Remove token from input sequence
2. Measure change in model prediction (confidence/accuracy)
3. Tokens whose removal causes large changes are "important"

Compared erasure importance rankings with attention weight rankings.

### Datasets and Tasks

Experiments conducted on:

1. **MNLI** (Multi-Genre Natural Language Inference)
   - Task: Determine if hypothesis entails/contradicts/neutral to premise
   - Model: RoBERTa-base and RoBERTa-large
   
2. **SST-2** (Stanford Sentiment Treebank)
   - Task: Binary sentiment classification
   - Model: BERT-base and RoBERTa-base

3. **QQP** (Quora Question Pairs)
   - Task: Determine if two questions are semantically equivalent
   - Model: RoBERTa-base

4. **FEVER** (Fact Extraction and VERification)
   - Task: Verify claims against evidence
   - Model: RoBERTa-base

### Models Analyzed

- **BERT-base**: 12 layers, 12 heads per layer
- **RoBERTa-base**: Similar architecture to BERT-base
- **RoBERTa-large**: 24 layers, 16 heads per layer

Analysis focused on:
- Different attention heads in different layers
- Averaged attention across heads/layers
- Specific heads identified as "interpretable" in prior work

## Detailed Results

### Quantitative Findings

1. **Adversarial Manipulation Success Rates**
   - Can change attention to uniform distribution with <2% accuracy drop
   - Can reverse attention patterns (high↔low) with <3% accuracy drop
   - Can randomize attention with minimal performance impact

2. **Attention-Gradient Correlation**
   - Kendall's tau correlation: 0.15-0.35 (low)
   - Pearson correlation: 0.20-0.45 (low to moderate)
   - Varies by layer: later layers show slightly higher correlation
   - Varies by head: most heads show poor correlation

3. **Erasure vs. Attention Agreement**
   - Top-k token agreement (k=3): ~30-40%
   - Top-k token agreement (k=5): ~35-45%
   - Random chance would be ~15% (k=3), ~25% (k=5)
   - Slight agreement but far from perfect alignment

### Qualitative Observations

1. **Attention Patterns Seem Interpretable**
   - Attention to syntactic heads (e.g., verbs attend to subjects)
   - Attention to semantic relations (e.g., entities to modifiers)
   - Attention to punctuation and special tokens ([CLS], [SEP])
   
2. **But Patterns Are Misleading**
   - Example: In MNLI, hypothesis words attend to premise words
   - Looks like "alignment" between premise and hypothesis
   - But manipulating these alignments doesn't affect predictions
   - The model doesn't rely on attention for its reasoning

3. **Special Token Attention**
   - [CLS] token often receives high attention
   - Serves as "no-op" or aggregation point
   - Not necessarily meaningful for interpretation
   - Can be redistributed without affecting outputs

## Implications and Discussion

### For Interpretability Research

1. **Attention ≠ Explanation**
   - Attention weights should not be used alone for interpretation
   - Need to validate with counterfactual or gradient-based methods
   - Correlation with linguistic structure doesn't imply causal role

2. **Need for Faithful Explanations**
   - Explanation should reflect actual model computation
   - Attention can be part of story but not whole story
   - Gradient-based methods are more faithful but less intuitive

3. **Human Interpretation Bias**
   - Humans over-interpret structured patterns
   - Confirmation bias in finding "meaningful" attention
   - Need objective evaluation metrics

### For Model Understanding

1. **Attention's Role in Transformers**
   - Attention computes representations but isn't sole mechanism
   - Feed-forward layers and residual connections matter
   - Later layer MLPs may "override" attention patterns

2. **Where Does Computation Happen?**
   - Not just in attention mechanism
   - Value projections and MLPs are critical
   - Attention might be more about routing than reasoning

### Limitations Acknowledged

1. **Scope of Analysis**
   - Focused on BERT/RoBERTa
   - Other architectures (GPT, T5) may differ
   - Task-specific vs. general findings unclear

2. **Alternative Interpretability Methods**
   - Didn't comprehensively evaluate all methods
   - Probing classifiers, causal interventions not tested
   - Gradient-based methods also have limitations

3. **Attention Might Matter in Some Cases**
   - Certain heads in certain layers might be more faithful
   - Copy mechanisms, coreference might rely on attention
   - Need task-specific analysis

## Related Work

### Prior Work on Attention Interpretability

1. **Attention as Explanation** (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019)
   - Earlier debate on whether attention explains predictions
   - Mixed results: sometimes correlates, sometimes doesn't
   - This paper extends analysis to BERT specifically

2. **BERTology Literature** 
   - Clark et al. (2019): Attention heads capture linguistic features
   - Tenney et al. (2019): BERT layers encode linguistic hierarchy
   - This paper argues these patterns may not be explanatory

3. **Gradient-Based Attribution**
   - Sundararajan et al. (2017): Integrated Gradients
   - Smilkov et al. (2017): SmoothGrad
   - Alternative to attention for understanding importance

### Alternative Interpretability Approaches

1. **Probing Classifiers** (Conneau et al., 2018)
   - Train classifier on hidden states to test what's encoded
   - Doesn't directly explain predictions
   
2. **Causal Interventions** (Vig et al., 2020)
   - Directly modify attention and measure effects
   - Related to adversarial approach in this paper

3. **Mechanistic Interpretability** (Olah et al., 2020)
   - Reverse-engineer circuits in neural networks
   - More recent direction in transformer understanding

## Critical Analysis

### Strengths

1. **Rigorous Experimental Design**
   - Multiple tasks, datasets, models
   - Adversarial framework is clever and convincing
   - Multiple importance baselines (gradients, erasure)

2. **Clear Communication**
   - Well-motivated problem
   - Accessible to practitioners
   - Important message for community

3. **Reproducible**
   - Methods clearly described
   - Standard models and datasets
   - Can be replicated by others

### Potential Weaknesses

1. **Adversarial Training Details**
   - How sensitive to hyperparameter λ?
   - How many steps needed for convergence?
   - Could provide more ablations

2. **Gradient Methods as Ground Truth**
   - Assumes gradients are "correct" importance
   - Gradients have their own issues (saturation, sharp minima)
   - Both might be incomplete views

3. **Attention Might Still Be Useful**
   - Even if not faithful, could be pedagogically valuable
   - Might capture some aspects of computation
   - Shouldn't completely dismiss attention analysis

## Open Questions and Future Directions

1. **Which Interpretability Methods ARE Faithful?**
   - If not attention or gradients, then what?
   - Need comprehensive evaluation framework
   - Formal criteria for faithfulness

2. **Task and Architecture Dependence**
   - Does illusion occur in GPT, T5, other models?
   - What about vision transformers?
   - Are some tasks/domains different?

3. **Can We Make Attention More Interpretable?**
   - Training objectives that encourage faithful attention?
   - Architectural changes (e.g., sparse attention)?
   - Explicit alignment with causality?

4. **Human Studies**
   - Do practitioners actually misinterpret attention?
   - What are consequences of misinterpretation?
   - Can we train people to be more careful?

## Practical Takeaways

### For Researchers

1. **Don't rely on attention alone** for model interpretation
2. **Validate interpretations** with multiple methods (gradients, erasure, probing)
3. **Be skeptical** of patterns that "seem meaningful"
4. **Use causal interventions** to test explanations
5. **Report correlations** between different importance measures

### For Practitioners

1. **Attention visualizations** are not sufficient for debugging
2. **Model behavior** requires deeper investigation
3. **Gradient-based methods** are more reliable (but still imperfect)
4. **Domain knowledge** should guide interpretation, not just attention
5. **Test counterfactuals** to verify understanding

### For Tool Developers

1. **Provide multiple views** of model behavior
2. **Include warnings** about attention limitations
3. **Make gradient computation** easy and accessible
4. **Support counterfactual analysis**
5. **Educate users** about interpretation pitfalls

## Code and Data Availability

- Code: Expected to be released by authors (check paper repository)
- Models: Standard HuggingFace models (BERT, RoBERTa)
- Data: Standard GLUE benchmark datasets
- Should be fully reproducible with public resources

## Citations and References

Key papers cited:
- Jain & Wallace (2019): Attention is not Explanation
- Wiegreffe & Pinter (2019): Attention is not not Explanation
- Clark et al. (2019): What Does BERT Look At?
- Vaswani et al. (2017): Attention is All You Need
- Sundararajan et al. (2017): Axiomatic Attribution for Deep Networks

## Appendix: Mathematical Details

### Attention Mechanism Recap

Standard multi-head attention:
```
Q = XW_Q, K = XW_K, V = XW_V
Attention(Q,K,V) = softmax(QK^T / √d_k) V
```

Where:
- X: input embeddings
- W_Q, W_K, W_V: learned projection matrices
- d_k: dimension of key vectors

### Adversarial Optimization

Formally, the adversarial attention objective is:

```
min_{W_Q, W_K} α · D_KL(A_adv || A_target) + (1-α) · L_task

subject to: A_adv = softmax(Q_adv K_adv^T / √d_k)
            Q_adv = XW_Q, K_adv = XW_K
```

Where:
- A_adv: adversarially modified attention
- A_target: target attention distribution (uniform, reversed, etc.)
- L_task: task loss (cross-entropy for classification)
- α: balancing hyperparameter

### Importance Metrics

**Input × Gradient:**
```
I_i^IxG = ||e_i ⊙ ∂L/∂e_i||
```

**Integrated Gradients:**
```
I_i^IG = ||e_i ⊙ ∫_{α=0}^1 ∂f(e_0 + α(e - e_0))/∂e_i dα||
```

**Attention:**
```
I_i^Attn = mean over heads and layers of A[:,i]
```

**Erasure:**
```
I_i^Erase = |f(x) - f(x \ {x_i})|
```

### Correlation Metrics

**Kendall's Tau:**
Measures rank correlation between two importance rankings. Robust to outliers, ranges from -1 to 1.

**Pearson's r:**
Measures linear correlation between importance scores. Sensitive to scale, ranges from -1 to 1.

Both used to compare attention weights with gradient/erasure importance.
