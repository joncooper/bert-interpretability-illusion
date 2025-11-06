# An Interpretability Illusion for BERT - Detailed Technical Notes

**Paper:** "An Interpretability Illusion for BERT"
**Authors:** Tolga Bolukbasi*, Adam Pearce*, Ann Yuan*, Andy Coenen, Emily Reif, Fernanda Viégas, Martin Wattenberg (Google Research)
**Published:** arXiv:2104.07143v1 [cs.CL] 14 Apr 2021

## Executive Summary

This paper describes a critical phenomenon in neural network interpretability research: an "interpretability illusion" where individual neurons or linear combinations of activations in BERT appear to encode simple, interpretable concepts when analyzed on a single dataset, but these interpretations fail to generalize when tested on different datasets. The same neuron can appear to encode completely different concepts depending on the dataset used for analysis.

## Core Finding: The Interpretability Illusion

### The Central Problem

When analyzing BERT using the standard interpretability technique of examining sentences that maximally activate specific neurons:
- **Single dataset view:** Neurons appear to have clear, consistent, interpretable meanings
- **Multi-dataset view:** The same neurons show completely different activation patterns across datasets
- **Key insight:** What appears to be a simple concept is actually something far more complex

### Motivating Example: Neuron 221 (Layer 12)

**On Quora Question Pairs (QQP) dataset:**
- "What is the meaning behind the song ""Angel"" by Eric Clapton?"
- "What's the meaning of Johnny Cash's song ""King of the Hill""?"
- "What is the meaning behind the Tears for Fears song ""Mad World""..."
- **Interpretation:** Appears to encode "song titles" or specific question syntax

**On QNLI (Wikipedia Q&A) dataset:**
- "On 16 June 2006, it was announced that Everton had entered into talks..."
- "On 15 September 1940, known as the Battle of Britain Day..."
- "On 20 August 2010, Queen's manager Jim Beach put out a Newsletter..."
- **Interpretation:** Appears to encode "historical events" or "sentences beginning with dates"

**On Toronto BookCorpus dataset:**
- "Lara pulled out the document Reed had supplied from Gresham's briefcase."
- "I take Kellan's business card from my pocket and stretch it over to Realm."
- "Pilcher took a walkie-talkie out of his coat and spoke into the receiver."
- **Interpretation:** Appears to encode something about "objects being taken from containers"

## Experimental Methodology

### Model Configuration

- **Model:** BERT-base uncased from HuggingFace Transformers library
- **No fine-tuning:** Pretrained model used as-is
- **Embedding extraction:** Final layer (layer 12) hidden state of [CLS] token
- **Dimensionality:** 768-dimensional embedding space
- **Total sentences analyzed:** 624,712 across all datasets

### Datasets

1. **Quora Question Pairs (QQP)**
   - 164,246 datapoints
   - Questions from Quora website
   - Source: Iyer et al., 2017

2. **Question-answering Natural Language Inference (QNLI)**
   - 58,645 datapoints
   - Passages from Wikipedia
   - Source: Wang et al., 2019

3. **Wikipedia (Wiki)**
   - 203,736 datapoints
   - Random subset of English Wikipedia
   - Prepared as in Devlin et al., 2018

4. **Toronto BookCorpus (Books)**
   - 198,085 datapoints (sampled)
   - Sentences from online novels
   - Source: Zhu et al., 2015

### Neuron and Direction Analysis

**Neuron identification:**
- A neuron is identified with a basis vector in the 768-dimensional space
- For neuron d: x(d)_l = 1 if l=d, else 0

**Top activating sentence identification:**
- For a dataset S and vector v:
  - Top activating sentence = arg max_{x∈S} ⟨x, v⟩
- Projection score: dot product between sentence embedding and direction

**Experimental conditions:**
1. Top 10 activating sentences for specific neurons
2. Top 10 activating sentences for random directions
3. Random sets of 10 sentences (baseline)

### Annotation Protocol

**Interface design:**
- Shows 10 sentences per condition
- Annotators know the dataset but not the condition
- Two annotators per set of sentences

**Pattern definition:**
- A pattern is any property shared by a set of sentences
- Can be structural (e.g., sentence length)
- Can be lexical (e.g., contains specific phrases)
- Used as proxies for learned concepts

**Annotation scope:**
- 25 randomly selected neurons
- 33 random directions
- 29 random sentence sets
- Total annotations tracked in Table 1

## Main Results

### Quantitative Findings

**Pattern prevalence (Table 1):**
- **Neurons:** 80% show patterns, 12% show no patterns, 8% conflicting
- **Random directions:** 82% show patterns, 0% show no patterns, 18% conflicting
- **Random sentences:** 14% show patterns, 48% show no patterns, 38% conflicting

**Baseline comparison:**
- Random sentences: 14% contain patterns (intrinsic dataset biases)
- Neurons/directions: >80% contain patterns (much higher than baseline)
- Patterns from top activating sentences are stronger (more positive examples)

**Cross-dataset consistency:**
- Average of **2.5 distinct patterns** per neuron across 4 datasets
- Patterns change dramatically depending on dataset
- Only 38% overlap in activation ranges between dataset pairs

**Annotator agreement:**
- Neurons: 8% conflicting annotations
- Random directions: 18% conflicting
- Random sentences: 38% conflicting
- Higher agreement for top activating → patterns are more salient

### Pattern Examples by Dataset

**QQP patterns:**
- Nested quotes, Colors, Mathematics, Military conflict, Population statistics
- Relationship advice, School exam questions, Questions of comparison, Programming

**QNLI patterns:**
- Biology, Geography, Technology, Numbers and dates, Military conflict
- Population statistics, War history, Windows 8, Etymology

**Wiki patterns:**
- Direct statement of fact, Music, Sporting, Age distribution
- Television shows, Olympic facts, Legalese, Measurements, School districts

**Books patterns:**
- Interpersonal relationships, Nature, Quoted speech, Spanish
- Sentence fragments, Medieval Europe, Very long sentences, Flirtation

## Explaining the Illusion: Three Sources

### 1. Dataset Idiosyncrasy

**Hypothesis 1:** QQP, QNLI, Wiki, and Books occupy distinct regions of BERT's embedding space.

**Evidence:**

**Visualization (Figure 1):**
- UMAP dimensionality reduction shows datasets form distinct clusters
- Little overlap between dataset regions in embedding space

**Classification accuracy (Figure 4):**
- Linear SVM trained to distinguish datasets based on embeddings
- High accuracy in separating datasets
- Confusion matrix shows minimal misclassification:
  - QQP: 1.0 precision on diagonal
  - QNLI: 0.83 precision
  - Wiki: 0.73 precision
  - Books: 0.96 precision

**Implication:**
- Each dataset occupies a distinct "slice" of the embedding space
- Top activating sentences from different datasets come from different regions
- Same direction can intersect different semantic regions per dataset
- Aligns with Aharoni & Goldberg (2020) - BERT representations can disambiguate datasets

### 2. Local Semantic Coherence

**Key observation:** Top activating sentences manifest patterns from BOTH local semantic coherence AND global directions.

#### Taxonomy of Concept Types

**Global concepts:**
- Become increasingly prevalent moving along a linear trajectory
- Direction exists that monotonically increases concept density
- Example: "math" - moving in direction increases math-related content
- Example: "positivity" - direction increases occurrence of happy, sunny, bliss, awesome

**Dataset-level concepts:**
- Associated with a direction, like global concepts
- BUT only meaningful within dataset's region of embedding space
- Do NOT generalize to arbitrary inputs outside dataset region

**Local concepts:**
- Emerge only as clusters in embedding space
- NO associated direction
- Example: math cluster around "e=mc²" contains similar math sentences
- But no direction makes math sentences increasingly prevalent

#### Evidence for Global/Dataset-Level Concepts

**Token frequency analysis (Table 3):**
- Measured monotonicity of token frequency along neuron directions
- Used quintiles to measure monotonicity
- Baseline probability of random monotonicity: 2/5! = 1.7%

**Monotonicity rates:**
- Single dataset: 22.7% - 29.6% of neuron/token pairs show monotonicity
- Two datasets: 7.4% - 10.9%
- Three datasets: 3.0% - 4.2%
- All four datasets: 1.9%

**Most monotonic tokens (Table 4):**
Top tokens changing monotonically across datasets (number of neurons in parentheses):
- Punctuation: " (125)
- Verbs: can (120), is (99), are (98), was (97), were (86), would (84), to (85)
- Pronouns: it (80), they (78), she (70)
- Other: that (91), if (88), not (77), god (76), for (75), which (73), more (73), of (68)

**Interpretation:**
- BERT learns to encode pervasive concepts as global directions
- Common grammatical elements (pronouns, verbs) have directional encoding
- Many concepts are dataset-specific rather than global

#### Evidence for Local Concepts

**Hypothesis 2:** When annotating top activating sentences, people identify concepts emerging from local semantic coherence.

**Analysis methodology:**

Define k-nearest neighbors in original embedding space:
- N_k(s) = arg max_{S'⊂S, |S'|=k} Σ_{ŝ∈S'} e_s · e_ŝ
- Used k=10 (consistent with annotation protocol)

For direction p with top-k activating sentences S_{p,k}:
- **D_{p,nearest}:** Dot products between each top sentence and its k nearest neighbors
- **D_{p,top}:** Dot products between all pairs of top activating sentences
- **D_{p,random}:** Dot products between top sentences and k random sentences

**Locality score:**
- L(h₁, h₂) = Σᵢ min(h₁(i), h₂(i)) / Σᵢ max(h₁(i), h₂(i))
- Jaccard similarity (intersection over union) between histograms
- Applied to D_{p,nearest} and D_{p,top}

**Results (Table 5):**

Locality scores (mean):
- **Meaningful neurons:** 0.026 overall (0.012 QQP, 0.014 QNLI, 0.037 Wiki, 0.042 Books)
- **Meaningless neurons:** 0.010 overall (0.003 QQP, 0.008 QNLI, 0.016 Wiki, 0.012 Books)
- **p-value:** 0.0004 overall (0.009 QQP, 0.090 QNLI, 0.064 Wiki, 0.014 Books)

**Interpretation:**
- Meaningful neurons have significantly higher locality scores
- Patterns arise primarily from local geometry, not global directions
- Top activating sentences cluster in embedding space
- Figure 7 shows histogram overlap between D_{p,nearest} and D_{p,top} for meaningful neurons

#### Outlier Analysis

**Question:** Are results driven by a small set of outlier sentences?

**Analysis on QQP dataset:**

Most frequently top-activating sentences:
1. "what does snoop dogg mean by ""lolos""..." (56 top activations)
2. "what are some songs like ""the dying of the light - noel gallaghar""?" (40)
3. "what inspired the tv show ""skins""?" (32)

**Distance analysis:**
- These 3 sentences have highest mean pairwise Euclidean distances in QQP
- Top 20 most distant sentences all have ≥11 top activations
- Most distant 1% of sentences account for 48% of all top activations

**But:**
- Most sentences are only top-activating once or twice
- 7,680 top-ten activating sentences include 4,551 unique sentences
- Illusion persists when most distant 1% and 10% are omitted

**Conclusion:** Results not solely due to outliers, but outliers contribute

### 3. Annotator Error/Bias

**Observation:** Humans may see patterns where none objectively exist.

**Evidence (Table 6):**

Annotator variability:
- Annotator 0: 27/30 (0.90) patterns found
- Annotator 1: 27/33 (0.82)
- Annotator 2: 23/24 (0.92)
- Annotator 3: 23/40 (0.56)
- Annotator 4: 30/52 (0.58)
- Annotator 5: 40/25 (1.6) - *finds more patterns than directions annotated*

**Key finding:** Significant inter-annotator variability in pattern detection

**Implication:** Some patterns may reflect annotator imagination rather than objective properties

## Key Contributions

1. **Identification of interpretability illusion** in language model activation space analysis

2. **Methodological recommendation:** Interpretability researchers MUST conduct experiments on multiple datasets

3. **Taxonomy of concept representations:**
   - **Global concepts:** Linear directions meaningful across entire embedding space
   - **Dataset-level concepts:** Directional but only valid within dataset region
   - **Local concepts:** Clustered but without associated global direction

4. **Evidence that common interpretability methods may be misleading** when applied to single datasets

## Technical Details & Formulae

### Embedding Extraction
- Model: BERT-base uncased
- Layer: 12 (final layer)
- Token: [CLS] hidden state
- Dimensionality: 768

### Neuron Representation
- Neuron d as one-hot vector: x(d) ∈ R^768
- x(d)_l = 1 if l=d, else 0

### Activation Scoring
- Top activating sentence for vector v: arg max_{x∈S} ⟨x, v⟩
- Projection score: ⟨x, v⟩ (dot product)

### Concept Distribution
- C = [c₁, c₂, ..., c_N] where Σcᵢ = 1.0
- Concept purity: skew of concept distribution vector
- Pure concept: one-hot distribution

### Locality Analysis
- Nearest neighbors: N_k(s) = arg max_{S'⊂S, |S'|=k} Σ_{ŝ∈S'} e_s · e_ŝ
- Locality score: L(h₁, h₂) = Σᵢ min(h₁(i), h₂(i)) / Σᵢ max(h₁(i), h₂(i))

### Monotonicity Testing
- Used quintiles of activation levels
- Baseline monotonicity probability: 2/5! = 1.7%
- Token counted as monotonic if frequency increases/decreases monotonically across quintiles

### Embedding Properties (Appendix A)
- Un-normalized embeddings used
- Most vector norms concentrated around ~14 ≈ sqrt(768)/2
- Neuron outputs typically in range [-1.0, 1.0]
- Normalization tested but didn't change top activating sentences

## Methodological Implications

### For Interpretability Research

1. **Multiple dataset testing is essential**
   - Single dataset can give misleading interpretations
   - Patterns must be validated across diverse data sources
   - Cross-dataset consistency indicates more robust concepts

2. **Distinguish between concept types**
   - Test whether concepts are local, dataset-level, or global
   - Use monotonicity analysis across datasets
   - Check locality scores

3. **Be aware of dataset biases**
   - Even random sentences show patterns (14% baseline)
   - Common corpora represent narrow slices of possible English
   - Dataset clustering in embedding space affects interpretability

4. **Account for annotator variability**
   - Use multiple annotators
   - Measure inter-annotator agreement
   - Some patterns may be subjective

### For Bias Mitigation

**Warning:** Using concept directions for bias measurement/mitigation without multi-dataset validation could have unintended effects.

**Recommendation:** Validate bias-related concept directions across multiple datasets before deploying mitigation strategies.

## Limitations & Future Work

### Acknowledged Limitations

1. **Scope:** Focused on layer 12 of BERT
   - Informal investigation of layers 2 and 7 suggests illusion occurs there too
   - Full analysis of all layers not performed

2. **Model scope:** Only BERT-base analyzed
   - Other sentence models not tested
   - Token-level embeddings not examined

3. **Dataset scope:** Four text datasets
   - All are standard NLP benchmarks
   - May not represent full diversity of English

### Suggested Future Directions

1. **Extend to other models:**
   - Other transformer architectures
   - Different model sizes
   - Token-level vs. sentence-level embeddings

2. **Extend to other layers:**
   - Systematic analysis of all BERT layers
   - Characterize how illusion changes by layer

3. **Extend to other modalities:**
   - Images
   - Graphs
   - Multimodal models

4. **Geometric characterization:**
   - More rigorous mathematical framework for concept geometry
   - Better methods to distinguish local vs. global concepts
   - Quantify embedding space coverage of datasets

5. **Improved interpretability methods:**
   - Develop techniques robust to dataset idiosyncrasy
   - Methods to identify truly global concepts
   - Better ways to validate interpretations

## Related Work Context

### Interpretability Methods

**Feature visualization & maximally activating inputs:**
- Nguyen et al. (2016): Generating inputs that maximize activations
- Poerner et al. (2018): Interpretable textual neuron representations
- Bäuerle & Wexler: "What does BERT dream of?"
- Na et al. (2019): Natural language concepts in CNN units

**Analysis of maximally activating real samples:**
- Zhou et al. (2015): Neurons for objects in scenes
- Bau et al. (2017): Network dissection for visual representations
- Zeiler & Fergus (2014): Visualizing top activating image patches
- Olah et al. (2017): Comparing dataset and generated maximally activating images
- Dalvi et al. (2019): Comparing maximally activating with probing methods

**This paper's contribution:** Shows these methods can be misleading without multi-dataset validation

### Embedding Space Analysis

**Global structure:**
- Mikolov et al. (2013): Directions encode concepts (king-man+woman=queen)
- Bolukbasi & Chang (2016): Gender bias directions
- Raghu et al. (2017): SVCCA for interpretability
- Olah et al. (2020): Circuits in neural networks
- Vig et al. (2020): Causal mediation analysis

**Local structure:**
- Bengio et al. (2003): Nearest neighbors are similar

**Linguistic understanding:**
- Tenney et al. (2019): Probing for sentence structure
- Durrani et al. (2020): Localizing linguistic elements to neurons
- Manning et al. (2020): Emergent linguistic structure

**This paper's contribution:** Distinguishes local, global, and dataset-level concept directions

### Bias Detection & Mitigation

- Kaneko & Bollegala (2019): Gender-preserving debiasing
- Manzini et al. (2019): Multiclass bias in embeddings

**This paper's warning:** Bias mitigation using concept directions needs multi-dataset validation

## Practical Takeaways

1. **Don't trust single-dataset interpretability analyses**
   - Always validate on multiple datasets
   - Different datasets can show completely different patterns

2. **Be skeptical of simple neuron interpretations**
   - Neurons likely encode complex, multifaceted information
   - Simple patterns may be artifacts of dataset geometry

3. **Test concept generalization**
   - Use monotonicity analysis across datasets
   - Check locality scores
   - Verify interpretations hold across data distributions

4. **Consider the geometry**
   - Datasets cluster in embedding space
   - Local coherence creates illusory patterns
   - True global concepts are rarer than they appear

5. **Account for human factors**
   - Annotators vary in pattern detection
   - Some patterns may be subjective
   - Use rigorous validation methods

## Open Questions

1. How prevalent is this illusion in other models and modalities?
2. Can we develop interpretability methods robust to dataset idiosyncrasy?
3. What fraction of apparent concepts are truly global vs. local/dataset-level?
4. How does the illusion vary across layers?
5. Can we better characterize the geometry of concept representation?
6. Are there universal concepts that transcend dataset boundaries?
