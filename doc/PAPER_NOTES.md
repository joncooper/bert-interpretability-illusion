# Paper Notes: An Interpretability Illusion for BERT

**Paper**: An Interpretability Illusion for BERT  
**arXiv ID**: 2104.07143v1  
**Authors**: Tolga Bolukbasi*, Adam Pearce*, Ann Yuan*, Andy Coenen, Emily Reif, Fernanda Viégas, Martin Wattenberg (*Equal contribution)  
**Affiliation**: Google Research, Cambridge, MA, USA  
**Date**: April 14, 2021  
**Pages**: 12

## Abstract and Main Thesis

The paper describes an "interpretability illusion" that arises when analyzing BERT's internal representations. Individual neurons (and linear combinations of activations) in BERT may **spuriously appear to encode single, simple concepts** when examined on one dataset, but reveal completely different patterns on another dataset. The authors trace this illusion to:
1. Geometric properties of BERT's embedding space
2. The fact that common text corpora represent narrow slices of possible English sentences

**Key insight**: Patterns that look consistent and interpretable within a single dataset do not generalize across datasets, making them unreliable for understanding what the model has actually learned.

## Key Findings

### 1. The Core Illusion: Dataset-Dependent Patterns

The central finding is demonstrated through neuron 221 in layer 12:

**On Quora Question Pairs dataset**, top activating sentences:
- "What is the meaning behind the song ""Angel"" by Eric Clapton?"
- "What's the meaning of Johnny Cash's song ""King of the Hill""?"
- "What is the meaning behind the Tears for Fears song ""Mad World""..."

→ **Interpretation**: Neuron encodes "song titles" or specific syntactic structure

**On Wikipedia QA dataset (QNLI)**, same neuron's top activating sentences:
- "On 16 June 2006, it was announced that Everton had entered into talks..."
- "On 15 September 1940, known as the Battle of Britain Day..."
- "On 20 August 2010, Queen's manager Jim Beach put out a Newsletter..."

→ **Interpretation**: Neuron encodes "historical events" or "sentences beginning with dates"

**On Toronto BookCorpus**, same neuron's top activating sentences:
- "Lara pulled out the document Reed had supplied from Gresham's briefcase."
- "I take Kellan's business card from my pocket and stretch it over to Realm."
- "Pilcher took a walkie-talkie out of his coat and spoke into the receiver."

→ **Interpretation**: Yet another completely different pattern!

**Conclusion**: The neuron doesn't consistently encode any single concept - the apparent pattern is an illusion that depends entirely on the dataset.

### 2. The Illusion is Widespread

- Tested 25 randomly selected neurons
- Tested 33 random directions (linear combinations)
- Found patterns in >80% of neurons/directions within each dataset
- On average, each neuron showed **2.5 distinct patterns** across the 4 datasets
- The illusion affects both individual neurons AND random directions equally
- Appears in layer 12 (final layer) and informally observed in layers 2 and 7

### 3. Quantitative Results

**Pattern Detection Rates** (Table 1):
- Neurons: 80% contain patterns, 12% no pattern, 8% conflicting annotations
- Random directions: 82% contain patterns, 0% no pattern, 18% conflicting
- Random sentence sets (baseline): 14% contain patterns, 48% no pattern, 38% conflicting

**Pattern Strength**:
- Top activating sentences have ~6.8 examples per pattern on average
- Random sentences have ~5.0 examples per pattern
- Meaningful patterns have significantly more positive examples than baseline

## Technical Details

### Methodology

#### 1. Datasets Used

Four diverse text corpora totaling 624,712 sentences:

1. **Quora Question Pairs (QQP)**: 164,246 questions from Quora
2. **Question-answering NLI (QNLI)**: 58,645 Wikipedia passages  
3. **Wikipedia (Wiki)**: 203,736 random sentences from English Wikipedia
4. **Toronto BookCorpus (Books)**: 198,085 sentences from online novels

#### 2. Embedding Extraction

**Model**: BERT-base uncased from HuggingFace Transformers
- No fine-tuning or dataset-specific modifications
- Used final layer (layer 12) hidden state of [CLS] token as sentence embedding
- Results in 768-dimensional embedding for each sentence

**Rationale**: This is the default method in HuggingFace pipelines and used in original BERT paper.

**Validation**: Created UMAP visualization showing embeddings form highly coherent clusters (Figure 1).

#### 3. Neuron Analysis Method

**Neuron Definition**: A neuron is identified with a basis vector in the 768-dimensional space:

```
x^(d)_l = 1 if l = d
          0 otherwise
```

**Finding Top Activating Sentences**:
For a neuron/direction `v` and dataset `S`:

```
Top activating sentence for v = arg max_{x ∈ S} ⟨x, v⟩
```

Where the dot product ⟨x, v⟩ is called the "projection score."

**Analysis Process**:
1. Randomly select neurons (basis vectors) from layer 12
2. Also test random directions (random vectors in 768-d space)
3. For each neuron/direction, find top 10 activating sentences per dataset
4. Human annotators examine sentences for patterns

#### 4. Annotation Protocol

**Interface Design**:
- Shows 10 sentences in one of three conditions:
  1. Top 10 for a neuron
  2. Top 10 for a random direction  
  3. Random 10 sentences (control)
- Annotators know dataset but not condition
- Annotators mark whether pattern exists and which sentences match it

**Annotation Scope**:
- 25 neurons (randomly selected)
- 33 random directions
- 29 random sentence sets
- Two annotators per set for reliability

**Pattern Definition**:
A pattern is any property shared by multiple sentences:
- **Structural**: e.g., all same length, specific syntactic structure
- **Lexical**: e.g., contain phrase variations like "coat of arms"
- **Semantic**: e.g., all about a topic like "military conflict"

Patterns serve as proxies for concepts the model may have learned.

## Explaining the Illusion: Three Sources

The authors identify three contributing factors:

### Source 1: Dataset Idiosyncrasy

**Hypothesis**: QQP, QNLI, Wiki, and Books occupy distinct, non-overlapping regions of BERT's embedding space.

**Evidence**:
1. **UMAP Visualization** (Figure 1): Sentences cluster neatly by dataset - the four datasets form distinct, separate clusters

2. **Linear SVM Classification** (Figure 4): Trained classifier to distinguish datasets based on embeddings
   - Achieves very high accuracy
   - Shows datasets are easily separable in embedding space

3. **Activation Range Overlap**: Only 38% of neuron/dataset pairs have overlapping activation ranges for top-10 sentences

**Implication**: If datasets occupy different regions, then moving along any direction will encounter different sentences from different datasets, leading to different apparent patterns.

**Schematic (Figure 3)**: Imagine an arrow (neuron direction) through embedding space. Red dots (Dataset A) and blue dots (Dataset B) are in different regions. The highest-activating red dots vs. blue dots will be semantically unrelated, even though they're all "high-activating."

### Source 2: Local Semantic Coherence

**Key Observation**: Top activating sentences show patterns from BOTH:
- Local semantic clustering (nearby sentences are similar)
- Global concept directions (some directions correlate with specific tokens/concepts)

#### Global vs. Dataset-Level vs. Local Concepts

The paper introduces a taxonomy:

**Global Concepts**: Become increasingly prevalent along a linear trajectory through embedding space, starting from ANY point
- Example: "math" - moving along direction increases math-related sentences
- Detected by monotonic token frequency changes

**Dataset-Level Concepts**: Similar to global, but only meaningful within one dataset's region of embedding space
- Direction works for that dataset but doesn't generalize

**Local Concepts**: Emerge as clusters without any associated direction
- Example: math sentences cluster together near "e=mc²"
- No specific direction makes math more prevalent - it's just local grouping

####  Evidence for Global/Dataset-Level Concepts (Table 3)

**Monotonic Token Analysis**: 
- Examined 915 tokens appearing ≥100 times in all datasets
- For each neuron/token pair, checked if token frequency changes monotonically across activation quintiles
- Baseline probability of monotonicity by chance: 1.7%

**Results**:
- ~27% of neuron/token pairs show monotonic relationships within single datasets
- Only 1.9% show monotonicity across ALL four datasets
- Suggests most "concept directions" are dataset-specific, not global

**Most Monotonic Tokens** (Table 4): Common function words and pronouns
- Parenthesis "(", "can", "is", "are", "was", "that", "if", "were", "to", "would"
- Suggests BERT learns some global directions for pervasive linguistic features

#### Evidence for Local Concepts (Locality Score Analysis)

**Hypothesis 2**: Annotators identify concepts from local semantic coherence, not directional information.

**Test Method**:
1. For each sentence `s` in top-10 activating for direction `p`:
   - Find its k=10 nearest neighbors N_k(s) in original embedding space
   - Compute dot products between s and its neighbors → D_{p,nearest}
   - Compute dot products between all pairs of top-10 → D_{p,top}
   - Compute dot products between top-10 and random sentences → D_{p,random}

2. **Locality Score**: Jaccard similarity between histograms of D_{p,nearest} and D_{p,top}

```
L(h1, h2) = Σ_i min(h1(i), h2(i)) / Σ_i max(h1(i), h2(i))
```

**Interpretation**: 
- High L → top activating sentences are close to each other in original space (local clustering)
- Low L → top activating sentences are scattered (relying on direction, not locality)

**Results** (Table 5):
- Meaningful neurons: L = 0.026 (mean across datasets)
- Meaningless neurons: L = 0.010  
- p-value: 0.0004 (highly significant)

**Conclusion**: Patterns found by annotators arise primarily from local geometry, not from the neuron direction itself.

#### Outlier Analysis

**Concern**: Maybe patterns only exist for a small set of weird outlier sentences?

**Analysis on QQP**:
- Top 3 most distant (outlier) sentences each activate 32-56 neurons
- Most distant 1% of sentences account for 48% of all top activations
- Shows lack of diversity among outliers

**However**: 
- Out of 7,680 top-ten activating slots, there are 4,551 unique sentences
- Illusion persists even when removing most distant 1% or 10% of sentences
- Not purely an outlier phenomenon

### Source 3: Annotator Error/Bias

**Evidence** (Table 6):
- Annotators disagree on whether sentences contain patterns
- Wide variation in individual annotator tendency to find patterns:
  - Annotator 0: Found patterns in 90% of cases
  - Annotator 3: Found patterns in 56% of cases
  - Annotator 5: Found patterns in 160% of cases (multiple patterns per set!)

**Implication**: Some "patterns" may reflect annotator imagination rather than objective properties of the sentences.

## Sample Annotated Patterns (Table 2)

**QQP**: Nested quotes, Colors, Mathematics, Military conflict, Population statistics, Relationship advice, School exam questions, Questions of comparison, Programming

**QNLI**: Biology, Geography, Technology, Numbers and dates, Military conflict, Population statistics, War history, Windows 8, Etymology

**Wiki**: Direct statement of fact, Music, Sporting, Age distribution, Television shows, Olympic facts, Legalese, Measurements, School districts

**Books**: Interpersonal relationships, Nature, Quoted speech, Spanish, Sentence fragments, Medieval Europe, Very long sentences, Flirtation

(Full list of all patterns in Table 7 of Appendix)

## Implications and Discussion

### For Interpretability Research

**Primary Recommendation**: **Test interpretability hypotheses on multiple datasets**

The core lesson is that an interpretation that seems valid on one dataset may completely fail to generalize. This has several implications:

1. **Validation is Essential**
   - Never trust patterns found on a single dataset
   - Test on at least 2-3 diverse datasets
   - Look for consistency across data distributions

2. **Distinguish Local from Global Concepts**
   - Local patterns (clustering) ≠ directional concepts
   - Dataset-level directions ≠ universal representations
   - Need to characterize geometry of concept representation

3. **Be Skeptical of Intuitive Patterns**
   - Human tendency to see patterns even in randomness
   - Annotator bias significantly affects interpretation
   - Use objective metrics alongside human judgment

4. **Consider Dataset Bias**
   - Common NLP corpora are NOT representative of all English
   - Different corpora occupy different embedding regions
   - Model representations may be dataset-specific

### For Model Understanding

1. **What Neurons Actually Encode**
   - Individual neurons don't have simple, universal meanings
   - Activations reflect complex interaction of:
     - Local clustering in embedding space
     - Dataset-specific patterns
     - Some global concept directions
   - Meaning is context-dependent

2. **Geometry of BERT's Representation Space**
   - Strong local semantic coherence (similar sentences cluster)
   - Some global directions exist (function words, pronouns)
   - Most apparent "concept directions" are dataset-specific
   - Embedding space partitioned by data source

3. **Implications for Concept-Based Methods**
   - Using concept directions for bias mitigation could have unintended effects
   - Must validate directions across multiple contexts
   - Dataset-specific directions won't generalize to new inputs

### Methodological Recommendations

1. **For Researchers Analyzing Neurons**:
   - Use multiple diverse datasets
   - Measure locality scores to distinguish local vs. directional patterns
   - Test for monotonic token frequency changes
   - Employ multiple annotators and measure agreement

2. **For Building Interpretability Tools**:
   - Provide warnings about single-dataset analysis
   - Automatically test across multiple corpora
   - Visualize embedding space geometry
   - Show uncertainty in concept labels

3. **For Practitioners**:
   - Don't assume neuron interpretations from research papers generalize
   - Test on your specific data distribution
   - Use interpretations as hypotheses, not facts
   - Validate with multiple methods (probing, interventions, etc.)

## Related Work

### Prior Work on Neuron Interpretability

**Analyzing Neurons via Max-Activating Inputs**:
- Zhou et al. (2015), Bau et al. (2017): Found neurons responding to specific objects in images
- Szegedy et al. (2014): Convolutional networks have neurons for semantically related inputs
- Zeiler & Fergus (2014): Visualized top activating image patches
- Olah et al. (2017): Compared max-activating dataset images vs. synthesized images

**For Language Models**:
- Na et al. (2019): Looked for patterns in max-activating sentences
- Dalvi et al. (2019): Compared max-activation method to probing classifiers
- Poerner et al. (2018): Generated inputs to maximize neuron activation
- Durrani et al. (2020): Found linguistic elements localized to individual neurons

**This paper's contribution**: First to systematically show that max-activation patterns DON'T generalize across datasets.

### Embedding Space Structure

**Local Structure**:
- Bengio et al. (2003): Nearest neighbors in embedding space are semantically similar
- Aharoni & Goldberg (2020): BERT representations can disambiguate datasets

**Global Directions**:
- Mikolov et al. (2013): Word2Vec directions encode semantic relationships (king-man+woman=queen)
- Bolukbasi & Chang (2016): Bias directions in word embeddings
- Li et al. (2015): Visualizing neural model representations
- Olah et al. (2020): Concept directions in vision models
- Vig et al. (2020): Causal mediation analysis for gender bias

**This paper's contribution**: Distinguishes global, dataset-level, and local concepts; shows most apparent directions are dataset-specific.

### Probing and Linguistic Structure

- Tenney et al. (2019): Probing for sentence structure in contextualized embeddings
- Manning et al. (2020): Emergent linguistic structure in self-supervised neural networks
- Different from this work: Probing tests what CAN be extracted, not what neurons natively encode

### Bias and Fairness

- Bolukbasi & Chang (2016), Manzini et al. (2019), Kaneko & Bollegala (2019): Using concept directions to measure/mitigate bias

**Warning from this paper**: Using concept directions without validating across datasets could have unintended effects.

## Limitations and Future Work

### Acknowledged Limitations

1. **Scope**:
   - Only studied BERT-base uncased
   - Only examined layers 2, 7, and 12 (final layer most thoroughly)
   - Only used [CLS] token embeddings (other methods like mean pooling not tested)

2. **Annotation**:
   - Human annotators have biases
   - "Pattern" definition is somewhat subjective  
   - Only used 2 annotators per set (more would be better)

3. **Datasets**:
   - Only tested 4 specific corpora
   - Need to test on more diverse data sources
   - Unclear how results generalize to non-English text

### Suggested Future Directions

1. **Broader Model Coverage**:
   - Other sentence models (Sentence-BERT, USE, etc.)
   - Token-level embeddings instead of sentence-level
   - Other layers (systematic study of all 12 layers)
   - Other model families (GPT, T5, RoBERTa, etc.)

2. **Other Modalities**:
   - Do similar illusions occur in vision transformers?
   - What about multimodal models?
   - Audio, graph data, etc.?

3. **Geometric Characterization**:
   - More rigorous mathematical analysis of embedding space
   - Better methods to distinguish local vs. global vs. dataset concepts
   - Formal criteria for when a direction is "meaningful"

4. **Cross-Dataset Validity**:
   - Framework for testing interpretation validity
   - Methods to find truly global concept directions
   - Understanding when/why interpretations transfer

5. **Practical Tools**:
   - Automated systems to test interpretations across datasets
   - Visualization tools showing embedding space partitioning
   - Calibrated confidence scores for concept labels

## Critical Analysis

### Strengths

1. **Novel and Important Finding**
   - First systematic demonstration that neuron interpretations don't generalize
   - Challenges common practice in interpretability research
   - Has immediate practical implications

2. **Rigorous Experimental Design**
   - Multiple datasets with different characteristics
   - Controlled comparison (neurons vs. random directions vs. random sentences)
   - Multiple annotators for reliability
   - Both qualitative and quantitative analysis

3. **Thorough Explanation**
   - Doesn't just identify the problem
   - Provides detailed analysis of causes (dataset idiosyncrasy, local coherence, annotator bias)
   - Introduces useful taxonomy (global/dataset-level/local concepts)

4. **Clear Presentation**
   - Compelling opening example (neuron 221)
   - Excellent visualizations (UMAP, confusion matrix, locality distributions)
   - Well-structured from observation to explanation

5. **Actionable Recommendations**
   - Clear advice: test on multiple datasets
   - Provides specific metrics (locality score, monotonicity)
   - Acknowledges limitations honestly

### Potential Weaknesses

1. **Limited Model Coverage**
   - Only BERT-base (what about BERT-large? RoBERTa? GPT?)
   - Only final layer studied in depth
   - Only one type of embedding extraction ([CLS] token)

2. **Dataset Selection**
   - Four datasets may not be enough
   - All are English, relatively formal text
   - What about: social media, code, dialogue, non-English?

3. **Annotation Subjectivity**
   - High inter-annotator variability (Table 6)
   - "Pattern" definition is loose
   - Some annotators find 1.6x more patterns than others

4. **Locality Score Interpretation**
   - Mean locality scores are quite low (0.026 vs 0.010)
   - Effect size is small in absolute terms
   - Statistical significance doesn't mean practical importance

5. **Missing Comparisons**
   - Doesn't compare to other interpretability methods (probing, causal intervention)
   - Doesn't test whether ANY method finds generalizable patterns
   - Could the datasets themselves be the problem, not the neuron analysis method?

6. **Practical Alternatives Not Explored**
   - Paper identifies problem but doesn't propose solutions
   - How SHOULD we interpret neurons if this method doesn't work?
   - What methods DO produce reliable interpretations?

### Questions Raised

1. **Are there ANY meaningful neurons?**
   - Paper shows most neurons don't have simple universal meanings
   - But maybe some do? Were any found that are consistent across datasets?

2. **Is this specific to BERT?**
   - Different architectures might have different properties
   - Explicit multitask models might develop more universal representations

3. **What about fine-tuned models?**
   - All experiments on pre-trained, not fine-tuned BERT
   - Task-specific fine-tuning might create more interpretable neurons

4. **Token-level vs. Sentence-level**
   - Paper uses [CLS] token for sentence representation
   - Individual token representations might be more/less interpretable

5. **Can we fix this?**
   - Training objectives that encourage universal concepts?
   - Architectural changes?
   - Or is this fundamental to how neural networks learn?

## Practical Takeaways

### For Researchers

**DO**:
- ✅ Test interpretability hypotheses on multiple diverse datasets
- ✅ Use locality scores to distinguish local clustering from directional concepts
- ✅ Measure monotonic token frequency changes to detect global directions
- ✅ Report inter-annotator agreement
- ✅ Visualize embedding space geometry
- ✅ Consider dataset idiosyncrasy in your analysis

**DON'T**:
- ❌ Trust neuron interpretations from a single dataset
- ❌ Assume patterns found by human annotators reflect true model concepts
- ❌ Ignore the possibility of local semantic coherence
- ❌ Use concept directions for applications (e.g., bias mitigation) without cross-dataset validation

### For Practitioners

**When analyzing models**:
- Test on your specific data distribution (don't assume research findings generalize)
- Use multiple interpretability methods, not just neuron activation analysis
- Be skeptical of intuitive patterns - validate quantitatively
- Consider that different data sources may activate different "concepts"

**When building systems**:
- Don't rely on single neuron activations for decisions
- If using concept directions, validate across multiple contexts
- Include multiple datasets in testing pipelines
- Monitor how interpretations change across data distributions

### For Tool Developers

**Interpretability tools should**:
- Automatically test across multiple datasets
- Provide confidence scores/uncertainty estimates
- Show embedding space visualizations
- Warn users about single-dataset limitations
- Include locality metrics in neuron descriptions
- Allow comparison across data sources

## Key Equations and Metrics

### Neuron Representation
```
Basis vector: x^(d)_l = 1 if l=d, 0 otherwise
```

### Projection Score
```
Score(sentence, direction) = ⟨embedding, direction⟩
```

### Locality Score
```
L(h1, h2) = Σ_i min(h1(i), h2(i)) / Σ_i max(h1(i), h2(i))
```
Measures overlap between histogram of distances to nearest neighbors vs. distances to top-activating sentences.

### Monotonicity Test
```
For neuron n and token t, check if token frequency changes
monotonically across activation quintiles.
Baseline probability: 2/5! = 1.7%
Observed: ~27% for single datasets, ~2% across all datasets
```

## Conclusion

This paper makes a crucial contribution to neural network interpretability by demonstrating that **interpretations that seem valid on one dataset may be completely invalid on another**. The "interpretability illusion" arises from:
1. Datasets occupying distinct regions of embedding space
2. Local semantic coherence creating apparent patterns
3. Human tendency to see patterns

The key lesson: **always validate interpretability findings across multiple diverse datasets**. Patterns that don't generalize are likely spurious.

Despite revealing this illusion, the paper also provides evidence that BERT's embedding space DOES contain some structure:
- Global concept directions exist (for common linguistic features)
- Strong local semantic coherence (similar sentences cluster)
- Dataset-level patterns (though these don't generalize)

The challenge for future work is developing methods that can reliably distinguish meaningful, generalizable concepts from dataset-specific or locally-coherent patterns.

## Code and Data Availability

- **Model**: BERT-base uncased from HuggingFace Transformers
- **Datasets**: All publicly available (QQP, QNLI, Wikipedia, BookCorpus)
- **Code**: Not explicitly mentioned in paper (check authors' GitHub)
- **Annotations**: Appear to be manual (not released as dataset)

## Citations

Key papers referenced:
- Devlin et al. (2018): Original BERT paper
- Bau et al. (2017): Network dissection for vision
- Tenney et al. (2019): Probing for linguistic structure  
- Aharoni & Goldberg (2020): Dataset clustering in pretrained LMs
- Bolukbasi & Chang (2016): Debiasing word embeddings
- Olah et al. (2020): Circuits in vision transformers

## Appendix Notes

**Table 7**: Complete list of all annotated patterns across datasets (not reproduced here but available in paper)

**Normalization**: Authors chose to use raw (un-normalized) embeddings to stay closer to how BERT actually uses them. Most vector norms concentrated around 14 ≈ √768/2.

**Key Figure Numbers**:
- Figure 1: UMAP visualization showing dataset clustering
- Figure 3: Schematic of how dataset separation causes illusion  
- Figure 4: Linear SVM confusion matrix
- Figure 5: Global vs. local concepts illustration
- Figure 6: Token frequency monotonicity examples
- Figure 7: Locality score distributions
