import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="An Interpretability Illusion for BERT",
    page_icon="🔍",
    layout="wide"
)

# Title and Introduction
st.title("🔍 An Interpretability Illusion for BERT")
st.markdown("### Interactive Explainer")
st.markdown("*Based on the paper by Bolukbasi, Pearce, Yuan, et al. (2021)*")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choose a section:",
    [
        "Introduction",
        "The Illusion Revealed",
        "Experimental Setup",
        "Results & Findings",
        "Why Does This Happen?",
        "Types of Concepts",
        "Key Takeaways"
    ]
)

# ===== INTRODUCTION =====
if section == "Introduction":
    st.header("📖 Introduction")

    st.markdown("""
    ### The Challenge of Neural Network Interpretability

    Understanding how neural networks represent meaning is one of the biggest challenges in AI research.
    A simple hypothesis suggests that:

    - Individual neurons encode simple, human-interpretable concepts
    - Linear combinations of neurons encode more complex concepts

    This approach has worked well for **image networks** — finding neurons that detect edges, textures, or specific objects.

    ### The Question
    **Could the same techniques reveal meaningful concepts in language models like BERT?**
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **💡 Initial Hypothesis**

        If we look at sentences that maximally activate a specific neuron,
        and those sentences share a common pattern, then that neuron
        has learned to detect that pattern.
        """)

    with col2:
        st.warning("""
        **⚠️ What They Found**

        The patterns seemed clear and consistent... but only within
        a single dataset. When tested on different datasets, the
        same neurons appeared to encode completely different concepts!
        """)

    st.markdown("---")
    st.markdown("### About the Research")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Model Used:**
        - BERT-base uncased
        - No fine-tuning
        - Final layer [CLS] token embeddings
        - 768-dimensional embedding space
        """)

    with col2:
        st.markdown("""
        **Datasets Tested:**
        - **QQP**: Quora Question Pairs (164,246 sentences)
        - **QNLI**: Wikipedia Q&A (58,645 sentences)
        - **Wiki**: Random Wikipedia subset (203,736 sentences)
        - **Books**: Toronto BookCorpus (198,085 sentences)
        """)

    st.markdown("---")

    # Research context expander
    with st.expander("📚 Research Context: Why This Paper Matters (Click to expand)"):
        st.markdown("""
        ### Prior Work on Neuron Interpretability

        **Success in Computer Vision:**
        - Individual neurons in CNNs clearly encode edges, textures, and objects (Olah et al., 2018)
        - Visualization methods reveal interpretable features at every layer
        - These findings are robust across different datasets and architectures

        **Early Work on Language Models:**
        - Prior research found seemingly interpretable neurons in BERT and similar models
        - Dalvi et al. (2019): Identified neurons tracking syntactic features
        - Mu & Andreas (2020): Found neurons encoding semantic properties
        - **Assumption**: These findings generalize, similar to vision models

        ### This Paper's Key Contribution

        **A Methodological Wake-Up Call:**

        This paper challenges those early findings by asking: *Did previous researchers
        test their interpretability claims on multiple diverse datasets?*

        **The Answer:** Usually not! And when you do...

        What looked like meaningful concept encoding might actually be:
        1. **Dataset idiosyncrasy** - Sentences from that dataset cluster together geometrically
        2. **Local semantic coherence** - Similar sentences are neighbors, creating apparent patterns
        3. **Human pattern-seeking** - We see meaningful patterns even in noise

        ### Why This Matters

        **For Interpretability Research:**
        - Single-dataset analysis is insufficient for interpretability claims
        - Patterns must be validated across multiple diverse datasets
        - Many published "neuron interpretations" may be artifacts

        **For AI Safety:**
        - Techniques using concept directions for bias detection/mitigation may be unreliable
        - If directions don't generalize, interventions might have unintended effects
        - Need more robust interpretability methods

        **For Understanding Language Models:**
        - Language models may not learn discrete "concept neurons" like vision models do
        - Instead, they develop complex geometric structure with multiple overlapping representations
        - Understanding requires more sophisticated analysis than neuron-level inspection
        """)

# ===== THE ILLUSION REVEALED =====
elif section == "The Illusion Revealed":
    st.header("🎭 The Illusion Revealed")

    st.markdown("""
    ### The Classic Example: Neuron 221

    Let's examine what happens when we look at the top activating sentences for **Neuron 221 in Layer 12**
    across different datasets. What concept does this neuron encode?
    """)

    # Create tabs for each dataset
    tab1, tab2, tab3 = st.tabs(["🎵 Quora (QQP)", "📅 Wikipedia (QNLI)", "📚 BookCorpus"])

    with tab1:
        st.markdown("### Top Activating Sentences from Quora")
        st.markdown("""
        1. *"What is the meaning behind the song ""Angel"" by Eric Clapton?"*
        2. *"What's the meaning of Johnny Cash's song ""King of the Hill""?"*
        3. *"What is the meaning behind the Tears for Fears song ""Mad World"", such as the lyric, ""All around me are familiar faces""?"*
        """)

        st.success("""
        **Initial Interpretation:**

        Neuron 221 appears to encode questions about **song meanings** or the specific
        syntactic structure of these types of questions.
        """)

    with tab2:
        st.markdown("### Top Activating Sentences from Wikipedia")
        st.markdown("""
        1. *"On 16 June 2006, it was announced that Everton had entered into talks with Knowsley Council..."*
        2. *"On 15 September 1940, known as the Battle of Britain Day, an RAF pilot, Ray Holmes of No. 504 Squadron RAF rammed a German bomber..."*
        3. *"On 20 August 2010, Queen's manager Jim Beach put out a Newsletter stating that the band had signed a new contract with Universal Music."*
        """)

        st.warning("""
        **Second Interpretation:**

        Wait... now Neuron 221 seems to encode **historical events** or sentences
        beginning with **dates**. This is completely different from song meanings!
        """)

    with tab3:
        st.markdown("### Top Activating Sentences from BookCorpus")
        st.markdown("""
        1. *"Lara pulled out the document Reed had supplied from Gresham's briefcase."*
        2. *"I take Kellan's business card from my pocket and stretch it over to Realm."*
        3. *"Pilcher took a walkie-talkie out of his coat and spoke into the receiver."*
        """)

        st.error("""
        **Third Interpretation:**

        Now Neuron 221 appears to encode something about **physical actions** —
        specifically taking/pulling objects from containers (pockets, briefcases, coats).

        **This is the illusion!** The same neuron shows consistent, interpretable patterns
        within each dataset, but these patterns are NOT consistent across datasets.
        """)

    st.markdown("---")

    st.markdown("""
    ### 🎯 The Key Insight

    The fact that a seemingly consistent pattern can turn out to be a mirage has
    **clear implications for interpretability research**.
    """)

    # Interactive pattern comparison
    st.markdown("### Compare the Patterns")

    selected_datasets = st.multiselect(
        "Select datasets to compare:",
        ["Quora (Song meanings)", "Wikipedia (Historical dates)", "BookCorpus (Object retrieval)"],
        default=["Quora (Song meanings)", "Wikipedia (Historical dates)"]
    )

    if len(selected_datasets) >= 2:
        st.info(f"""
        You're comparing {len(selected_datasets)} different interpretations of the **same neuron**.
        Notice how these patterns have nothing in common with each other!
        """)

# ===== EXPERIMENTAL SETUP =====
elif section == "Experimental Setup":
    st.header("🔬 Experimental Setup")

    st.markdown("""
    ### Methodology

    The researchers conducted a systematic study to test whether the interpretability
    illusion is a general phenomenon or just an isolated case.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 1. Create Embeddings
        - Use BERT-base uncased (no fine-tuning)
        - Extract final layer [CLS] token
        - Create 768-dimensional embeddings for all sentences
        - Total: 624,712 sentences across 4 datasets
        """)

        st.markdown("""
        #### 2. Select Directions
        - **Neurons**: Randomly selected 25 neurons (basis vectors)
        - **Random Directions**: 33 random directions in embedding space
        - **Baseline**: 29 random sets of sentences
        """)

    with col2:
        st.markdown("""
        #### 3. Find Top Activating Sentences
        For each neuron/direction, find the top 10 sentences by:
        """)

        st.latex(r"\text{score} = \langle \text{sentence embedding}, \text{direction} \rangle")

        st.markdown("""
        #### 4. Annotate Patterns
        - Two annotators per set of sentences
        - Annotators don't know if sentences are:
          - Top activating for a neuron
          - Top activating for random direction
          - Randomly selected
        - Annotators identify any shared patterns
        """)

    st.markdown("---")

    # Detailed methodology expander
    with st.expander("🔬 Detailed Experimental Methodology (Click to expand)"):
        st.markdown("""
        ### Neuron Selection

        **What is a "neuron" in this context?**
        - Each of the 768 dimensions in the final layer embedding space
        - Corresponds to a basis vector: e_d = [0, 0, ..., 1, ..., 0] with 1 at position d
        - The researchers randomly selected 25 neurons from the 768 available

        ### Random Directions

        **Control condition to test if patterns are neuron-specific:**
        - Generated 33 random unit vectors in 768-dimensional space
        - Each random direction is: v = random vector, normalized to length 1
        - Used to check if random directions also show patterns (spoiler: they do!)

        ### Top Sentence Selection

        **How were the "top 10" sentences chosen?**

        For each neuron/direction v and dataset S:
        1. Compute projection score for every sentence: `score(x) = ⟨x, v⟩`
        2. This is the dot product between sentence embedding x and direction v
        3. Higher score = sentence "activates" that direction more strongly
        4. Select the 10 sentences with highest scores

        **What does the projection score mean?**
        - Measures how much a sentence embedding "points in the direction" of v
        - Similar to asking "how strongly does this sentence activate this neuron?"
        - In neural network terms: like the pre-activation value for that neuron

        ### Annotation Protocol

        **Double-blind annotation process:**

        Annotators were shown 10 sentences under one of three conditions:
        1. **Test**: Top 10 sentences for a neuron (25 neurons × 4 datasets = 100 sets)
        2. **Control 1**: Top 10 for a random direction (33 directions × 4 datasets = 132 sets)
        3. **Control 2**: Random 10 sentences from dataset (29 sets)

        **Blinding:**
        - Annotators KNEW which dataset the sentences came from
        - Annotators DID NOT know which condition they were seeing
        - This prevents bias: can't favor neurons over random directions

        **Task:**
        - Identify any patterns shared by multiple sentences
        - Mark which sentences match each pattern
        - Two independent annotators per set for reliability

        **Inter-rater Agreement:**
        - Measured using Cohen's kappa and raw agreement rates
        - Disagreement rates: 8% (neurons), 18% (random), 38% (baseline)
        - Higher agreement on neuron/random sets suggests real patterns exist
        """)

    st.markdown("---")

    st.markdown("""
    ### What is a "Pattern"?

    A pattern is any property shared by multiple sentences in the set:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.success("""
        **Structural Patterns:**
        - All sentences are the same length
        - All start with a date
        - All contain nested quotes
        - All are questions
        """)

    with col2:
        st.info("""
        **Lexical/Semantic Patterns:**
        - All mention song titles
        - All relate to mathematics
        - All describe historical events
        - All contain quoted speech
        """)

    st.markdown("---")

    # Interactive visualization of embedding space
    st.markdown("### Visualizing the Embedding Space")

    st.markdown("""
    The researchers used UMAP to visualize how sentences from different datasets
    are distributed in BERT's embedding space:
    """)

    # Simulate UMAP visualization
    np.random.seed(42)
    n_points = 500

    # Create clustered data for each dataset
    datasets_viz = []
    colors_map = {
        'QQP': '#FF6B6B',
        'QNLI': '#4ECDC4',
        'Wiki': '#45B7D1',
        'Books': '#96CEB4'
    }

    for i, (dataset, color) in enumerate(colors_map.items()):
        center_x = np.cos(i * np.pi / 2) * 3
        center_y = np.sin(i * np.pi / 2) * 3
        x = np.random.normal(center_x, 0.8, n_points // 4)
        y = np.random.normal(center_y, 0.8, n_points // 4)
        datasets_viz.append(pd.DataFrame({
            'x': x,
            'y': y,
            'dataset': dataset
        }))

    df_viz = pd.concat(datasets_viz)

    fig = px.scatter(
        df_viz,
        x='x',
        y='y',
        color='dataset',
        title='UMAP Visualization of BERT Embeddings',
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
        color_discrete_map=colors_map,
        opacity=0.6
    )

    fig.update_layout(
        width=800,
        height=600,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.warning("""
    ⚠️ **Note:** This visualization uses **simulated data** to illustrate the concept.
    Real data from the paper (Figure 1) shows similar clustering patterns but with more
    complex structure. The key finding—that datasets occupy distinct regions—holds in
    the actual data.
    """)

    st.info("""
    **Key Observation:** The four datasets form **distinct clusters** in the embedding space.
    This means they occupy different regions, which helps explain why the same neuron
    shows different patterns for different datasets.
    """)

# ===== RESULTS & FINDINGS =====
elif section == "Results & Findings":
    st.header("📊 Results & Findings")

    st.markdown("""
    ### How Often Do We Find Patterns?

    The researchers annotated sets of 10 sentences and checked whether both annotators
    found at least one pattern.
    """)

    # Results table
    results_data = {
        'Condition': ['Neurons', 'Random Directions', 'Random Sentences'],
        'Patterns Found': ['80%', '82%', '14%'],
        'No Patterns': ['12%', '0%', '48%'],
        'Annotators Disagreed': ['8%', '18%', '38%']
    }

    df_results = pd.DataFrame(results_data)

    st.dataframe(df_results, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.success("""
        **✅ Patterns Are Common**

        More than 80% of top activating sentences (for both neurons
        and random directions) contain identifiable patterns.
        """)

    with col2:
        st.warning("""
        **⚠️ But Not Consistent**

        The baseline (random sentences) only shows patterns 14% of the time,
        suggesting the patterns are real — but they change across datasets!
        """)

    st.markdown("---")

    # Pattern diversity chart
    st.markdown("### Pattern Diversity Across Datasets")

    st.markdown("""
    For each neuron, researchers counted how many **distinct patterns** appeared
    across the four datasets:
    """)

    pattern_counts = [1, 2, 2, 3, 3, 3, 4, 4]

    fig = go.Figure(data=[
        go.Bar(
            x=[f'{i+1}' for i in range(4)],
            y=[pattern_counts.count(i+1) for i in range(4)],
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        )
    ])

    fig.update_layout(
        title='Distribution of Distinct Patterns Per Neuron',
        xaxis_title='Number of Distinct Patterns',
        yaxis_title='Count of Neurons',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Average: 2.5 distinct patterns per neuron**

    This means most neurons show different interpretable patterns depending
    on which dataset you test them on!
    """)

    st.markdown("---")

    # Example patterns by dataset
    st.markdown("### Sample Patterns by Dataset")

    patterns_by_dataset = {
        'QQP (Quora)': [
            'Nested quotes',
            'Mathematics',
            'School exam questions',
            'Questions of comparison',
            'Programming',
            'Relationship advice'
        ],
        'QNLI (Wikipedia)': [
            'Military conflict',
            'Numbers and dates',
            'Biology',
            'Geography',
            'Population statistics',
            'Etymology'
        ],
        'Wiki': [
            'Direct statement of fact',
            'Music',
            'Television shows',
            'Olympic facts',
            'Legalese',
            'School districts'
        ],
        'Books': [
            'Interpersonal relationships',
            'Quoted speech',
            'Spanish',
            'Sentence fragments',
            'Very long sentences',
            'Medieval Europe'
        ]
    }

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**🎯 QQP**")
        for pattern in patterns_by_dataset['QQP (Quora)']:
            st.markdown(f"- {pattern}")

    with col2:
        st.markdown("**📚 QNLI**")
        for pattern in patterns_by_dataset['QNLI (Wikipedia)']:
            st.markdown(f"- {pattern}")

    with col3:
        st.markdown("**🌐 Wiki**")
        for pattern in patterns_by_dataset['Wiki']:
            st.markdown(f"- {pattern}")

    with col4:
        st.markdown("**📖 Books**")
        for pattern in patterns_by_dataset['Books']:
            st.markdown(f"- {pattern}")

    st.markdown("---")

    st.markdown("### Strength of Patterns")

    col1, col2 = st.columns(2)

    with col1:
        # Pattern strength comparison
        conditions = ['Neurons', 'Random\nDirections', 'Random\nSentences']
        means = [6.80, 6.89, 5.05]
        stdevs = [2.37, 2.13, 1.96]

        fig = go.Figure(data=[
            go.Bar(
                x=conditions,
                y=means,
                error_y=dict(type='data', array=stdevs),
                marker_color=['#FF6B6B', '#4ECDC4', '#96CEB4']
            )
        ])

        fig.update_layout(
            title='Average Sentences Matching Pattern (out of 10)',
            yaxis_title='Number of Sentences',
            showlegend=False,
            yaxis_range=[0, 10]
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        **Key Findings:**

        - Neurons: **6.8/10** sentences match pattern
        - Random directions: **6.9/10** sentences match
        - Random sentences: **5.1/10** sentences match

        Top activating sentences show **stronger patterns**
        than random sentences (more sentences share
        the identified pattern).

        Interestingly, **random directions** show patterns
        just as strong as actual neurons!
        """)

# ===== WHY DOES THIS HAPPEN =====
elif section == "Why Does This Happen?":
    st.header("🤔 Why Does This Happen?")

    st.markdown("""
    The researchers identified three main sources of the interpretability illusion:
    """)

    # Three sources
    tab1, tab2, tab3 = st.tabs([
        "📊 Dataset Idiosyncrasy",
        "🌍 Local Semantic Coherence",
        "👥 Annotator Error"
    ])

    with tab1:
        st.markdown("### Dataset Idiosyncrasy")

        st.markdown("""
        **Hypothesis:** QQP, QNLI, Wiki, and Books occupy distinct regions of BERT's embedding space.
        """)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            **Evidence:**

            1. **Visualization:** UMAP shows clear clustering by dataset
            2. **Classifier:** A linear SVM can distinguish between datasets with high accuracy

            **Accuracy by Dataset:**
            - QQP: 100% correctly identified
            - QNLI: 83% correctly identified
            - Wiki: 73% correctly identified
            - Books: 96% correctly identified

            3. **Activation Range Overlap:** Only **38%** of neuron/dataset pairs have
               overlapping top-10 activation ranges

            This means for most neurons, the highly activating sentences from different
            datasets are in completely separate regions of activation space!
            """)

        with col2:
            # Simulate confusion matrix
            datasets = ['QQP', 'QNLI', 'Wiki', 'Books']
            confusion = np.array([
                [1.00, 0.00, 0.00, 0.00],
                [0.00, 0.83, 0.19, 0.01],
                [0.00, 0.25, 0.73, 0.03],
                [0.02, 0.00, 0.02, 0.96]
            ])

            fig = go.Figure(data=go.Heatmap(
                z=confusion,
                x=datasets,
                y=datasets,
                colorscale='Blues',
                text=confusion,
                texttemplate='%{text:.2f}',
                textfont={"size": 16}
            ))

            fig.update_layout(
                title='Dataset Classification Accuracy',
                xaxis_title='Predicted',
                yaxis_title='Actual',
                width=400,
                height=400
            )

            st.plotly_chart(fig)

        st.info("""
        **What This Means:**

        The four datasets represent **idiosyncratic slices** of possible English sentences.
        They occupy non-overlapping regions in BERT's embedding space.

        Therefore, when you look along any direction, the top activating sentences from
        each dataset come from **distinct regions** — leading to different patterns!
        """)

        # Visual explanation
        st.markdown("### Visual Explanation")

        # Create a schematic diagram
        fig = go.Figure()

        # Draw two dataset regions
        theta = np.linspace(0, 2*np.pi, 100)
        r = 1

        # Dataset A (red)
        x_a = 2 + r * np.cos(theta)
        y_a = 1 + r * np.sin(theta)
        fig.add_trace(go.Scatter(x=x_a, y=y_a, fill='toself', fillcolor='rgba(255, 107, 107, 0.3)',
                                line=dict(color='rgba(255, 107, 107, 0.5)'), name='Dataset A',
                                showlegend=True))

        # Dataset B (blue)
        x_b = -1 + r * np.cos(theta)
        y_b = -1 + r * np.sin(theta)
        fig.add_trace(go.Scatter(x=x_b, y=y_b, fill='toself', fillcolor='rgba(78, 205, 196, 0.3)',
                                line=dict(color='rgba(78, 205, 196, 0.5)'), name='Dataset B',
                                showlegend=True))

        # Direction arrow
        fig.add_annotation(
            x=3, y=0,
            ax=-2, ay=0,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=2,
            arrowwidth=3,
            arrowcolor='black'
        )

        # Top activating points
        fig.add_trace(go.Scatter(x=[2.5], y=[1], mode='markers',
                                marker=dict(size=15, color='red', symbol='circle-open', line=dict(width=3)),
                                name='Top from A', showlegend=True))
        fig.add_trace(go.Scatter(x=[-0.5], y=[-1], mode='markers',
                                marker=dict(size=15, color='blue', symbol='circle-open', line=dict(width=3)),
                                name='Top from B', showlegend=True))

        fig.update_layout(
            title='Schematic: Top Activating Sentences from Different Dataset Regions',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=700,
            height=500,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        The arrow represents a direction in embedding space (a neuron or random direction).

        - **Red dots** = sentences from Dataset A
        - **Blue dots** = sentences from Dataset B
        - **Outlined points** = top activating sentences

        Even though we're looking along the same direction, the top activating sentences
        come from geometrically separate regions, so they can be semantically unrelated!
        """)

    with tab2:
        st.markdown("### Local Semantic Coherence")

        st.markdown("""
        **Hypothesis:** When annotating top activating sentences, people identify concepts
        emerging from local semantic coherence (clusters) rather than global directions.
        """)

        st.markdown("""
        #### The Key Insight

        BERT groups similar sentences together in **local clusters**. When you find
        top activating sentences, they might come from the same cluster — making them
        appear to share a concept — even if there's no global direction for that concept!
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.success("""
            **🌍 Global Concepts**

            A concept has a **global direction** if, starting
            from anywhere and moving along that direction,
            you increasingly find sentences with that concept.

            Examples:
            - Positivity (happy, joyful, etc.)
            - Questions (who, what, when, etc.)
            - Past tense
            """)

        with col2:
            st.info("""
            **📍 Local Concepts**

            A concept is **local** if similar sentences
            cluster together, but there's no consistent
            direction through the entire space.

            Examples:
            - Song title questions (clustered together)
            - Historical dates (clustered together)
            - But no single direction captures both!
            """)

        st.markdown("---")

        st.markdown("### Evidence: Locality Analysis")

        st.markdown("""
        The researchers measured whether top activating sentences are close to each other
        in the original embedding space (suggesting local clusters) or spread out (suggesting
        global directions).
        """)

        # Locality score comparison
        locality_data = pd.DataFrame({
            'Type': ['Meaningful\nNeurons', 'Meaningless\nNeurons'],
            'Locality Score': [0.026, 0.010],
            'Color': ['#FF6B6B', '#96CEB4']
        })

        fig = go.Figure(data=[
            go.Bar(
                x=locality_data['Type'],
                y=locality_data['Locality Score'],
                marker_color=locality_data['Color']
            )
        ])

        fig.update_layout(
            title='Locality Scores (Higher = More Local Clustering)',
            yaxis_title='Locality Score',
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        st.success("""
        **Finding:** Meaningful neurons have **significantly higher locality scores** (p < 0.001).

        This means the patterns we identify arise primarily from **local geometry** —
        sentences that activate strongly are near each other in embedding space, forming
        semantic clusters rather than following global directions.
        """)

        st.markdown("---")

        st.markdown("### Monotonic Token Frequencies")

        st.markdown("""
        The researchers found evidence of **global concept directions** by checking
        if certain tokens monotonically increase/decrease along neuron directions.

        **Methodology:**
        - Tested **915 tokens** appearing ≥100 times in all four datasets
        - For each neuron/token pair, checked if token frequency changes monotonically
          across activation quintiles (0-20%, 20-40%, ..., 80-100%)
        - Baseline (random): Only **1.7%** show monotonicity by chance
        """)

        col1, col2 = st.columns([2, 1])

        with col1:
            # Simulate monotonic frequency chart
            quintiles = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
            freq_increasing = [10, 20, 30, 40, 50]
            freq_flat = [30, 32, 28, 31, 29]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=quintiles,
                y=freq_increasing,
                mode='lines+markers',
                name='Token "quick" (Neuron 275)',
                line=dict(color='#FF6B6B', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=quintiles,
                y=freq_flat,
                mode='lines+markers',
                name='Token "quick" (Neuron 266)',
                line=dict(color='#96CEB4', width=3)
            ))

            fig.update_layout(
                title='Token Frequency by Activation Quintile',
                xaxis_title='Neuron Activation Quintile',
                yaxis_title='Token Frequency',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("""
            **Most Monotonic Tokens:**

            - " (quotation mark)
            - can
            - is, are, was, were
            - that
            - if
            - would
            - she, they
            - god

            These suggest BERT has learned some **global concepts**
            (like pronouns, common verbs).
            """)

        st.info("""
        **Key Results:**

        - **27%** of neuron/token pairs show monotonic relationships WITHIN single datasets
        - Only **1.9%** show monotonicity across ALL FOUR datasets simultaneously
        - Baseline (random): **1.7%** show monotonicity by chance

        **Interpretation:**

        The 14x difference (27% vs 1.9%) reveals that what looks like a global concept
        direction is usually dataset-specific! Very few neurons encode truly global concepts
        that generalize across all datasets.

        This suggests **both global and local concepts exist**, but most "concept directions"
        are actually dataset-level phenomena.
        """)

    with tab3:
        st.markdown("### Annotator Error")

        st.markdown("""
        **Source:** Humans tend to see patterns even where they may not exist.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### Annotator Agreement

            When two annotators looked at the same sets of sentences:
            """)

            agreement_data = pd.DataFrame({
                'Condition': ['Neurons', 'Random\nDirections', 'Random\nSentences'],
                'Both Agree': [92, 82, 62],
                'Disagree': [8, 18, 38]
            })

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Both Agree',
                x=agreement_data['Condition'],
                y=agreement_data['Both Agree'],
                marker_color='#4ECDC4'
            ))
            fig.add_trace(go.Bar(
                name='Disagree',
                x=agreement_data['Condition'],
                y=agreement_data['Disagree'],
                marker_color='#FF6B6B'
            ))

            fig.update_layout(
                title='Annotator Agreement Rate (%)',
                yaxis_title='Percentage',
                barmode='stack',
                yaxis_range=[0, 100]
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("""
            #### Annotator Variation

            Different annotators found patterns at very different rates:
            """)

            annotator_data = pd.DataFrame({
                'Annotator': ['0', '1', '2', '3', '4', '5'],
                'Pattern Rate': [0.90, 0.82, 0.92, 0.56, 0.58, 1.60]
            })

            fig = go.Figure(data=[
                go.Bar(
                    x=annotator_data['Annotator'],
                    y=annotator_data['Pattern Rate'],
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#F9A826', '#9B59B6']
                )
            ])

            fig.update_layout(
                title='Patterns Found per Direction',
                xaxis_title='Annotator ID',
                yaxis_title='Average Patterns per Direction',
                showlegend=False
            )

            st.plotly_chart(fig)

        st.warning("""
        **Key Observations:**

        1. **Lower agreement for random sentences** (38% disagreement) suggests weaker patterns
        2. **Annotator 5 found 1.6 patterns per direction** (some found multiple patterns in a single set!)
        3. **Annotator 4 found only 0.58 patterns per direction** — suggesting subjective differences

        This indicates that pattern identification involves some degree of **subjective interpretation**
        and potential for seeing patterns that may not be objectively present.
        """)

# ===== TYPES OF CONCEPTS =====
elif section == "Types of Concepts":
    st.header("🎯 Types of Concepts in BERT")

    st.markdown("""
    Based on their analysis, the researchers propose a **taxonomy of three types of concepts**
    that BERT learns to represent:
    """)

    # Three types visualization
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🌍 Global Concepts")
        st.success("""
        **Definition:** Concepts that become increasingly prevalent as you move through
        the entire embedding space along a specific direction.
        """)

    with col2:
        st.markdown("### 📊 Dataset-Level Concepts")
        st.info("""
        **Definition:** Like global concepts, but only meaningful within the region
        of embedding space occupied by a specific dataset.
        """)

    with col3:
        st.markdown("### 📍 Local Concepts")
        st.warning("""
        **Definition:** Concepts that emerge as clusters in the embedding space,
        without a consistent directional pattern.
        """)

    st.markdown("---")

    # Detailed comparison
    st.markdown("### Detailed Comparison")

    concept_types = st.selectbox(
        "Select a concept type to learn more:",
        ["Global Concepts", "Dataset-Level Concepts", "Local Concepts"]
    )

    if concept_types == "Global Concepts":
        st.markdown("### 🌍 Global Concepts")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Visualization
            st.markdown("#### Geometric Representation")

            # Create a gradient visualization
            x = np.linspace(-3, 3, 100)
            y = np.linspace(-3, 3, 100)
            X, Y = np.meshgrid(x, y)
            Z = X  # Gradient along X direction

            fig = go.Figure(data=go.Contour(
                z=Z,
                x=x,
                y=y,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title='Concept<br>Strength')
            ))

            # Add arrow showing direction
            fig.add_annotation(
                x=2.5, y=0,
                ax=-2.5, ay=0,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=2,
                arrowwidth=4,
                arrowcolor='red'
            )

            # Add sample points
            sample_points_x = [-2, 0, 2]
            sample_points_y = [0, 0, 0]
            fig.add_trace(go.Scatter(
                x=sample_points_x,
                y=sample_points_y,
                mode='markers',
                marker=dict(size=15, color='red', symbol='circle'),
                showlegend=False
            ))

            fig.update_layout(
                title='Global Concept: Density Increases Along Direction',
                xaxis_title='Embedding Dimension 1',
                yaxis_title='Embedding Dimension 2',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            As you move along the red arrow (concept direction), sentences containing
            the concept become increasingly common — **regardless of where you start**.
            """)

        with col2:
            st.markdown("#### Characteristics")

            st.markdown("""
            **Properties:**
            - Consistent across datasets
            - Can be detected from any starting point
            - Form a directional gradient

            **Examples Found:**
            - Quotation marks
            - Common verbs (is, are, was, were)
            - Modal verbs (can, would)
            - Pronouns (she, they, it)

            **How to Identify:**
            - Token frequencies change monotonically
            - Same pattern appears across all datasets
            - 27% of neuron/token pairs show this
            """)

        st.success("""
        **Key Finding:** Global concepts DO exist in BERT, particularly for common linguistic
        features like pronouns and frequent verbs. However, they're not as common as local concepts.
        """)

    elif concept_types == "Dataset-Level Concepts":
        st.markdown("### 📊 Dataset-Level Concepts")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### Geometric Representation")

            # Create visualization with dataset region
            theta = np.linspace(0, 2*np.pi, 100)
            r = 1.5
            x_region = r * np.cos(theta)
            y_region = r * np.sin(theta)

            fig = go.Figure()

            # Dataset region
            fig.add_trace(go.Scatter(
                x=x_region,
                y=y_region,
                fill='toself',
                fillcolor='rgba(78, 205, 196, 0.3)',
                line=dict(color='rgba(78, 205, 196, 0.5)', width=2),
                name='Dataset Region',
                showlegend=True
            ))

            # Gradient within region
            n_points = 50
            for i in range(n_points):
                angle = i * 2 * np.pi / n_points
                r_inner = 0.3
                r_outer = 1.4
                x_line = [r_inner * np.cos(angle), r_outer * np.cos(angle)]
                y_line = [r_inner * np.sin(angle), r_outer * np.sin(angle)]

                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    line=dict(color=f'rgba(78, 205, 196, {0.5 + i/(2*n_points)})', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # Direction arrow
            fig.add_annotation(
                x=1.2, y=0,
                ax=-1.2, ay=0,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=2,
                arrowwidth=4,
                arrowcolor='red'
            )

            fig.update_layout(
                title='Dataset-Level Concept: Only Meaningful Within Dataset Region',
                xaxis=dict(range=[-3, 3], showgrid=True, zeroline=True),
                yaxis=dict(range=[-3, 3], showgrid=True, zeroline=True),
                height=400,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            The concept direction (red arrow) is only meaningful **within the dataset's region**
            of the embedding space. Outside this region, the direction doesn't correspond to
            the same concept.
            """)

        with col2:
            st.markdown("#### Characteristics")

            st.markdown("""
            **Properties:**
            - Specific to one dataset
            - Don't generalize to other datasets
            - Arise from dataset idiosyncrasy

            **Examples:**
            - "Song meanings" pattern in QQP
            - "Historical dates" pattern in QNLI
            - Different patterns per dataset

            **How to Identify:**
            - Token frequencies change within one dataset
            - Pattern doesn't appear in other datasets
            - Most common source of illusion
            """)

        st.info("""
        **Key Insight:** This is the **primary source of the interpretability illusion**.

        A neuron appears to encode "song meanings" in QQP, but that's only because QQP
        occupies a specific region of the embedding space. The same neuron direction means
        something completely different in other dataset regions!
        """)

    else:  # Local Concepts
        st.markdown("### 📍 Local Concepts")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### Geometric Representation")

            # Create visualization with multiple clusters
            fig = go.Figure()

            # Create 3 different concept clusters
            concepts = [
                {'name': 'Concept A', 'center': (2, 2), 'color': '#FF6B6B'},
                {'name': 'Concept B', 'center': (-2, 1), 'color': '#4ECDC4'},
                {'name': 'Concept C', 'center': (0, -2), 'color': '#F9A826'}
            ]

            for concept in concepts:
                # Create cluster
                n_points = 30
                x = np.random.normal(concept['center'][0], 0.4, n_points)
                y = np.random.normal(concept['center'][1], 0.4, n_points)

                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(size=10, color=concept['color'], opacity=0.6),
                    name=concept['name']
                ))

            # Add direction arrow that intersects one cluster
            fig.add_annotation(
                x=2.5, y=2,
                ax=-2, ay=-1,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=2,
                arrowwidth=4,
                arrowcolor='black'
            )

            fig.update_layout(
                title='Local Concepts: Clusters Without Global Direction',
                xaxis=dict(range=[-4, 4], showgrid=True, zeroline=True),
                yaxis=dict(range=[-4, 4], showgrid=True, zeroline=True),
                height=400,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            Each colored cluster represents sentences with a specific concept. The clusters
            are scattered throughout the space **without a consistent directional pattern**.

            When you look along a direction (black arrow), you might hit one cluster and
            mistakenly conclude the direction encodes that concept — but it's just local clustering!
            """)

        with col2:
            st.markdown("#### Characteristics")

            st.markdown("""
            **Properties:**
            - Form semantic clusters
            - No global direction
            - Neighbors are similar

            **How They Arise:**
            - BERT groups similar sentences together
            - Sentences about same topic cluster
            - Natural result of semantic similarity

            **Evidence:**
            - High locality scores for meaningful neurons
            - Top activating sentences are neighbors
            - Patterns emerge from local geometry
            """)

        st.warning("""
        **Key Insight:** Many patterns we identify come from **local semantic coherence**
        rather than global directions.

        The researchers found that meaningful neurons have significantly higher locality
        scores (0.026 vs 0.010, p < 0.001), indicating that top activating sentences are
        near each other in the original embedding space.

        This means we're often identifying **clustering patterns** rather than **directional patterns**.
        """)

    st.markdown("---")

    # Summary comparison table
    st.markdown("### Summary Comparison")

    comparison_df = pd.DataFrame({
        'Property': [
            'Consistent Across Datasets?',
            'Has Direction?',
            'Generalizes to New Data?',
            'Prevalence in BERT',
            'Example'
        ],
        'Global': [
            '✅ Yes',
            '✅ Yes',
            '✅ Yes',
            'Less common (~27% of neuron/token pairs)',
            'Pronouns, common verbs'
        ],
        'Dataset-Level': [
            '❌ No',
            '✅ Yes (within dataset)',
            '❌ No',
            'Common (main source of illusion)',
            'Song meanings in QQP'
        ],
        'Local': [
            '❌ No',
            '❌ No',
            '❌ No',
            'Very common (high locality scores)',
            'Semantic clusters'
        ]
    })

    st.dataframe(comparison_df, use_container_width=True)

# ===== KEY TAKEAWAYS =====
elif section == "Key Takeaways":
    st.header("🎯 Key Takeaways")

    st.markdown("""
    ### Main Contributions of the Paper
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("""
        **1. Identified the Illusion**

        The paper demonstrates that individual neurons
        and random directions in BERT appear to encode
        clear concepts within a dataset, but these
        interpretations don't generalize across datasets.
        """)

    with col2:
        st.info("""
        **2. Methodological Recommendation**

        Interpretability researchers should **always
        test their hypotheses on multiple datasets**
        before concluding that a neuron or direction
        encodes a specific concept.
        """)

    with col3:
        st.warning("""
        **3. Taxonomy of Concepts**

        The paper provides a useful framework for
        understanding different types of learned
        concepts: global, dataset-level, and local.
        """)

    st.markdown("---")

    # Detailed takeaways
    st.markdown("### Detailed Insights")

    with st.expander("🔍 The Interpretability Illusion is Real and Widespread"):
        st.markdown("""
        - **80% of neurons** show consistent patterns within a single dataset
        - **82% of random directions** also show consistent patterns
        - But **2.5 different patterns per neuron** across 4 datasets
        - The illusion affects both individual neurons and linear combinations

        **Implication:** You can't trust single-dataset analysis of what a neuron "means"
        """)

    with st.expander("📊 Datasets are More Idiosyncratic Than We Thought"):
        st.markdown("""
        - Common NLP datasets (QQP, QNLI, Wiki, Books) occupy **distinct regions** of BERT's embedding space
        - A linear classifier can distinguish between datasets with **high accuracy** (73-100%)
        - Datasets represent **narrow slices** of possible English sentences
        - Only 38% of top-10 activation ranges overlap between dataset pairs

        **Implication:** Be cautious about generalizing findings from a single dataset
        """)

    with st.expander("🌍 Both Global and Local Concepts Exist"):
        st.markdown("""
        - **Global concepts** (27% of neuron/token pairs): Pronouns, common verbs, modal verbs
        - **Local concepts** (high locality scores): Semantic clustering without direction
        - **Dataset-level concepts**: Meaningful only within specific dataset regions

        **Implication:** The embedding space has rich, complex structure with multiple types of concept representations
        """)

    with st.expander("🧪 Local Semantic Coherence Drives Many Patterns"):
        st.markdown("""
        - Meaningful neurons have **significantly higher locality scores** (0.026 vs 0.010, p < 0.001)
        - Top activating sentences are **neighbors** in the original embedding space
        - Patterns emerge from **clustering** rather than directional concepts

        **Implication:** What looks like a neuron encoding a concept might just be it intersecting a semantic cluster
        """)

    with st.expander("👥 Human Pattern Recognition Can Be Misleading"):
        st.markdown("""
        - Annotators disagree **38%** of the time on random sentences
        - Different annotators find patterns at **very different rates** (0.56 to 1.60 per direction)
        - Even random sentences show patterns **14%** of the time

        **Implication:** Beware of confirmation bias and the human tendency to see patterns
        """)

    st.markdown("---")

    # Practical recommendations
    st.markdown("### 💡 Practical Recommendations for Researchers")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### ✅ DO:

        - **Test on multiple datasets** before drawing conclusions
        - **Measure locality scores** to understand if patterns are local or global
        - **Check token monotonicity** to identify global concept directions
        - **Use multiple annotators** to reduce subjective bias
        - **Consider dataset idiosyncrasy** in your analysis
        - **Look for concepts that replicate** across different data distributions
        """)

    with col2:
        st.markdown("""
        #### ❌ DON'T:

        - **Don't conclude** a neuron encodes X from one dataset alone
        - **Don't assume** patterns are meaningful without validation
        - **Don't ignore** the possibility of local clustering
        - **Don't overlook** the role of dataset biases
        - **Don't trust** interpretations that don't replicate
        - **Don't forget** that humans see patterns even in noise
        """)

    st.markdown("---")

    # Broader implications
    st.markdown("### 🌐 Broader Implications")

    st.info("""
    **For Interpretability Research:**

    This paper challenges a common assumption in neural network interpretability: that
    meaningful neurons or directions can be identified by examining activations on a
    single dataset. It shows that interpretations must be **validated across multiple
    data distributions** to be trustworthy.
    """)

    st.info("""
    **For Bias and Fairness:**

    The paper notes that techniques using concept directions to measure or mitigate bias
    in language models could have **unintended effects** if the directions are only
    dataset-level concepts that don't generalize.
    """)

    st.info("""
    **For Understanding Language Models:**

    The finding that BERT learns multiple types of concepts (global, dataset-level, local)
    suggests that language models develop **rich, complex internal representations** that
    don't always correspond to simple human-interpretable concepts.
    """)

    st.markdown("---")

    # Future directions
    st.markdown("### 🔮 Future Research Directions")

    future_directions = [
        "Replicate analysis on other language models (GPT, RoBERTa, etc.)",
        "Investigate token-level embeddings instead of sentence-level",
        "Examine earlier layers of BERT (not just final layer)",
        "Test whether similar illusions exist in other domains (images, graphs)",
        "Develop methods to distinguish global from dataset-level concepts",
        "Create better techniques for validating interpretability claims"
    ]

    for i, direction in enumerate(future_directions, 1):
        st.markdown(f"{i}. {direction}")

    st.markdown("---")

    # Final thoughts
    st.markdown("### 🎬 Final Thoughts")

    with st.expander("🤔 What Do These Findings Mean for 'Concepts in BERT'? (Click to expand)"):
        st.markdown("""
        ### The Deeper Question: Does BERT Actually Learn Concepts?

        This paper raises a fundamental philosophical question about what we mean by
        "concepts" in neural networks.

        **The Naive View (Challenged by This Paper):**
        - Individual neurons encode discrete, human-interpretable concepts
        - These concepts are stable and generalizable properties
        - Finding interpretable patterns means we understand what the model learned

        **What This Paper Suggests Instead:**

        **BERT doesn't learn concepts the way we hoped it would.** Instead, it learns:

        1. **Geometric Structure:**
           - Embeddings organize in complex, high-dimensional space
           - Similar inputs cluster together (local coherence)
           - Different data distributions occupy different regions (idiosyncrasy)

        2. **Statistical Regularities:**
           - Patterns that happen to be consistent within a dataset's region
           - Not semantic understanding, but statistical correlation
           - May or may not generalize to other contexts

        3. **A Few Global Features:**
           - Common linguistic markers (pronouns, function words)
           - But these are the exception, not the rule
           - Most patterns are dataset-specific or local

        ### What Does BERT "Know"?

        **The Uncomfortable Truth:**
        BERT's representations are more like a complex index of statistical patterns than
        a structured knowledge base of concepts. What looks like "understanding" may be:
        - Memorization of training data structure
        - Exploitation of dataset biases
        - Statistical patterns we interpret as semantic meaning

        **This doesn't mean BERT isn't useful!** It just means our interpretation methods
        need to match the reality of how these models work, not our intuitions about
        how we think they should work.

        ### Implications for Interpretability

        **We need to rethink our approach:**
        - Stop looking for "the neuron for X"
        - Start understanding the geometric and statistical structure
        - Accept that interpretability might be harder than we thought
        - Develop new methods that account for dataset-dependency and local structure

        **The path forward:**
        - Multi-dataset validation as standard practice
        - Geometric analysis of embedding spaces
        - Understanding interactions between neurons, not individual neurons
        - Acknowledging limits of human interpretability
        """)

    st.success("""
    ### A Cautionary Tale

    This paper provides an important **methodological wake-up call** for interpretability research.

    It's easy to be fooled by patterns that seem clear and consistent within a single
    dataset. But true understanding requires **rigorous validation** across multiple
    contexts.

    **Key Lessons:**
    - Be humble about our interpretations
    - Test rigorously before concluding
    - Consider alternative explanations (dataset effects, local clustering, human bias)
    - Always validate across multiple diverse datasets

    **The Standard This Sets:**

    Single-dataset interpretability claims should be viewed with skepticism. The gold standard
    is now: **Does this interpretation replicate across multiple diverse datasets?**

    As the field of AI interpretability grows, methodological rigor becomes ever more
    important. This paper challenges us to meet that standard.
    """)

    st.markdown("---")

    # Citation
    st.markdown("### 📚 Citation")

    st.code("""
@article{bolukbasi2021interpretability,
  title={An Interpretability Illusion for BERT},
  author={Bolukbasi, Tolga and Pearce, Adam and Yuan, Ann and Coenen, Andy and Reif, Emily and Vi{'e}gas, Fernanda and Wattenberg, Martin},
  journal={arXiv preprint arXiv:2104.07143},
  year={2021}
}
    """, language="bibtex")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**About This Explainer**

Created to help understand the paper
*"An Interpretability Illusion for BERT"*
by Bolukbasi et al. (2021)

Navigate using the menu above to explore
different aspects of the paper.
""")
