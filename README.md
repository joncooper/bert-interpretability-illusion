# bert-interpretability-illusion

Interactive explainer for the paper "An Interpretability Illusion for BERT" by Bolukbasi, Pearce, Yuan, et al. (2021)

## Overview

This repository contains an interactive Streamlit application that walks users through the key findings of the paper "An Interpretability Illusion for BERT". The paper demonstrates that individual neurons in BERT may appear to encode simple, interpretable concepts when examined on a single dataset, but these interpretations often fail to generalize across different datasets.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Explainer

```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## Features

The interactive explainer includes:

- **Introduction**: Overview of neural network interpretability challenges
- **The Illusion Revealed**: Interactive demonstration of Neuron 221's changing interpretations
- **Experimental Setup**: Detailed methodology and datasets used
- **Results & Findings**: Interactive charts showing pattern prevalence and diversity
- **Why Does This Happen?**: Exploration of dataset idiosyncrasy, local semantic coherence, and annotator error
- **Types of Concepts**: Taxonomy of global, dataset-level, and local concepts with visualizations
- **Key Takeaways**: Practical recommendations and broader implications

## Paper Reference

```bibtex
@article{bolukbasi2021interpretability,
  title={An Interpretability Illusion for BERT},
  author={Bolukbasi, Tolga and Pearce, Adam and Yuan, Ann and Coenen, Andy and Reif, Emily and Vi{'e}gas, Fernanda and Wattenberg, Martin},
  journal={arXiv preprint arXiv:2104.07143},
  year={2021}
}
```

## Files

- `streamlit_app.py`: Main interactive explainer application
- `2104.07143v1.pdf`: Original research paper
- `requirements.txt`: Python dependencies
