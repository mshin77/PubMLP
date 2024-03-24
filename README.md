---
license: mit
language:
- en
metrics:
- accuracy
library_name: transformers
tags:
- single-case research
- bert
- mlp
---

# PubMLP: Automatic Publication Classifier

[<img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" width="20"/> Hugging Face](https://huggingface.co/mshin/PubMLP) | [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"/> GitHub](https://github.com/mshin77/PubMLP) | [<img src="https://assets.dryicons.com/uploads/icon/svg/5923/473dc604-c750-41f5-b394-1b9d1799ff06.svg" width="20"/> Documentation](https://mshin77.github.io/PubMLP)

This model is a fine-tuned version of [bert-base-uncased](https://huggingface.co/bert-base-uncased) on bibliometric data exported from the `Web of Science (WOS)`. The tabular data include text, categorical data, and numeric features. The utilized functions include the BERT (Bidirectional Encoder Representations from Transformers) embeddings and an MLP for classifying whether the publication meets the researcher-designated inclusion criteria.

## Model description

PubMLP is a PyTorch module for a Multilayer Perceptron (MLP) model with BERT embeddings. 

## Pipeline

The project consists of several components executed sequentially for the classification of publications:
- **Prepare Data (prepare.py)**: Extract country entities and split data.
- **Preprocess Data (preprocess.py)**: Tokenize combined text columns and normalize numeric data.
- **Train and Evaluate the Model (model.py)**: Train and evaluate the MLP model using the preprocessed data.
- **Plot Results (plot.py)**: Visualize results.

## Training and evaluation data

The dataset includes a subset of WOS bibliometric data. The data are based on publications between 1970 and 2023 and focus on `single-case research`.

### Framework versions

- pandas==2.1.2
- numpy>=1.21,<1.25  
- spacy==3.7.2
- torch==2.1.0
- scikit-learn==1.3.2
- transformers>=4.36.0
- matplotlib==3.8.1
