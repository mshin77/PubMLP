---
license: mit
language:
- en
metrics:
- accuracy
tags:
- BERT
- PyTorch
- Bibliometrics
- Single-case design
- Technology
- Custom MLP
---

**Pre-trained model and code for PubMLP are available at the following locations:**

[<img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" width="20"/> Hugging Face](https://huggingface.co/mshin/PubMLP) \
[<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"/> GitHub](https://github.com/mshin77/PubMLP)

## PubMLP: Automatic Publication Classifier

This project contains code for an automatic publication classifier for bibliometric data exported from the `Web of Science (WOS)`. The tabular data include text, categorical data, and numeric features. The utilized functions include the BERT (Bidirectional Encoder Representations from Transformers) tokenizer for text preprocessing and an MLP (multilayer perceptron) for classifying whether the publication meets the researcher-designated inclusion criteria.

Data preprocessing involves tokenizing text features using `BERT's tokenizer` and encoding categorical features as dummy variables (1 for yes and 0 for no). As a neural network, the MLP consists of multiple layers, including input, hidden, and output layers. The MLP classifier was trained using `PyTorch's torch library` on 1,798 publications (80% for training, 10% for validation, and 10% for testing) and evaluated for its performance on the validation and test datasets.

The dataset includes a subset of WOS bibliometric data. The data are based on publications between 1970 and 2023 and focus on `single-case research` that utilizes `technology`.

**Pipeline**

The project consists of several components executed sequentially for the classification of publications:

- **Prepare Data (prepare.py)**: Extract country entities and split data.
- **Preprocess Data (preprocess.py)**: Tokenize text, encode categorical features, and prepare data for training.
- **Train and Test the Model (model.py)**: Train and evaluate the MLP model using the preprocessed data.
- **Plot Results (plot.py)**: Visualize training, validation, and test results.

**Usage**

To run the pipeline, follow these optional steps:

**Clone** the repository:
  - [VSCode](https://code.visualstudio.com/download): Go to the "Source Control" tab, click "Clone Repository," and enter the repository URL.
  - [RStudio](https://posit.co/downloads): Go to "File" > "New Project" > "Version Control" > "Git" and enter the repository URL.

**Install** the required dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt 
```
**Execute** the main script to generate outputs:

```bash
Quarto preview main.qmd
```