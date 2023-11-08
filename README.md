<!-- README.md is generated from README.Rmd. Please edit that file -->

## ‘PubMLP: Automatic Publication Classifier’

Pre-trained model and code for PubMLP at the following locations:

<img
src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg"
height="30" /> <https://huggingface.co/mshin/PubMLP> <img
src="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png"
height="30" /> <https://github.com/mshin77/PubMLP>

## Introduction

This project contains code for an automatic publication classifier for
bibliometric data exported from the
<span style="background-color: #C7CEEA">Web of Science (WOS)</span>. The
tabular data includes text, categorical data, and numeric features. The
utilized functions include the BERT (Bidirectional Encoder
Representations from Transformers) tokenizer for text preprocessing and
an MLP (multilayer perceptron) for classifying whether the publication
meets the researcher-designated inclusion criteria.

Data preprocessing involves tokenizing text features using
<span style="background-color: #B5EAD7">BERT’s tokenizer</span> and
encoding categorical features as dummy variables (1 for yes and 0 for
no). As a neural network, the MLP consists of multiple layers, including
input, hidden, and output layers. The MLP classifier was trained using
<span style="background-color: #B5EAD7">PyTorch’s torch library</span>
on 1,798 publications (80% for training, 10% for validation, and 10% for
testing) and evaluated for its performance on the validation and test
datasets.

## Dataset

The dataset includes a subset of WOS bibliometric data. The data are
based on publications between 1970 and 2023 and are focused on
<span style="background-color: #C7CEEA">single-case research</span> that
utilizes <span style="background-color: #C7CEEA">technology</span>.

## Pipeline

The project consists of several components that are executed
sequentially for the classification of publications:

1.  **Load Data (main.py)**: Load and preprocess the initial data from
    data.csv.

2.  **Preprocess Data (preprocess.py)**: Tokenize text, encode
    categorical features, and prepare data for training.

3.  **Train Model (train.py)**: Train the MLP model using the
    preprocessed data.

4.  **Plot Results (plot.py)**: Visualize training, validation, and test
    results.

## Usage

To run the pipeline, follow these optional steps :

1.  **Clone** the repository:

Open
<img src="https://code.visualstudio.com/assets/images/code-stable.png"
height="20" /> [VSCode](https://code.visualstudio.com/download), go to
the “Source Control” tab, click “Clone Repository,” and enter the
repository URL.

Open <img
src="https://www.rstudio.com/wp-content/uploads/2018/10/RStudio-Logo-Flat.png"
height="25" /> [RStudio](https://posit.co/downloads), go to “File” &gt;
“New Project” &gt; “Version Control” &gt; “Git” and enter the repository
URL.

1.  **Install** the required dependencies listed in the
    `requirements.txt` file.:

<!-- -->

    pip install -r requirements.txt

1.  **Execute** the main script to generate outputs:

<!-- -->

    Quarto preview main.qmd
