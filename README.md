---
license: mit
language: en
metrics: accuracy
library_name: transformers
pipeline_tag: text-classification
tags:
- BERT
- PyTorch
- education
---
# PubMLP: Automatic Publication Classifier Using MLP

Pre-trained model and code for PubMLP at the following locations:

- [Hugging Face](https://huggingface.co/mshin/PubMLP) \
- [GitHub](https://github.com/mshin77/PubMLP)

## Introduction
This projectC contains code for an automatic publication classifier for bibliometric data exported from the Web of Science (WOS). The tabular data includes text, categorical data, and numeric features. The utilized functions include the BERT (Bidirectional Encoder Representations from Transformers) tokenizer for text preprocessing and an MLP (multilayer perceptron) for classifying whether the publication meets the researcher-designated inclusion criteria.

Data preprocessing involves tokenizing text features using `BERT's tokenizer` and encoding categorical features as dummy variables (1 for yes and 0 for no). As a neural network, the MLP consists of multiple layers, including input, hidden, and output layers. The MLP classifier was trained using `PyTorch's torch library` on 1,798 publications (80% for training, 10% for validation, and 10% for testing) and evaluated for its performance on the validation and test datasets.

## Dataset
The dataset includes a subset of WOS bibliometric data. The data are based on publications between 1970 and 2023 and are focused on single-case research that utilizes technology.

## Project Structure

```
PubMLP/
├── data.csv
├── main.py
├── load.py
├── preprocess.py
├── train.py
├── plot.py
└── README.md
```
## Pipeline

The project consists of several components that are executed sequentially for the classification of publications:

1. **Load Data (main.py)**: Load and preprocess the initial data from `data.csv`.

2. **Preprocess Data (preprocess.py)**: Tokenize text, encode categorical features, and prepare data for training.

3. **Train Model (train.py)**: Train the MLP model using the preprocessed data.

4. **Plot Results (plot.py)**: Visualize training, validation, and test results.

## Usage

To run the pipeline, follow these steps:

1. Clone the repository:

   - In ![VSCode](https://img.shields.io/badge/VSCode-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white): Open VSCode, go to the "Source Control" tab, click "Clone Repository," and enter the repository URL.

   - In ![RStudio](https://img.shields.io/badge/RStudio-75AADB?style=for-the-badge&logo=rstudio&logoColor=white): Open RStudio, go to "File" > "New Project" > "Version Control" > "Git" and enter the repository URL.

2. Install the required dependencies:

   - In a terminal or command prompt (Bash shell):

     ```bash
     pip install -r requirements.txt
     ```

   This command installs the necessary Python dependencies listed in the `requirements.txt` file.

3. Execute the main script (runs the pipeline and generate outputs):

   - In a terminal or command prompt (Bash shell):

     ```bash
     Quarto preview main.qmd
     ```
