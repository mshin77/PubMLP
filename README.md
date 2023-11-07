---
license: 
- mit
language:
- en
metrics:
- accuracy
library_name: transformers
pipeline_tag: text-classification
tags:
- BERT
- PyTorch
- education
---
# PubMLP: Publication Classifier with BERT Tokenizer and Multilayer Perceptron

[Hugging Face](https://huggingface.co/mshin/PubMLP) \
[GitHub](https://github.com/mshin77/PubMLP)

This model demonstrates a publication classifier for bibliometric data exported from the Web of Science. The `tabular data` includes text (e.g., abstracts), categorical, and numeric (e.g., publication year) features. The utilized functions include the BERT tokenizer for text preprocessing and an MLP (multilayer perceptron) for classifying whether the publication meets the researcher-designated inclusion criteria. 

Data preprocessing includes tokenizing text features using `BERT's tokenizer` and encoding categorical features as dummy variables (yes for 1 and no for 0). As a neural network, MLP consists of multiple layers, including input, hidden, and output layers. MLP classifier was trained using `Pytorch's torch library` on 1,798 publications (80% for training, 10% for validation, and 10% for testing and evaluated its performance on validation and test datasets.

## Installation

To set up the project environment and install the necessary dependencies, run the following command:

pip install -r libraries.txt

## Documents 

[Python in RStudio by qmd file](https://mshin77.github.io/PubMLP/Python-RStudio-qmd) \
[Python in RStudio by Rmd file](https://mshin77.github.io/PubMLP/Python-RStudio-Rmd) \
[Python in VScode by ipynb file](https://mshin77.github.io/PubMLP/Python-VScode-ipynb) \
[Python in VScode by qmd flie](https://mshin77.github.io/PubMLP/Python-VScode-qmd)
