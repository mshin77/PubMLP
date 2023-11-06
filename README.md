# PubMLP: Publication Classifier with BERT Tokenizer and Multilayer Perceptron

This model demonstrates a publication classifier for bibliometric data exported from Web of Science. The tabular data includes text (e.g., abstracts), categorical, and numeric (e.g., publication year) features. The utilized functions include the BERT tokenizer for text preprocessing and an MLP (multilayer perceptron) for classifying whether the publication meets the researcher-designated inclusion criteria.

Data preprocessing includes tokenizing text features using BERT's tokenizer and encoding categorical features as dummy variables (yes for 1 and no for 0).

The MLP classifier is a neural network designed for text classification on tabular data. It consists of multiple layers, including input, hidden, and output layers.

The researcher has trained the MLP classifier on the preprocessed tabular data and evaluated its performance on a validation dataset.

## Installation

To set up the project environment and install the necessary dependencies, run the following command:

```bash
pip install -r libraries.txt