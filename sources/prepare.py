import pandas as pd
import numpy as np

# Define a function to extract country entities using NER
def extract_country_entities(text, ner_model):
    """
    Extracts country entities from the given text using the provided NER model.

    Args:
        text (str): The input text to extract country entities from.
        ner_model: The NER model used for entity recognition.

    Returns:
        str: A comma-separated string of country entities found in the text.
    """
    if pd.notna(text):  
        doc = ner_model(text)
        countries = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
        return ', '.join(countries)
    else:
        return ''

# Split the data for training, validation, and testing     
def split_data(data):
    """
    Splits the given data into train, validation, and test sets.

    Args:
        data (pandas.DataFrame): The input data to be split.

    Returns:
        train_df (pandas.DataFrame): The training set.
        valid_df (pandas.DataFrame): The validation set.
        test_df (pandas.DataFrame): The test set.
    """
    train_df, valid_df, test_df = np.split(
        data.sample(frac=1, random_state=42), [
            int(0.8 * len(data)), int(0.9 * len(data))
        ]
    )
    return train_df, valid_df, test_df