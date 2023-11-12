import pandas as pd
import spacy
import numpy as np
import pandas as pd
import spacy
import numpy as np

# Define a function to extract country entities using NER
def extract_country_entities(text, ner_model):
    """
    Extracts country entities from the given text using the provided NER model.

    Args:
        text (str): The text to extract country entities from.
        ner_model: The NER model to use for entity extraction.

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
    Splits the input data into training, validation, and test sets.

    Args:
        data (pandas.DataFrame): The input data to split.

    Returns:
        tuple: A tuple containing the training, validation, and test sets.
    """
    train_df, valid_df, test_df = np.split(
        data.sample(frac=1, random_state=42), [
            int(0.8 * len(data)), int(0.9 * len(data))
        ]
    )
    return train_df, valid_df, test_df