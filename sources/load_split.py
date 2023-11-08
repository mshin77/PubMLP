import pandas as pd
import spacy
import numpy as np

def load_split_data(data_file_path, ner_model):
    # Load the initial data
    data = pd.read_csv(data_file_path)

    # Define a function to extract country entities using NER
    def extract_country_entities(text):
        if pd.notna(text):  
            doc = ner_model(text)
            countries = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
            return ', '.join(countries)
        else:
            return ''

    # Apply the function to the 'C1' column and create a new 'Country' column
    data['Country'] = data['C1'].apply(extract_country_entities)

    # Select relevant columns
    raw_df = data[['UT', 'included', 'single_case', 'technology_use',
                    'PY', 'AF', 'TI', 'AB', 'DE', 'ID', 'SO', 'CR', 'Country']]
    
    raw_df = raw_df.copy()

    # Convert PY (publication year) to numeric format
    raw_df['PY'] = pd.to_numeric(raw_df['PY'], errors='coerce')

    # Check columns with text data
    # print(raw_df.describe(include=object))

    # Check columns with numeric data
    # print(raw_df.describe())

    missing_values = raw_df.isnull().sum()
    print(missing_values)

    df = raw_df.copy()

    # Replace missing values
    df.dropna(subset=['PY'], inplace=True)

    # Split the data for training, validation, and testing 
    train_df, valid_df, test_df = np.split(
        df.sample(frac=1, random_state=42), [
            int(0.8 * len(df)), int(0.9 * len(df))
        ]
    )

    return train_df, valid_df, test_df