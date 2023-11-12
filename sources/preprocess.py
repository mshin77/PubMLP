import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Preprocess text, categorical, and numeric Data
def preprocess_dataset(data, tokenizer, device):
    """
    Preprocesses the input dataset by tokenizing the text columns, converting categorical and numerical columns to tensors,
    and concatenating all the input data tensors. The label column is also converted to a tensor.

    Args:
        data (pandas.DataFrame): The input dataset.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenizing the text columns.
        device (str): The device to use for the tensors.

    Returns:
        torch.utils.data.TensorDataset: The preprocessed dataset as a TensorDataset object.
    """
def preprocess_dataset(data, tokenizer, device):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_cols = data[['AF', 'TI', 'AB', 'DE', 'ID', 'SO', 'CR', 'Country']]
    text_embeddings = []
    attention_masks = []
    for (i, row) in data.iterrows():  
        combined = ''
        for col in text_cols:  
            combined += (str(row[col]) + '[SEP] ')
        tokenized = tokenizer.encode_plus(
            combined,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        text_embeddings.append(tokenized["input_ids"])
        attention_masks.append(tokenized["attention_mask"])
    
    text_embeddings = torch.cat(text_embeddings, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Convert categorical data (one-hot encoding)
    categorical_cols = data[['single_case', 'technology_use']]
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_categorical_cols = encoder.fit_transform(categorical_cols)
    encoded_categorical_cols_np = encoded_categorical_cols.toarray()
    categorical_tensor = torch.tensor(encoded_categorical_cols_np, dtype=torch.float32).to(device)

    # Convert numerical data (publication year) 
    numerical_cols = data[['PY']]
    numerical_cols_np = numerical_cols.values
    numeric_tensor = torch.tensor(numerical_cols_np, dtype=torch.float32).to(device)

    # Concatenate all the input data tensors
    combined_tensors = torch.cat([text_embeddings, attention_masks, categorical_tensor, numeric_tensor], dim=1)
    input_size = combined_tensors.size(1)

    # Convert the label data (one-hot encoding)
    label_col = data[['included']]
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_label_col = encoder.fit_transform(label_col)
    encoded_label_col_np = encoded_label_col.toarray()
    label_tensor = torch.tensor(encoded_label_col_np, dtype=torch.float32).to(device)

    return TensorDataset(combined_tensors, label_tensor)  

# Create a DataLoader
from torch.utils.data import Dataset, Sampler, DataLoader

def get_dataloader(dataset: Dataset, sampler: Sampler, batch_size: int) -> DataLoader:
    """
    Returns a DataLoader object that batches data from the given dataset using the given sampler.

    Args:
        dataset (Dataset): The dataset to batch.
        sampler (Sampler): The sampler to use for sampling elements from the dataset.
        batch_size (int): The size of each batch.

    Returns:
        DataLoader: The DataLoader object that batches data from the given dataset using the given sampler.
    """
    data_sampler = sampler(dataset)
    dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=batch_size)
    return dataloader
    