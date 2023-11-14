from torch.utils.data import Dataset, Sampler, DataLoader
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder

def preprocess_dataset(data, tokenizer, device, col_lists):
    """
    Preprocesses the input dataset by encoding text, categorical, and numerical data, and concatenating them into a single tensor.

    Args:
        data (pandas.DataFrame): The input dataset to preprocess.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for encoding text data.
        device (str): The device to use for processing the tensors (either 'cpu' or 'cuda').
        col_lists (dict): A dictionary containing column names or lists of column names for text, categorical, numerical, and label data.

    Returns:
        torch.utils.data.TensorDataset: A PyTorch TensorDataset containing the preprocessed input data and labels.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for key in ['text_cols', 'categorical_cols', 'numerical_cols']:
        if key in col_lists and not isinstance(col_lists[key], list):
            col_lists[key] = [col_lists[key]]  

    # Process text columns
    text_embeddings = []
    attention_masks = []
    if 'text_cols' in col_lists:
        for _, row in data.iterrows():
            combined_text = '[SEP] '.join(row[col_lists['text_cols']].astype(str)) + '[SEP]'
            tokenized = tokenizer.encode_plus(
                combined_text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            text_embeddings.append(tokenized["input_ids"])
            attention_masks.append(tokenized["attention_mask"])

    text_embeddings = torch.cat(text_embeddings, dim=0) if text_embeddings else torch.tensor([], device=device)
    attention_masks = torch.cat(attention_masks, dim=0) if attention_masks else torch.tensor([], device=device)

    # Process categorical data
    categorical_tensor = torch.tensor([], device=device)
    if 'categorical_cols' in col_lists and col_lists['categorical_cols']:
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoded_categorical = encoder.fit_transform(data[col_lists['categorical_cols']])
        categorical_tensor = torch.tensor(encoded_categorical.toarray(), dtype=torch.float32).to(device)
    else:
        categorical_tensor = torch.tensor([], device=device)

    # Process numerical data
    numeric_tensor = torch.tensor(data[col_lists['numerical_cols']].values, dtype=torch.float32).to(device) if 'numerical_cols' in col_lists else torch.tensor([], device=device)

    # Concatenate all the input data tensors
    combined_tensors = torch.cat([text_embeddings, attention_masks, categorical_tensor, numeric_tensor], dim=1)

    # Process label data
    encoder = OneHotEncoder(handle_unknown='ignore')
    label_cols = col_lists.get('label_col', [])
    if isinstance(label_cols, str):
        label_cols = [label_cols]
    encoded_label_col = encoder.fit_transform(data[label_cols])
    label_tensor = torch.tensor(encoded_label_col.toarray(), dtype=torch.float32).to(device)

    return TensorDataset(combined_tensors, label_tensor)

# Create a DataLoader
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
    dataloader = DataLoader(
        dataset, sampler=data_sampler, batch_size=batch_size)
    return dataloader

