from datetime import datetime
from tzlocal import get_localzone
from torch.utils.data import Dataset, Sampler, DataLoader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import QuantileTransformer

class CustomDataset(Dataset):
    """
    A custom dataset class for handling input data and labels.

    Args:
        input_ids (list): List of input IDs.
        attention_mask (list): List of attention masks.
        numeric_tensor (list): List of numeric tensors.
        labels (list): List of labels.

    Returns:
        dict: A dictionary containing the input data and labels for a single sample.
    """
    def __init__(self, input_ids, attention_mask, numeric_tensor, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.numeric_tensor = numeric_tensor
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'numeric_tensor': self.numeric_tensor[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.input_ids)

def preprocess_dataset(data, tokenizer, device, col_lists, numeric_transform):
    """
    Preprocesses the dataset by performing text and numeric data transformations.

    Args:
        data (pandas.DataFrame): The input dataset.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for text data processing.
        device (torch.device): The device to be used for computation.
        col_lists (dict): A dictionary containing the column lists for text, categorical, and numeric data.
        numeric_transform (dict): A dictionary specifying the transformation options for numeric columns.

    Returns:
        CustomDataset: The preprocessed dataset as a CustomDataset object.
    """
    # Process text (and categorical) data 
    if 'categorical_cols' in col_lists:
        combined_cols = col_lists['text_cols'] + col_lists['categorical_cols']
    else:
        combined_cols = col_lists['text_cols']
    texts = ['[CLS] ' + ' [SEP] '.join(str(row[col]) for col in combined_cols) + ' [SEP]' for _, row in data.iterrows()]   
    preprocessed_data = data.drop(columns=combined_cols)
    preprocessed_data['texts'] = texts        
    encoding = tokenizer(texts, max_length = 512, truncation=True, padding="max_length", return_tensors="pt")
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Process numeric data    
    numeric_tensor = None
    if 'numeric_cols' in col_lists:
        for num_col in col_lists['numeric_cols']:
            if num_col in numeric_transform:
                option = numeric_transform[num_col]
                if option == 'min':
                    preprocessed_data[num_col] -= data[num_col].min()
                elif option == 'max':
                    preprocessed_data[num_col] /= data[num_col].max()
                elif option == 'mean':
                    preprocessed_data[num_col] -= data[num_col].mean()
                elif option == 'quantile':
                    transformed = QuantileTransformer(output_distribution='normal', random_state=0).fit_transform(data[[num_col]])
                    preprocessed_data[num_col] = transformed.flatten()
                else:
                    raise ValueError(f"Invalid or unspecified transform option for {num_col}.")
            else:
                raise ValueError(f"Column {num_col} not found in numeric_transform.")
        numeric_tensor = torch.tensor(preprocessed_data[col_lists['numeric_cols']].values, dtype=torch.float).to(device)

    labels = torch.tensor(data[col_lists['label_col']].values, dtype=torch.long).to(device)

    local_tz = get_localzone()
    now = datetime.now(local_tz) 
    label_col = col_lists['label_col'][0]
    date_time = now.strftime("%m%d%Y_%H%M%S")
    file_name = f"preprocessed_data_{label_col}_{date_time}.xlsx"
    preprocessed_data.to_excel(file_name, index=False)
    print(f"Preprocessed data saved to {file_name}")

    return CustomDataset(input_ids, attention_mask, numeric_tensor, labels)

def create_dataloader(dataset: Dataset, sampler: Sampler, batch_size: int) -> DataLoader:
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