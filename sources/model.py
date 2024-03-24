import os
import time
import torch
import torch.nn as nn
from transformers import BertModel

class PubMLP(nn.Module):
    def __init__(self, numeric_cols_num, mlp_hidden_size, output_size=1, dropout_rate=0.1):
        """
        Initializes a PubMLP model.

        Args:
            numeric_cols_num (int): Number of numeric columns in the input data.
            mlp_hidden_size (int): Size of the hidden layer in the MLP.
            output_size (int, optional): Size of the output layer. Defaults to 1.
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.1.
        """
        super(PubMLP, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.config = self.bert.config  
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + numeric_cols_num, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_size, output_size)  
        )

    def forward(self, input_ids, attention_mask, numeric_tensor):
        """
        Forward pass of the PubMLP model.

        Args:
            input_ids (torch.Tensor): Input IDs for the BERT model.
            attention_mask (torch.Tensor): Attention mask for the BERT model.
            numeric_tensor (torch.Tensor): Numeric features tensor.

        Returns:
            torch.Tensor: Output of the classifier.
        """
        sentence_embedding  = self.bert(input_ids, attention_mask).pooler_output
        concat_features = torch.cat((sentence_embedding, numeric_tensor), dim=1)
        dropout_output = self.dropout(concat_features)
        classifier_output = self.classifier(dropout_output)
        return classifier_output
        
    @classmethod
    def from_pretrained(cls, model_dir, best_model_state, numeric_cols_num, mlp_hidden_size, output_size, dropout_rate):
        """
        Creates a new instance of the model and loads the pretrained weights from the specified directory.

        Args:
            model_dir (str): The directory where the pretrained model weights are stored.
            best_model_state (str): The filename of the best model state.
            numeric_cols_num (int): The number of numeric columns in the input data.
            mlp_hidden_size (int): The size of the hidden layer in the MLP.
            output_size (int): The size of the output layer.
            dropout_rate (float): The dropout rate to be applied in the model.

        Returns:
            model (Model): The pretrained model with loaded weights.
        """
        model = cls(numeric_cols_num, mlp_hidden_size, output_size, dropout_rate)
        model_path = os.path.join(model_dir, best_model_state)
        model.load_state_dict(torch.load(model_path))
        return model

    
def calculate_loss(model, dataloader, criterion, device):
    """
    Calculates the average loss of the model on the given dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader containing the data.
        criterion: The loss function used to calculate the loss.
        device (torch.device): The device to perform the calculations on.

    Returns:
        float: The average loss over the entire dataset.
    """
    model.eval()
    total_loss, batch_size = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numeric_tensor = batch['numeric_tensor'].to(device)
            labels = batch['labels'].to(device).float()
            outputs = model(input_ids, attention_mask, numeric_tensor)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * input_ids.size(0)
            batch_size += input_ids.size(0)
    average_batch_loss = total_loss / batch_size
    return average_batch_loss


def calculate_accuracy(model, dataloader, device):
    """
    Calculate the accuracy of a model on a given dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader containing the data.
        device (torch.device): The device to perform the evaluation on.

    Returns:
        float: The average batch accuracy in percentage.
    """
    model.eval()
    correct_pred, batch_size = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numeric_tensor = batch['numeric_tensor'].to(device)
            labels = batch['labels'].to(device).float()
            outputs = model(input_ids, attention_mask, numeric_tensor)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct_pred += (predictions == labels).sum().item()
            batch_size += labels.size(0)
    average_batch_accuracy = correct_pred / batch_size * 100
    return average_batch_accuracy



def train_evaluate_model(model, train_dataloader, validation_dataloader, test_dataloader, optim, criterion, device, epochs):
    """
    Evaluates the performance of a given model on the training, validation, and test datasets.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        train_dataloader (torch.utils.data.DataLoader): The dataloader for the training dataset.
        validation_dataloader (torch.utils.data.DataLoader): The dataloader for the validation dataset.
        test_dataloader (torch.utils.data.DataLoader): The dataloader for the test dataset.
        optim (torch.optim.Optimizer): The optimizer used for training the model.
        criterion (torch.nn.Module): The loss function used for training the model.
        device (torch.device): The device on which the model and data are located.
        epochs (int): The number of epochs to train the model.

    Returns:
        tuple: A tuple containing the following elements:
            - train_losses (list): A list of training losses for each epoch.
            - validation_losses (list): A list of validation losses for each epoch.
            - train_accuracies (list): A list of training accuracies for each epoch.
            - validation_accuracies (list): A list of validation accuracies for each epoch.
            - test_accuracy (float): The accuracy of the model on the test dataset.
            - best_val_loss (float): The best validation loss achieved during training.
            - best_model_state (dict): The state dictionary of the best model.
    """
    start_time = time.time()
    model.to(device)
    train_losses, validation_losses = [], []
    train_accuracies, validation_accuracies = [], []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss, batch_size = 0.0, 0

        for batch_id, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numeric_tensor = batch['numeric_tensor'].to(device)
            labels = batch['labels'].to(device).float()
            outputs = model(input_ids, attention_mask, numeric_tensor)
            loss = criterion(outputs, labels)

            optim.zero_grad()
            loss.backward(retain_graph=True)
            optim.step()

            total_loss += loss.item() * input_ids.size(0)
            batch_size += input_ids.size(0)

        average_batch_loss = total_loss / batch_size
        train_losses.append(average_batch_loss)

        model.eval()
        with torch.set_grad_enabled(False):
            validation_loss = calculate_loss(model, validation_dataloader, criterion, device)
            validation_losses.append(validation_loss)

            train_accuracy = calculate_accuracy(model, train_dataloader, device)
            train_accuracies.append(train_accuracy)

            validation_accuracy = calculate_accuracy(model, validation_dataloader, device)
            validation_accuracies.append(validation_accuracy)

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                best_model_state = model.state_dict()

            print(f'Epoch: {epoch+1:04d}/{epochs:04d} | Training Loss: {average_batch_loss:.3f} | Validation Loss: {validation_loss:.3f}')
            print(f'Training Accuracy: {train_accuracy:.3f}% | Validation Accuracy: {validation_accuracy:.3f}%')
            print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')

    test_accuracy = calculate_accuracy(model, test_dataloader, device)
    print(f'Test Accuracy: {test_accuracy:.3f}%')

    return train_losses, validation_losses, train_accuracies, validation_accuracies, test_accuracy, best_val_loss, best_model_state


def predict_model(model, unlabeled_dataloader, device):
    """
    Predicts labels for unlabeled data using the given model.

    Args:
        model (torch.nn.Module): The trained model.
        unlabeled_dataloader (torch.utils.data.DataLoader): The dataloader for unlabeled data.
        device (torch.device): The device to perform the prediction on.

    Returns:
        list: A list of predicted labels.
    """
    model.eval()
    predicted_labels_list = []

    with torch.no_grad():
        for batch in unlabeled_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numeric_tensor = batch['numeric_tensor'].to(device)
            outputs = model(input_ids, attention_mask, numeric_tensor)
            logits = outputs.squeeze() 
            probs = torch.sigmoid(logits)  
            predicted_labels = (probs > 0.5).long()  
            predicted_labels_list.extend(predicted_labels.tolist())
    return predicted_labels_list
