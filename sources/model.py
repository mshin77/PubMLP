import torch
import torch.nn as nn
import torch.optim as optim
from transformers import PreTrainedModel, PretrainedConfig

# Define the multilayer perceptron model 
class PubMLPConfig(PretrainedConfig):
    """
    Configuration class for PubMLP.

    Args:
        input_size (int, optional): The size of the input features. Defaults to 1029.
        hidden_size (int, optional): The size of the hidden layer. Defaults to 32.
        num_classes (int, optional): The number of classes in the output layer. Defaults to 2.
        dropout_prob (float, optional): The dropout probability. Defaults to 0.5.
        **kwargs: Additional keyword arguments passed to the superclass.
    """
    def __init__(self, input_size=1029, hidden_size=32, num_classes=2, dropout_prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.model_type = 'pubmlp' 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

class PubMLP(PreTrainedModel):
    """
    A custom multilayer perceptron (MLP) for classification, compatible with the Hugging Face ecosystem.
    
    Args:
        config (PubMLPConfig): The configuration object specifying the model architecture and hyperparameters.
    """
    config_class = PubMLPConfig

    def __init__(self, config):
        super(PubMLP, self).__init__(config)
        self.fc1 = nn.Linear(config.input_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=config.dropout_prob)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, device, epochs=5, patience=3):
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.argmax(dim=1).long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

        training_loss = total_loss / len(train_dataloader)
        training_accuracy = total_correct / len(train_dataloader.dataset)
        training_losses.append(training_loss)
        training_accuracies.append(training_accuracy)

        model.eval()
        valid_loss, valid_correct = 0, 0
        with torch.no_grad():
            for inputs, labels in valid_dataloader:
                inputs, labels = inputs.to(device), labels.argmax(dim=1).long().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                valid_correct += (outputs.argmax(dim=1) == labels).sum().item()

        validation_loss = valid_loss / len(valid_dataloader)
        validation_accuracy = valid_correct / len(valid_dataloader.dataset)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_model_state = model.state_dict().copy()

    return training_losses, validation_losses, training_accuracies, validation_accuracies, best_val_loss, best_model_state

def test_model(model, test_dataloader, criterion, device):
    model.eval()
    test_loss, test_correct = 0, 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.argmax(dim=1).long().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_correct += (outputs.argmax(dim=1) == labels).sum().item()

    test_average_loss = test_loss / len(test_dataloader)
    test_accuracy = test_correct / len(test_dataloader.dataset)
    return test_average_loss, test_accuracy




