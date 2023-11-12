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

def train_model(
    model, 
    train_dataset, 
    valid_dataset, 
    test_dataset, 
    train_dataloader, 
    valid_dataloader, 
    test_dataloader, 
    criterion, 
    optimizer, 
    device=None, 
    epochs=5
):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    training_losses = []
    validation_losses = []
    test_losses = []
    training_accuracies = []
    validation_accuracies = []
    test_accuracies = []

    # Evaluation on the training dataset
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0

        for batch in train_dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.argmax(dim=1).long().to(device)  

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

        average_loss = total_loss / len(train_dataloader)
        accuracy = total_correct / len(train_dataset)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {average_loss:.4f}, Training Accuracy: {accuracy:.4f}")
        
        # Evaluation on the validation dataset
        model.eval()
        valid_loss = 0.0
        valid_correct = 0

        with torch.no_grad():
            for batch in valid_dataloader:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.argmax(dim=1).long().to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                valid_correct += (outputs.argmax(dim=1) == labels).sum().item()

        valid_average_loss = valid_loss / len(valid_dataloader)
        valid_accuracy = valid_correct / len(valid_dataset)

        print(f"Validation Loss: {valid_average_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}")

        # Evaluation on the test dataset
        test_loss = 0.0
        test_correct = 0

        with torch.no_grad():
            for batch in test_dataloader:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.argmax(dim=1).long().to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                test_correct += (outputs.argmax(dim=1) == labels).sum().item()

        test_average_loss = test_loss / len(test_dataloader)
        test_accuracy = test_correct / len(test_dataset)

        print(f"Test Loss: {test_average_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        training_losses.append(average_loss)
        validation_losses.append(valid_average_loss)
        test_losses.append(test_average_loss)
        training_accuracies.append(accuracy)
        validation_accuracies.append(valid_accuracy)
        test_accuracies.append(test_accuracy)
        
    return training_losses, validation_losses, test_losses, training_accuracies, validation_accuracies, test_accuracies   





