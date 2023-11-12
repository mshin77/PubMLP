import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_results(epochs, training_losses, validation_losses, test_losses, training_accuracies, validation_accuracies, test_accuracies):
    """
    Plots the training, validation, and test losses and accuracies over the specified number of epochs.

    Args:
    epochs (int): The number of epochs.
    training_losses (list): The training losses.
    validation_losses (list): The validation losses.
    test_losses (list): The test losses.
    training_accuracies (list): The training accuracies.
    validation_accuracies (list): The validation accuracies.
    test_accuracies (list): The test accuracies.
    """
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), training_losses, label='Training Loss', marker='o')
    plt.plot(range(1, epochs + 1), validation_losses, label='Validation Loss', marker='o')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Test Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), training_accuracies, label='Training Accuracy', marker='o')
    plt.plot(range(1, epochs + 1), validation_accuracies, label='Validation Accuracy', marker='o')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training, Validation, and Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

