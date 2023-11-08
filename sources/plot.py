import matplotlib.pyplot as plt

def plot_results(epochs, training_losses, validation_losses, test_losses, training_accuracies, validation_accuracies, test_accuracies):
    
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

