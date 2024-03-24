import matplotlib.pyplot as plt

def plot_results(train_losses, validation_losses, train_accuracies, validation_accuracies, test_accuracy, best_val_loss):
    if len(train_losses) != len(validation_losses) or len(train_accuracies) != len(validation_accuracies):
        raise ValueError("Input lists must have the same length")

    num_epochs = len(train_losses)
    epochs = list(range(1, num_epochs + 1))
    
    plt.figure(figsize=(9, 4))  

    # Training and validation loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.scatter(num_epochs, best_val_loss, color='red', marker='o', s=50, label='Best Validation Loss: {:.3f}'.format(best_val_loss))
    plt.legend()
    
    # Training and validation accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy')
    plt.scatter(num_epochs, test_accuracy, color='blue', marker='o', s=50, label='Test Accuracy: {:.3f}'.format(test_accuracy))
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()