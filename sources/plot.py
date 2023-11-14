import matplotlib.pyplot as plt

def plot_results(training_losses, validation_losses, test_loss, training_accuracies, validation_accuracies, test_accuracy, best_val_loss):
    num_epochs = len(training_losses)
    epochs = list(range(1, num_epochs + 1))
    
    plt.figure(figsize=(9, 4))  

    # Training and Validation Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Training and Validation Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_accuracies, label='Training Accuracy')
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy')

    plt.scatter(num_epochs, test_accuracy, color='green', marker='o', s=50, label='Test Accuracy')

    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    best_epoch = validation_losses.index(min(validation_losses)) + 1  

    plt.annotate(f'Best Validation Loss: {best_val_loss:.4f}', xy=(best_epoch, best_val_loss),
                 xytext=(best_epoch - 3, 0.5), textcoords='data', fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='#D0FE1D'),
                 arrowprops=dict(arrowstyle='->', color='blue'))

    plt.show()
