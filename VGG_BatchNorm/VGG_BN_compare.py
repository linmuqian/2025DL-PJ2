import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm  # This network needs to be implemented
from data.loaders import load_cifar10
from torch.utils.data import TensorDataset, DataLoader

# ## Constant (parameter) initialization
device_id = [0, 1, 2, 3]
num_workers = 2
batch_size = 128

# Add package directory to path
module_path = rf'PJ/VGG_BatchNorm'
home_path = module_path
figures_path = os.path.join(home_path, 'reports_compare', 'figures')
models_path = os.path.join(home_path, 'reports_compare', 'models')

os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# Ensure using the correct device
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:{}".format(4) if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(4)) if torch.cuda.is_available() else print("Using CPU")


# Load data
train_data, train_labels, test_data, test_labels = load_cifar10("/home/user79/PJ/CIFAR-10/data/cifar-10-batches-py")
valid_data = train_data[:10000]
valid_labels = train_labels[:10000]
train_data = train_data[10000:]
train_labels = train_labels[10000:]
train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(TensorDataset(valid_data, valid_labels), batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Print data sample information
for X, y in train_loader:
    print(f"Input shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Sample label: {y[0]}")
    break


# Calculate model classification accuracy
def get_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


# Set random seeds to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Complete training process, recording training loss, training set accuracy, and validation set accuracy
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None, figure_name="training_progress"):
    model.to(device)
    learning_curve = [np.nan] * epochs_n  # Training loss curve
    train_accuracy_curve = [np.nan] * epochs_n  # Training set accuracy curve
    val_accuracy_curve = [np.nan] * epochs_n  # Validation set accuracy curve
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # Record loss values for each batch

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

        epoch_loss = np.mean(loss_list)
        learning_curve[epoch] = epoch_loss

        # Calculate training set accuracy
        train_accuracy = get_accuracy(model, train_loader, device)
        train_accuracy_curve[epoch] = train_accuracy

        # Calculate validation set accuracy
        val_accuracy = get_accuracy(model, val_loader, device)
        val_accuracy_curve[epoch] = val_accuracy

        # Save the best model
        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            max_val_accuracy_epoch = epoch
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)

        print(f'\nEpoch {epoch + 1}/{epochs_n}, Loss: {epoch_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%, '
              f'Best Val Accuracy: {max_val_accuracy:.2f}% at Epoch {max_val_accuracy_epoch + 1}')

    return learning_curve, train_accuracy_curve, val_accuracy_curve


# Compare the performance of two models under the same learning rate
def compare_models(model_classes, lr, epochs=20):
    results = {}
    
    for model_class in model_classes:
        print(f"\nTraining {model_class.__name__} with learning rate {lr}...")
        set_random_seeds(seed_value=2020, device=device)
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model_name = f"{model_class.__name__}_lr_{lr}"
        best_model_path = os.path.join(models_path, f"{model_name}.pth")
        
        # Train the model and get results
        train_loss, train_acc, val_acc = train(
            model, optimizer, criterion, train_loader, val_loader,
            epochs_n=epochs, best_model_path=best_model_path,
            figure_name=f'{model_name}_training_progress'
        )
        
        results[model_class.__name__] = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
    
    return results


# Plot comparison charts
def plot_comparison(results, lr, epochs):
    plt.figure(figsize=(18, 5))
    
    # 1. Training loss comparison chart
    plt.subplot(1, 3, 1)
    for model_name, metrics in results.items():
        plt.plot(metrics['train_loss'], label=model_name)
    
    plt.title(f'Training Loss Comparison (Learning Rate: {lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Training set accuracy comparison chart
    plt.subplot(1, 3, 2)
    for model_name, metrics in results.items():
        plt.plot(metrics['train_acc'], label=model_name)
    
    plt.title(f'Training Set Accuracy Comparison (Learning Rate: {lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Validation set accuracy comparison chart
    plt.subplot(1, 3, 3)
    for model_name, metrics in results.items():
        plt.plot(metrics['val_acc'], label=model_name)
    
    plt.title(f'Validation Set Accuracy Comparison (Learning Rate: {lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'model_comparison_lr_{lr}.png'))
    plt.close()
    
    print(f"Model comparison chart saved to: {os.path.join(figures_path, f'model_comparison_lr_{lr}.png')}")


def main():
    learning_rate = 5e-4  # Specified learning rate
    epochs = 20  # Number of training epochs
    
    # Models to compare
    model_classes = [VGG_A, VGG_A_BatchNorm]
    
    # Train and compare models
    results = compare_models(model_classes, learning_rate, epochs)
    
    # Plot comparison charts
    plot_comparison(results, learning_rate, epochs)
    
    print("\nTraining and visualization completed!")


if __name__ == "__main__":
    main()