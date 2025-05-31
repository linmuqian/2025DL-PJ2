import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle
import pandas as pd
import numpy as np
import os 



def train(model, trainloader, criterion, optimizer, scheduler, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        total_batches = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        avg_loss = running_loss / total_batches
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.3f}")

        scheduler.step()

    print("Training finished.")



def train(model, trainloader, validationloader, criterion, optimizer, scheduler, device, model_path, epochs=50, early_stopping=True, patience=5):
    # Ensure the model path exists
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    model.train()
    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    # Track losses and accuracies
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        running_loss = 0.0
        total_batches = 0

        # Training phase
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        avg_train_loss = running_loss / total_batches
        train_losses.append(avg_train_loss)

        # Validation phase (calculate accuracy)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in validationloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Early stopping logic
        if early_stopping:
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
                # Save the model if it improved
                torch.save(model.state_dict(), os.path.join(model_path, 'best_model.pth'))
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch + 1} epochs due to no improvement.")
                break
        else:
            # If not using early stopping, save the final model
            torch.save(model.state_dict(), os.path.join(model_path, 'final_model.pth'))

        model.train()

        # Update learning rate based on the scheduler type
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # If using ReduceLROnPlateau, pass validation loss or accuracy to scheduler
            scheduler.step(avg_train_loss) 
        else:
            # For other schedulers, no need to pass any argument
            scheduler.step()

    print("Training finished.")

    # Plot loss and accuracy
    plot_loss_and_score(train_losses, val_accuracies)





def plot_loss_and_score(train_losses, val_accuracies):
    # Plot train loss and validation accuracy
    plt.figure(figsize=(12, 6))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Validation Accuracy", color="green")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def test(model, testloader, device, model_path):
    # Load the best model (if available)
    model_path_pth = rf'{model_path}/best_model.pth'
    if os.path.exists(model_path):
        print(f"Loading the best model from {model_path}...")
        model.load_state_dict(torch.load(model_path_pth))
    else:
        print("No best model found. Using the provided model.")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy



def load_cifar10(data_dir):
    def load_batch(filename):
        with open(filename, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
            data = batch['data'].reshape(-1, 3, 32, 32)
            labels = batch['labels']
            return data, labels

    transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data, train_labels = [], []
    for i in range(1, 6):
        d, l = load_batch(os.path.join(data_dir, f'data_batch_{i}'))
        train_data.append(d)
        train_labels.extend(l)
    train_data = np.concatenate(train_data)
    train_labels = torch.tensor(train_labels)

    test_data, test_labels = load_batch(os.path.join(data_dir, 'test_batch'))
    test_labels = torch.tensor(test_labels)

    train_data = torch.tensor(train_data, dtype=torch.float32) / 255.0
    test_data = torch.tensor(test_data, dtype=torch.float32) / 255.0

    train_data = transform(train_data)
    test_data = transform(test_data)

    return train_data, train_labels, test_data, test_labels


def load_combined_data(data_path):
    with open(data_path, 'rb') as f:
        augmented_data = pickle.load(f)
    
    # Load the data and labels from the saved file
    combined_data = torch.tensor(augmented_data['data'], dtype=torch.float32)
    combined_labels = torch.tensor(augmented_data['labels'], dtype=torch.long)
    
    # Create a TensorDataset and DataLoader for batching
    dataset = TensorDataset(combined_data, combined_labels)
    return DataLoader(dataset, batch_size=128, shuffle=True)
