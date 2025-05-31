import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import load_cifar10
from torch.utils.data import TensorDataset, DataLoader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 2
batch_size = 128

# add our package dir to path 
module_path = rf'PJ/VGG_BatchNorm'
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# Make sure you are using the right device.
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:{}".format(4) if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(4))


# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.

train_data, train_labels, test_data, test_labels = load_cifar10("/home/user79/PJ/CIFAR-10/data/cifar-10-batches-py")  
valid_data = train_data[:10000]
valid_labels = train_labels[:10000]
train_data = train_data[10000:]
train_labels = train_labels[10000:]
train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(TensorDataset(valid_data, valid_labels), batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=batch_size, shuffle=False, num_workers=num_workers)
# train_loader = get_cifar_loader(train=True)
# val_loader = get_cifar_loader(train=False)
for X,y in train_loader:
    # Add code as needed
    print(f"Input shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Sample label: {y[0]}")
    break


# This function is used to calculate the accuracy of model classification
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

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None, figure_name="training_progress"):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []  # record loss values of each epoch
    grads = [] # record the grad norm of each epoch
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            # Add your code
            loss_list.append(loss.item())
            loss.backward()

            if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
                if len(model.classifier) >= 4:
                    current_grad = model.classifier[4].weight.grad.clone().norm().item()
                    grad.append(current_grad)  # the gradient norm of the 4th layer in the classifier
            optimizer.step()
        
        epoch_loss = np.mean(loss_list)
        learning_curve[epoch] = epoch_loss
        losses_list.append(loss_list)
        grads.append(grad)

        # calculate train dataset accuracy
        train_accuracy = get_accuracy(model, train_loader, device)
        train_accuracy_curve[epoch] = train_accuracy

        # calculate valid dataset accuracy
        val_accuracy = get_accuracy(model, val_loader, device)
        val_accuracy_curve[epoch] = val_accuracy

        # save the best model  
        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            max_val_accuracy_epoch = epoch
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)
        
        print(f'\nEpoch {epoch+1}/{epochs_n}, Loss: {epoch_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%, '
              f'Best Val Acc: {max_val_accuracy:.2f}% at Epoch {max_val_accuracy_epoch+1}')

    # Visualize training metrics after all epochs complete
    f, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Training loss curve
    axes[0].plot(learning_curve, label='Training Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plot 2: Accuracy curves
    axes[1].plot(train_accuracy_curve, label='Train Accuracy')
    axes[1].plot(val_accuracy_curve, label='Validation Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()

    '''
    # Plot 3: Gradient norm (only show the first epoch's gradient)
    if grads and grads[0]:  # Check if gradient data exists
        axes[2].plot(grads[0], label='Epoch 0')
        axes[2].set_title('Gradient Norm (Epoch 0)')
        axes[2].set_xlabel('Batch')
        axes[2].set_ylabel('Gradient Norm')
        axes[2].legend()'''

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'{figure_name}.png'))
    plt.close()
    
    return losses_list, grads


# Train your model
# feel free to modify
epo = 20
loss_save_path = models_path
grad_save_path = models_path

# train model with different learning rate 
def train_multiple_models(model_class, learning_rates, epochs=epo):
    all_losses = []
    all_grads = []
    
    for lr in learning_rates:
        print(f"\nTraining {model_class.__name__} with lr={lr}")
        set_random_seeds(seed_value=2020, device=device)
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model_name = f"{model_class.__name__}_lr_{lr}"
        best_model_path = os.path.join(models_path, f"{model_name}.pth")
        
        losses, grads = train(
            model, optimizer, criterion, train_loader, val_loader, 
            epochs_n=epochs, best_model_path=best_model_path,
            figure_name=f'{model_name}_training_progress'
        )
        
        all_losses.append(losses)
        all_grads.append(grads)
        
        # 保存损失和梯度数据
        np.savetxt(os.path.join(loss_save_path, f"{model_name}_loss.txt"), 
                  np.array([item for sublist in losses for item in sublist]), 
                  fmt='%s', delimiter=' ')
        np.savetxt(os.path.join(grad_save_path, f"{model_name}_grads.txt"), 
                  np.array([item for sublist in grads for item in sublist]), 
                  fmt='%s', delimiter=' ')
    
    return all_losses, all_grads


# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
# Add your code
# 计算min_curve和max_curve
def compute_min_max_curves(all_losses):
    min_curve = []
    max_curve = []
    
    # since the batch and epoch of every model is the same
    epochs = len(all_losses[0])
    batches_per_epoch = len(all_losses[0][0])
    
    for epoch in range(epochs):
        for batch in range(batches_per_epoch):
            # get the losses
            batch_losses = [model_losses[epoch][batch] for model_losses in all_losses]
            min_curve.append(min(batch_losses))
            max_curve.append(max(batch_losses))
    
    return min_curve, max_curve

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape(vgg_min, vgg_max, vgg_bn_min, vgg_bn_max):
    # Add your code
    plt.figure(figsize=(12, 6))
    
    # lanscape of VGG-A
    plt.fill_between(range(len(vgg_min)), vgg_min, vgg_max, 
                     alpha=0.3, label='VGG-A', color='green')
    
    # landscape of VGG-A-BatchNorm
    plt.fill_between(range(len(vgg_bn_min)), vgg_bn_min, vgg_bn_max, 
                     alpha=0.3, label='VGG-A-BatchNorm', color='red')
    
    plt.title('Loss Landscape Comparison: VGG-A vs VGG-A-BatchNorm')
    plt.xlabel('Steps')
    plt.ylabel('Loss Landscape')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(figures_path, 'loss_landscape_1.png'))
    plt.close()
    
    print("Loss landscape comparison plot saved to:", 
          os.path.join(figures_path, 'loss_landscape_1.png'))
    

def main():
    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
    
    print("Training VGG-A models...")
    vgg_losses, vgg_grads = train_multiple_models(VGG_A, learning_rates)
    
    print("\nTraining VGG-A-BatchNorm models...")
    vgg_bn_losses, vgg_bn_grads = train_multiple_models(VGG_A_BatchNorm, learning_rates)
    
    # loss landscape
    print("\nComputing loss landscape curves...")
    vgg_min_curve, vgg_max_curve = compute_min_max_curves(vgg_losses)
    vgg_bn_min_curve, vgg_bn_max_curve = compute_min_max_curves(vgg_bn_losses)
    
    # loss lanscape
    print("\nPlotting loss landscape comparison...")
    plot_loss_landscape(vgg_min_curve, vgg_max_curve, 
                        vgg_bn_min_curve, vgg_bn_max_curve)
    
    print("\nTraining and visualization completed!")

if __name__ == "__main__":
    main() 