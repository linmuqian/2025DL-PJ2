"""
Data loaders
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as datasets
import torch
import pickle
import os


class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self):
        return self.dataset.__getitem__()

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def get_cifar_loader(root='../data/', batch_size=128, train=True, shuffle=True, num_workers=4, n_items=-1):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    data_transforms = transforms.Compose(
        [transforms.ToTensor(),
        normalize])

    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=data_transforms)
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader


# Since I have download the cifar10 dataset, I use this function to read the dataset
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

if __name__ == '__main__':
    train_loader = get_cifar_loader()
    for X, y in train_loader:
        print(X[0])
        print(y[0])
        print(X[0].shape)
        img = np.transpose(X[0], [1,2,0])
        plt.imshow(img*0.5 + 0.5)
        plt.savefig('sample.png')
        print(X[0].max())
        print(X[0].min())
        break