import torch
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split

SUPPORTED_DATASETS = ["CIFAR10"]


class MaskedAutoencoderDataset(Dataset):

    def __init__(self, data, masking_ratio: float = 0.01):
        self.data = data
        self.masking_ratio = masking_ratio
        self.input_length = data[0].shape[2]
        self.number_unmasked_indices = np.floor(
            self.input_length * self.masking_ratio)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_input = self.data[idx]
        masked_input = torch.zeros_like(self.data[idx])
        unmasked_indices = np.random.permutation(self.input_length)

        print("self.input_length", self.input_length)
        print("unmasked_indices", unmasked_indices)
        unmasked_indices = unmasked_indices[:self.number_unmasked_indices]
        print("unmasked_indices", unmasked_indices)
        masked_input[:, unmasked_indices] = full_input[:, unmasked_indices]
        return masked_input, full_input


def get_data(dataset: str = "CIFAR10"):

    if dataset not in SUPPORTED_DATASETS:
        raise RuntimeError(f"The dataset {dataset} is not supported.")

    if dataset == "CIFAR10":
        # Download / Load original data
        train_val_set_original = CIFAR10(
            root='data/', download=True, transform=ToTensor())
        test_set_original = CIFAR10(
            root='data/', train=False, transform=ToTensor())

        # Split train/val sets
        val_size = 500
        train_size = len(train_val_set_original) - val_size
        train_set_original, val_set_original = random_split(
            train_val_set_original, [train_size, val_size])

        # Ignore the classes and take only images
        train_set_input = [train_set_original[i][0].flatten(
            start_dim=2) for i in range(len(train_set_original))]
        val_set_input = [val_set_original[i][0].flatten(start_dim=2)
                         for i in range(len(val_set_original))]
        test_set_input = [test_set_original[i][0].flatten(
            start_dim=2) for i in range(len(test_set_original))]

    elif dataset == "IMAGENET":
        # TODO: Implement the preprocessing for Imagenet
        raise NotImplementedError()
    elif dataset == "MODELNET":
        # TODO: Implement the preprocessing for Modelnet
        raise NotImplementedError()

    train_ds = MaskedAutoencoderDataset(train_set_input)
    val_ds = MaskedAutoencoderDataset(val_set_input)
    test_ds = MaskedAutoencoderDataset(test_set_input)

    return train_ds, val_ds, test_ds
