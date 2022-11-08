import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split
from datasets import load_dataset
from transformers import PerceiverFeatureExtractor
from transformers import PerceiverTokenizer
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

SUPPORTED_DATASETS = ["CIFAR10", "IMDB"]

feature_extractor = PerceiverFeatureExtractor()


def preprocess_images(examples):
    examples['pixel_values'] = feature_extractor(
        examples['img'], return_tensors="pt").pixel_values
    return examples


class MaskedAutoencoderDataset(Dataset):

    def __init__(self, data, masking_ratio: float = 0.01):
        self.data = data
        self.masking_ratio = masking_ratio
        self.input_length = data[0].shape[1]
        self.number_unmasked_indices = np.floor(
            self.input_length * self.masking_ratio).astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_input = self.data[idx]
        masked_input = torch.zeros_like(self.data[idx])
        unmasked_indices = np.random.choice(
            self.input_length, self.number_unmasked_indices)
        masked_input[:, unmasked_indices] = full_input[:, unmasked_indices]
        return masked_input, full_input, unmasked_indices


def _prepare_CIFAR_input():
    # Load the dataset from HuggingFace
    train_ds, test_ds = load_dataset('cifar10', split=['train', 'test'])
    # Split up training into training + validation
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']

    train_ds.set_transform(preprocess_images)
    val_ds.set_transform(preprocess_images)
    test_ds.set_transform(preprocess_images)

    # Ignore the classes and take only images
    # We also flatten images into 1d
    train_set_input = [train_ds[i]['pixel_values'].flatten(
        start_dim=1) for i in range(len(train_ds))]
    val_set_input = [val_ds[i]['pixel_values'].flatten(
        start_dim=1) for i in range(len(val_ds))]
    test_set_input = [test_ds[i]['pixel_values'].flatten(
        start_dim=1) for i in range(len(test_ds))]
    return train_set_input, val_set_input, test_set_input


def _prepare_IMDB_input():
    # Load the dataset from HuggingFace
    train_ds, test_ds = load_dataset("imdb", split=['train', 'test'])
    # Split up training into training + validation
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']

    tokenizer = PerceiverTokenizer.from_pretrained(
        "deepmind/language-perceiver")
    train_ds = train_ds.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True),
                            batched=True)
    val_ds = val_ds.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True),
                        batched=True)
    test_ds = test_ds.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True),
                          batched=True)

    train_ds.set_format(type="torch", columns=[
                        'input_ids', 'attention_mask', 'label'])
    val_ds.set_format(type="torch", columns=[
                      'input_ids', 'attention_mask', 'label'])
    test_ds.set_format(type="torch", columns=[
        'input_ids', 'attention_mask', 'label'])

    # Ignore the classes and take only images
    # We add one dimension as the channel dimension
    train_set_input = [train_ds[i]["input_ids"][None, :]
                       for i in range(len(train_ds))]
    val_set_input = [val_ds[i]["input_ids"][None, :]
                     for i in range(len(val_ds))]
    test_set_input = [test_ds[i]["input_ids"][None, :]
                      for i in range(len(test_ds))]
    return train_set_input, val_set_input, test_set_input


def get_data(dataset: str = "CIFAR10", masking_ratio: float = 0.01):

    if dataset not in SUPPORTED_DATASETS:
        raise RuntimeError(f"The dataset {dataset} is not supported.")

    if dataset == "CIFAR10":
        train_set_input, val_set_input, test_set_input = _prepare_CIFAR_input()
    elif dataset == "IMDB":
        train_set_input, val_set_input, test_set_input = _prepare_IMDB_input()

    train_ds = MaskedAutoencoderDataset(train_set_input, masking_ratio)
    val_ds = MaskedAutoencoderDataset(val_set_input, masking_ratio)
    test_ds = MaskedAutoencoderDataset(test_set_input, masking_ratio)

    return train_ds, val_ds, test_ds


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str):
        train_ds, val_ds, test_ds = get_data("CIFAR10")
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)

    def teardown(self, stage: str):
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None


class IMDBDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str):
        train_ds, val_ds, test_ds = get_data("IMDB")
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)

    def teardown(self, stage: str):
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
