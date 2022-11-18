import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from datasets import load_dataset
from transformers import PerceiverFeatureExtractor
from transformers import PerceiverTokenizer
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

SUPPORTED_DATASETS = ["CIFAR10", "IMDB"]

feature_extractor = PerceiverFeatureExtractor()


def preprocess_images(examples):
    examples["pixel_values"] = feature_extractor(
        examples["img"], return_tensors="pt"
    ).pixel_values
    return examples


class MaskedAutoencoderDataset(Dataset):
    def __init__(
        self,
        dataset,
        masking_ratio: float = 0.01,
        dataset_name="CIFAR10",
        num_one_hot_classes=2**8 + 6,
    ):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.masking_ratio = masking_ratio
        self.num_one_hot_classes = 262  # Only used for IMDB dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        full_input = self.dataset[idx]
        if self.dataset_name == "CIFAR10":
            full_input = full_input["pixel_values"].flatten(start_dim=1)
        elif self.dataset_name == "IMDB":
            full_input = torch.t(
                F.one_hot(full_input["input_ids"], self.num_one_hot_classes).float()
            )

        input_length = full_input.shape[1]
        number_unmasked_indices = np.floor(input_length * self.masking_ratio).astype(
            int
        )

        masked_input = torch.zeros_like(full_input)
        unmasked_indices = np.random.choice(input_length, number_unmasked_indices)
        masked_input[:, unmasked_indices] = full_input[:, unmasked_indices]
        return masked_input, full_input, unmasked_indices


def _prepare_CIFAR_input():
    # Load the dataset from HuggingFace
    train_ds, test_ds = load_dataset("cifar10", split=["train", "test"])
    # Split up training into training + validation
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits["train"]
    val_ds = splits["test"]

    train_ds.set_transform(preprocess_images)
    val_ds.set_transform(preprocess_images)
    test_ds.set_transform(preprocess_images)

    return train_ds, val_ds, test_ds


def _prepare_IMDB_input():
    # Load the dataset from HuggingFace
    train_ds, test_ds = load_dataset("imdb", split=["train", "test"])
    # Split up training into training + validation
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits["train"]
    val_ds = splits["test"]

    tokenizer = PerceiverTokenizer.from_pretrained("deepmind/language-perceiver")
    train_ds = train_ds.map(
        lambda examples: tokenizer(
            examples["text"], padding="max_length", truncation=True
        ),
        batched=True,
    )
    val_ds = val_ds.map(
        lambda examples: tokenizer(
            examples["text"], padding="max_length", truncation=True
        ),
        batched=True,
    )
    test_ds = test_ds.map(
        lambda examples: tokenizer(
            examples["text"], padding="max_length", truncation=True
        ),
        batched=True,
    )

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return train_ds, val_ds, test_ds


def get_data(dataset: str = "CIFAR10", masking_ratio: float = 0.01):

    if dataset not in SUPPORTED_DATASETS:
        raise RuntimeError(f"The dataset {dataset} is not supported.")

    if dataset == "CIFAR10":
        train_ds, val_ds, test_ds = _prepare_CIFAR_input()
    elif dataset == "IMDB":
        train_ds, val_ds, test_ds = _prepare_IMDB_input()

    train_ds = MaskedAutoencoderDataset(train_ds, masking_ratio, dataset_name=dataset)
    val_ds = MaskedAutoencoderDataset(val_ds, masking_ratio, dataset_name=dataset)
    test_ds = MaskedAutoencoderDataset(test_ds, masking_ratio, dataset_name=dataset)

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
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=8)

    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=8)

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
