#!/usr/bin/env python
# coding: utf-8

from datasets import load_dataset
from transformers import PerceiverFeatureExtractor
from torch.utils.data import DataLoader
import torch
import numpy as np
from datasets import load_from_disk
from torch.optim import AdamW
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score
from transformers import AdamW
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

print(f"CUDA Available?: {torch.cuda.is_available()}")
print(f"Number of CUDA devices visible: {torch.cuda.device_count()}")
device = torch.device("cuda")

train_ds = load_from_disk("train_ds")
val_ds = load_from_disk("val_ds")
test_ds = load_from_disk("test_ds")


id2label = {idx: label for idx, label in enumerate(train_ds.features["label"].names)}
label2id = {label: idx for idx, label in id2label.items()}

# We can prepare the data for the model using the feature extractor.
#
# Note that this feature extractor is fairly basic: it will just do center cropping + resizing + normalizing of the color channels.
#
# One should actually add several data augmentations (available in libraries like [torchvision](https://pytorch.org/vision/stable/transforms.html) and [albumentations](https://albumentations.ai/) to achieve greater results. I refer to my [ViT notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_the_%F0%9F%A4%97_Trainer.ipynb) for an example.

feature_extractor = PerceiverFeatureExtractor()


# Note that HuggingFace Datasets has an Image feature, meaning that every image is a PIL (Pillow) image by default. The feature extractor will turn each Pillow image into a PyTorch tensor of shape (3, 224, 224).
#
# Note that Apache Arrow (which HuggingFace Datasets uses as a back-end) doesn't know PyTorch Tensors, but we can escape it by using the `set_transform` method on the Dataset, which allows to only prepare images when we need them (i.e. on-the-fly). This is awesome as it saves memory! Refer to the [docs](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.set_transform) for more information.


def preprocess_images(examples):
    examples["pixel_values"] = feature_extractor(
        examples["img"], return_tensors="pt"
    ).pixel_values
    return examples


# Set the transforms
train_ds.set_transform(preprocess_images)
val_ds.set_transform(preprocess_images)
test_ds.set_transform(preprocess_images)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


train_batch_size = 10
eval_batch_size = 10

train_dataloader = DataLoader(
    train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size
)
val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size)
test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size)

batch = next(iter(train_dataloader))
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)


assert batch["pixel_values"].shape == (train_batch_size, 3, 224, 224)
assert batch["labels"].shape == (train_batch_size,)


# ## Define model
#
# Here we only replace the final projection layer of the decoder (`PerceiverClassificationDecoder`) of the checkpoint that was trained on ImageNet. This means that we will use the same (learned) output queries as before, hence the cross-attention operation will give the same output. However, the final projection layer has 1000 output neurons during pre-training, while we only have 10.
#
# NOTE: note that the Perceiver has 3 variants for image classification:
# * PerceiverForImageClassificationLearned
# * PerceiverForImageClassificationFourier
# * PerceiverForImageClassificationConvProcessing.
#
# Here I'm using the first one, which adds learned 1D position embeddings to the pixel values. Note that the best results will be obtained with the latter.
#
# For in-depth understanding on how the Perceiver works, I refer to my [blog post](https://huggingface.co/blog/perceiver).
#
# We can use the handy `ignore_mismatched_sizes` to replace the head. We also set the `id2label` and `label2id` mappings we defined earlier (which will be handy when doing inference).

from transformers import (
    PerceiverConfig,
    PerceiverTokenizer,
    PerceiverFeatureExtractor,
    PerceiverModel,
)
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverTextPreprocessor,
    PerceiverImagePreprocessor,
    PerceiverClassificationDecoder,
)
import torch
import requests

# EXAMPLE 2: using the Perceiver to classify images
# - we define an ImagePreprocessor, which can be used to embed images
config = PerceiverConfig(
    image_size=224,
    use_labels=True,
    num_labels=10,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    num_self_attends_per_block=13,  # 26
    num_self_attention_heads=4,  # 8
    num_cross_attention_heads=4,  # 8
)
CHANNELS = 64
preprocessor = PerceiverImagePreprocessor(
    config,
    prep_type="conv1x1",
    spatial_downsample=1,
    out_channels=CHANNELS,
    position_encoding_type="trainable",
    concat_or_add_pos="concat",
    project_pos_dim=CHANNELS,
    trainable_position_encoding_kwargs=dict(
        num_channels=CHANNELS,
        index_dims=config.image_size**2,
    ),
)

model = PerceiverModel(
    config,
    input_preprocessor=preprocessor,
    decoder=PerceiverClassificationDecoder(
        config,
        num_channels=config.d_latents,
        trainable_position_encoding_kwargs=dict(
            num_channels=config.d_latents, index_dims=1
        ),
        use_query_residual=True,
    ),
)

model.to(device)


# ## Train the model
#
# Here we train the model using native PyTorch.

optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(10):
    print("Epoch:", epoch)
    for batch in tqdm(train_dataloader):
        # get the inputs;
        inputs = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs=inputs)
        logits = outputs.logits

        criterion = torch.nn.CrossEntropyLoss()

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = outputs.logits.argmax(-1).cpu().detach().numpy()
        accuracy = accuracy_score(y_true=batch["labels"].numpy(), y_pred=predictions)
        print(f"Loss: {loss.item()}, Accuracy: {accuracy}")

    predictions = outputs.logits.argmax(-1).cpu().detach().numpy()
    accuracy = accuracy_score(y_true=batch["labels"].numpy(), y_pred=predictions)
    print(f"Loss: {loss.item()}, Accuracy: {accuracy}")

accuracy = load_metric("accuracy")

model.eval()
for batch in tqdm(val_dataloader):
    # get the inputs;
    inputs = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)

    # forward pass
    outputs = model(inputs=inputs)
    logits = outputs.logits
    predictions = logits.argmax(-1).cpu().detach().numpy()
    references = batch["labels"].numpy()
    accuracy.add_batch(predictions=predictions, references=references)

final_score = accuracy.compute()
print("Accuracy on test set:", final_score)
