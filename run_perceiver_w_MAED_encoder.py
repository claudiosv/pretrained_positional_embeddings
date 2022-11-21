#!/usr/bin/env python
# coding: utf-8

from datasets import load_dataset
from transformers import PerceiverFeatureExtractor,PerceiverForImageClassificationLearned
from torch.utils.data import DataLoader
import torch
import numpy as np
from datasets import load_from_disk
from torch.optim import AdamW
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

print(f"CUDA: {torch.cuda.is_available()}")
print(torch.cuda.device_count())
device = torch.device("cuda")

train_ds = load_from_disk("train_ds")
val_ds = load_from_disk("val_ds")
test_ds = load_from_disk("test_ds")

id2label = {idx: label for idx, label in enumerate(train_ds.features["label"].names)}
label2id = {label: idx for idx, label in id2label.items()}

feature_extractor = PerceiverFeatureExtractor()


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


# preprocessor we customized to use the MAED encoder
from MAED_encoding_functions import PerceiverImagePreprocessor,PerceiverMAEDPositionEncoding

# perceiver modules from hugging face
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverConfig,
    PerceiverModel,
    PerceiverClassificationDecoder,
)

# config = PerceiverConfig(
#     # d_model=224,
#     image_size=224,
#     num_self_attends_per_block=13,  # 26
#     num_self_attention_heads=4,  # 8
#     num_cross_attention_heads=4,
#     use_labels=True,
#     num_labels=10,
#     num_latents=256,
#     id2label=id2label,
#     label2id=label2id,
#     ignore_mismatched_sizes=True,
# )

# preprocessor = PerceiverImagePreprocessor(
#     config,
#     in_channels=3,
#     prep_type="conv1x1",
#     spatial_downsample=1,
#     out_channels=64,
#     # position_encoding_type="MAED",
#     concat_or_add_pos="add",
#     project_pos_dim=64,
#     MAED_position_encoding_kwargs=dict(
#         num_channels=64,
#         index_dims=config.image_size**2,
#         dataset="cifar"
#     ),
# )

# modely = PerceiverModel(
#     config,
#     input_preprocessor=preprocessor,
#     decoder=PerceiverClassificationDecoder(
#         config,
#         num_channels=config.d_latents,
#         trainable_position_encoding_kwargs=dict(
#             num_channels=config.d_latents, index_dims=1
#         ),
#         use_query_residual=True,
#     ),
# ).from_pretrained("deepmind/vision-perceiver-learned")
model = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned",
                                                                num_labels=10,
                                                                id2label=id2label,
                                                                label2id=label2id,
                                                                ignore_mismatched_sizes=True)
model.perceiver.input_preprocessor.position_embeddings = PerceiverMAEDPositionEncoding(224**2)
print(dir(model))
#.position_embeddings.position_embeddings = torch.nn.Parameter(torch.randn(model.perceiver.input_preprocessor.position_embeddings.position_embeddings.shape))
modely = model
# modely.config.d_model=224
modely.to(device)

optimizer = AdamW(modely.parameters(), lr=1e-4)

modely.train()

for epoch in range(100):
    print("Epoch:", epoch)
    for batch in tqdm(train_dataloader):
        # get the inputs;
        inputs = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = modely(inputs=inputs)
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
    if epoch % 3 == 0:
        torch.save(modely.state_dict(), f"MAED_cifar_{epoch}.pt")
from datasets import load_metric
accuracy = load_metric("accuracy")

modely.eval()
for batch in tqdm(val_dataloader):
    # get the inputs;
    inputs = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)

    # forward pass
    outputs = modely(inputs=inputs)
    logits = outputs.logits
    predictions = logits.argmax(-1).cpu().detach().numpy()
    references = batch["labels"].numpy()
    accuracy.add_batch(predictions=predictions, references=references)

final_score = accuracy.compute()
print("Accuracy on test set:", final_score)
