#!/usr/bin/env python
# coding: utf-8

from datasets import load_dataset

train_ds, test_ds = load_dataset("imdb", split=["train", "test"])

labels = train_ds.features["label"].names
print(labels)


id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}
print(id2label)


from transformers import PerceiverTokenizer

tokenizer = PerceiverTokenizer.from_pretrained("deepmind/language-perceiver")

train_ds = train_ds.map(
    lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True),
    batched=True,
)
test_ds = test_ds.map(
    lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True),
    batched=True,
)


train_ds.set_format(type="torch", columns=["input_ids", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "label"])


from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_ds, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=50)


batch = next(iter(train_dataloader))
for k, v in batch.items():
    print(k, v.shape)


import numpy as np

train_ds["label"].double().mean()


# ## Define model
#
# Next, we define our model, and put it on the GPU.


# preprocessor we customized to use the tagkop encoder
from tagkop_encoding_functions import (
    PerceiverImagePreprocessor,
    TagkopPerceiverTextPreprocessor,
)
from transformers import PerceiverForSequenceClassification

import torch

from transformers.models.perceiver.modeling_perceiver import (
    PerceiverConfig,
    PerceiverModel,
    PerceiverClassificationDecoder,
    PerceiverTextPreprocessor,
    PerceiverClassificationDecoder,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = PerceiverConfig(num_self_attends_per_block=4, d_model=64)

print("config", config)

# Vanilla Perceiver Encodings


preprocessor = PerceiverTextPreprocessor(config)

# Our new awesome encodings
# preprocessor = TagkopPerceiverTextPreprocessor(config)


# preprocessor = PerceiverImagePreprocessor(config,
#                                           in_channels=1,
#                                           prep_type="1d",
#                                           position_encoding_type="fourier",


#                                           concat_or_add_pos="add",
#                                           out_channels=64,
#                                           project_pos_dim=64,
#                                           # tagkop_position_encoding_kwargs=dict(
#                                           #   num_channels=64,
#                                           #   index_dims=config.image_size**2,
#                                           #   ds="imdb"
#                                           #   ),
#                                           fourier_position_encoding_kwargs = dict(
#                                               concat_pos=False, max_resolution=(224, 224), num_bands=16, sine_only=False
#                                           )
#                                       )

decoder = PerceiverClassificationDecoder(
    config,
    num_channels=config.d_latents,
    trainable_position_encoding_kwargs=dict(
        num_channels=config.d_latents, index_dims=1
    ),
    use_query_residual=True,
)

# num_self_attends_per_block, num_self_attention_heads, num_cross_attention_heads to something more reasonable and out_channels project_pos_dim and num_channels to 64
model = PerceiverModel(config, input_preprocessor=preprocessor, decoder=decoder)


model.to(device)


# In[37]:


# # you can then do a forward pass as follows:
# tokenizer = PerceiverTokenizer()
# text = "hello world"
# inputs = tokenizer(text, return_tensors="pt").input_ids
# print(inputs)
# inputs.to(device)
# with torch.no_grad():
#    outputs = model(inputs=inputs.unsqueeze(1).to(device))
# logits = outputs.logits
# print('list(logits.shape): ', list(logits.shape))
# # to train, one can train the model using standard cross-entropy:
# criterion = torch.nn.CrossEntropyLoss()
# labels = torch.tensor([1]).to(device)
# loss = criterion(logits, labels)


# ## Train the model
#
# Here we train the model using native PyTorch.

# In[19]:


from transformers import AdamW
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

optimizer = AdamW(model.parameters(), lr=1e-4)

model.train()


batch = next(iter(train_dataloader))
for epoch in range(100):  # loop over the dataset multiple times
    print("saved model")
    print("Epoch:", epoch)
    for i in range(10):
        # for batch in tqdm(train_dataloader):
        # get the inputs;
        inputs = batch["input_ids"].to(device)
        #  attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs=inputs)
        logits = outputs.logits

        # to train, one can train the model using standard cross-entropy:
        criterion = torch.nn.CrossEntropyLoss()

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # evaluate
        predictions = outputs.logits.argmax(-1).cpu().detach().numpy()
        accuracy = accuracy_score(y_true=batch["label"].numpy(), y_pred=predictions)
        print(f"Loss: {loss.item()}, Accuracy: {accuracy}")
    torch.save(model.state_dict(), f"imdb_network_model_fourier_{epoch}.pt")


# ## Evaluate the model
#
# Finally, we evaluate the model on the test set. We use the Datasets library to compute the accuracy.

# In[40]:


# torch.save(model.state_dict(), '/content/drive/MyDrive/saved_model/small_network_model_fourier_embeddings.pt')


# In[15]:


# In[16]:


# import torch
# checkpoint = torch.load('/content/drive/MyDrive/saved_model/small_network_model.pt')
# model.load_state_dict(checkpoint)
# model.eval()


# In[18]:


from tqdm.notebook import tqdm
from datasets import load_metric

accuracy = load_metric("accuracy")

model.eval()
for batch in tqdm(test_dataloader):
    # get the inputs;
    inputs = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)

    # forward pass
    outputs = model(inputs=inputs, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = logits.argmax(-1).cpu().detach().numpy()
    references = batch["label"].numpy()
    accuracy.add_batch(predictions=predictions, references=references)

final_score = accuracy.compute()
print("Accuracy on test set:", final_score)


# In[45]:


# TAGOP
from tqdm.notebook import tqdm
from datasets import load_metric

accuracy = load_metric("accuracy")

model.eval()
for batch in tqdm(test_dataloader):
    # get the inputs;
    inputs = batch["input_ids"].to(device)
    labels = batch["label"].to(device)

    # forward pass
    outputs = model(inputs=inputs.unsqueeze(1))
    logits = outputs.logits
    predictions = logits.argmax(-1).cpu().detach().numpy()
    references = batch["label"].numpy()
    accuracy.add_batch(predictions=predictions, references=references)

final_score = accuracy.compute()
print("Accuracy on test set:", final_score)


# ## Inference

# In[22]:


text = "I hated this movie, it's really bad."

input_ids = tokenizer(text, return_tensors="pt").input_ids

# forward pass
outputs = model(inputs=input_ids.to(device))
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()

print("Predicted:", model.config.id2label[predicted_class_idx])


# In[ ]:
