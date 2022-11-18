#!/usr/bin/env python
# coding: utf-8

# ## Set-up environment
#
# As usual, we first install HuggingFace Transformers, and Datasets.

# In[2]:


gpu_info = get_ipython().getoutput("nvidia-smi")
gpu_info = "\n".join(gpu_info)
if gpu_info.find("failed") >= 0:
    print("Not connected to a GPU")
else:
    print(gpu_info)


# In[ ]:


from google.colab import drive

drive.mount("/content/drive")


# In[3]:


get_ipython().system(
    "pip install -q git+https://github.com/huggingface/transformers.git"
)


# In[4]:


get_ipython().system("pip install -q datasets")


# ## Prepare data
#
# Here we take a small portion of the IMDB dataset, a binary text classification dataset ("is a movie review positive or negative?").

# In[5]:


from datasets import load_dataset

train_ds, test_ds = load_dataset("imdb", split=["train", "test"])
# train_ds, test_ds = load_dataset("imdb", split=['train[:10]+train[-10:]', 'test[:5]+test[-5:]'])


# We create id2label and label2id mappings, which are handy at inference time.

# In[6]:


labels = train_ds.features["label"].names
print(labels)


# In[7]:


id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}
print(id2label)


# Next, we prepare the data for the model using the tokenizer.

# In[8]:


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


# We set the format to PyTorch tensors, and create familiar PyTorch dataloaders.

# In[9]:


train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# In[10]:


from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_ds, batch_size=100, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=50)


# Here we verify some things (always important to check out your data!).

# In[11]:


batch = next(iter(train_dataloader))
for k, v in batch.items():
    print(k, v.shape)


# In[12]:


tokenizer.decode(batch["input_ids"][3])


# In[13]:


batch["label"]


# ## Define model
#
# Next, we define our model, and put it on the GPU.

# In[14]:


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


config = PerceiverConfig(num_self_attends_per_block=4)
preprocessor = PerceiverTextPreprocessor(config)
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


# In[14]:


# you can then do a forward pass as follows:
tokenizer = PerceiverTokenizer()
text = "hello world"
inputs = tokenizer(text, return_tensors="pt").input_ids
inputs.to(device)
with torch.no_grad():
    outputs = model(inputs=inputs.to(device))
logits = outputs.logits
print("list(logits.shape): ", list(logits.shape))
# to train, one can train the model using standard cross-entropy:
criterion = torch.nn.CrossEntropyLoss()
labels = torch.tensor([1]).to(device)
loss = criterion(logits, labels)


# ## Train the model
#
# Here we train the model using native PyTorch.

# In[17]:


from transformers import AdamW
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(10):  # loop over the dataset multiple times
    torch.save(
        model.state_dict(), "/content/drive/MyDrive/saved_model/small_network_model.pt"
    )
    print("saved model")
    print("Epoch:", epoch)
    for batch in tqdm(train_dataloader):
        # get the inputs;
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs=inputs, attention_mask=attention_mask)
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


# ## Evaluate the model
#
# Finally, we evaluate the model on the test set. We use the Datasets library to compute the accuracy.

# In[23]:


torch.save(
    model.state_dict(),
    "/content/drive/MyDrive/saved_model/small_network_model_checkpoint.pt",
)


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
