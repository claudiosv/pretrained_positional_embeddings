# starting point for VAE: https://github.com/pytorch/examples/blob/main/vae/main.py

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms

torch.manual_seed(289)
batch_size = 128
epochs = 10
logging_interval = 10
sub_sampling_size = 100
original_size = 784

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

kwargs = {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=False, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    **kwargs
)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(sub_sampling_size, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, original_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, sub_sampling_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
model.load_state_dict(torch.load("dummy_vae.pth"))
model.eval()

import random


def build_sample_idx(batch_size, data_size, sample_size):
    all_idx = []
    for b in range(batch_size):
        idx = [
            x + (b * data_size)
            for x in sorted(random.sample(range(data_size), sample_size))
        ]
        all_idx += idx
    return torch.tensor(all_idx)


# 1 row per pixel
# 784 columns per row
# 2 dimensions to track total loss and number of updates
weights = torch.zeros([original_size, 2, original_size])
# array for iterating counter
iterator = torch.ones(original_size)

# object to calculate reconstruction loss
mseloss = nn.MSELoss(reduction="none")

for batch_idx, (data, _) in enumerate(test_loader):
    data = data.to(device)
    # flatten the images
    flat_data = data.view(batch_size * original_size)
    # randomly get subset of inputs
    indices = build_sample_idx(batch_size, original_size, sub_sampling_size)
    # subsample the original data
    subset = flat_data[indices]
    # reshape
    subset = subset.view(batch_size, 1, sub_sampling_size)
    indices = indices.view(batch_size, sub_sampling_size)
    # pass the sampled data to the model
    recon_batch, mu, logvar = model(subset)
    # go through each image in the current batch
    for i in range(batch_size):
        # get the difference between the original image and the model output for each index
        recon_loss = mseloss(data[i].view(-1), recon_batch[i]).data
        # get the indices that were in the sample
        idxs = indices[i].tolist()
        # recenter the indices
        idxs = [x - (i * original_size) for x in idxs]
        # add the losses to the index's loss tracker
        # iterate the number of samples seen
        for idx in idxs:
            weights[idx, 0] += recon_loss
            weights[idx, 1] += iterator

# total loss / number of times seen
average_weight = weights[:, 0, :] / weights[:, 1, :]

# normalize across rows
normalized_weights = F.normalize(average_weight, dim=0)

torch.save(normalized_weights, "embedding_tensor.pth")
