# starting point for VAE: https://github.com/pytorch/examples/blob/main/vae/main.py

from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

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

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)


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
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, sub_sampling_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, original_size), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

import random
# function to randomly select subsets of each image
def build_sample_idx(batch_size, data_size, sample_size):
    # list to keep track of all sample indices
    all_idx = []
    # make a sample for each image in the batch
    for b in range(batch_size):
        # offset the sample indices by the size of an image and where we are in the batch
        idx = [x+(b*data_size) for x in sorted(random.sample(range(data_size), sample_size))]
        all_idx += idx
    # convert to tensor
    return torch.tensor(all_idx)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        # flatten the images
        data = data.view(batch_size * original_size)
        # randomly get subset of inputs
        indices = build_sample_idx(batch_size, original_size, sub_sampling_size)
        # subsample the original data
        subset = data[indices]
        # reshape
        subset = subset.view(batch_size, 1, sub_sampling_size)
        optimizer.zero_grad()
        # pass the samples data to the model
        recon_batch, mu, logvar = model(subset)
        # validate model with orignal data
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % logging_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            # flatten the images
            flat_data = data.view(batch_size * original_size)
            # randomly get subset of inputs
            indices = build_sample_idx(batch_size, original_size, sub_sampling_size)
            # subsample the original data
            subset = flat_data[indices]
            # reshape
            subset = subset.view(batch_size, 1, sub_sampling_size)
            recon_batch, mu, logvar = model(subset)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         "dummy_results/reconstruction_" + str(epoch) + ".png", nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       "dummy_results/sample_" + str(epoch) + ".png")
    torch.save(model.state_dict(), "dummy_vae.pth")
