# Adapted from https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/PyTorch/unet.py
# The original implementation is a basic UNet architecture that works on images. We modified model to take any
# 1-Dimensional (linearized) input

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_c)

        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class TextEncoderBlock(nn.Module):
    def __init__(self, out_c):
        super().__init__()

        self.embeddings = nn.Embedding(
            num_embeddings=262, embedding_dim=out_c)
        self.pool = nn.MaxPool1d(2)

    def forward(self, inputs):
        x = self.embeddings(inputs)
        p = self.pool(x)

        return x, p


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = ConvolutionalBlock(in_c, out_c)
        self.pool = nn.MaxPool1d(2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose1d(
            in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvolutionalBlock(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class AutoEncoder(pl.LightningModule):
    def __init__(self, is_text_input=False, in_channel_size=3):
        super().__init__()
        self.is_text_input = is_text_input

        """ If the input is text ids, we use TextEncoder, else we use a ConvEncoder as the first module """
        if is_text_input:
            self.e1 = TextEncoderBlock(64)
        else:
            self.e1 = EncoderBlock(in_channel_size, 64)

        """ Encoder """
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        """ Bottleneck """
        self.b = ConvolutionalBlock(512, 1024)

        """ Decoder """
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        """ Classifier """
        self.outputs = nn.Conv1d(64, in_channel_size, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)

        return outputs

    def _get_reconstruction_loss(self, batch):
        masked, full, _ = batch
        pred = self.forward(masked)
        loss = nn.functional.mse_loss(pred, full)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


if __name__ == "__main__":

    in_channel_size = 3
    input_length = 100000
    inputs = torch.randn((2, in_channel_size, input_length))
    model = AutoEncoder(in_channel_size=in_channel_size)
    y = model(inputs)
    print(inputs.shape, y.shape)
