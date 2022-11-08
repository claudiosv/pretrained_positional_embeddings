from auto_encoder import AutoEncoder
from masked_dataset import CIFAR10DataModule, IMDBDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
from pytorch_lightning.loggers import TensorBoardLogger

cifar10 = CIFAR10DataModule()

in_channel_size = 3
num_epochs = 10
model = AutoEncoder(in_channel_size=3)

trainer = pl.Trainer(
    default_root_dir=os.getcwd(),
    logger=TensorBoardLogger(save_dir=os.getcwd(),
                             version=1, name="lightning_logs"),
    auto_lr_find=True,
    auto_scale_batch_size=True,
    accelerator="auto",
    log_every_n_steps=1,
    max_epochs=50,
    min_epochs=10,
    callbacks=[
        ModelCheckpoint(save_last=True),
        LearningRateMonitor("epoch"),
        EarlyStopping(monitor="val_loss", mode="min", patience=5)
    ])
trainer.logger._log_graph = True

# Basic hyperparameter tuning for batch size and lr
trainer.tune(model, cifar10)

# Run the training
trainer.fit(model, cifar10)
