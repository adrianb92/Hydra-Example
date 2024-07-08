import os
import sys
import logging
import hydra
from hydra.utils import instantiate
from hydra.utils import get_original_cwd, to_absolute_path

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

import torch
# lightning related imports
import pytorch_lightning as pl
#from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
    


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    data_module = instantiate(cfg.data) 
    data_module.prepare_data()
    data_module.setup()

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=False,
        mode='min'
    )

    MODEL_CKPT_PATH = 'model/'
    MODEL_CKPT = 'model-{epoch:02d}-{val_loss:.2f}'

    checkpoint_callback = instantiate(cfg.callbacks)

    # Init our model
    model = instantiate(cfg.network) 

    # Initialize logger
    tensorboard_logger =  TensorBoardLogger("lightning_logs", name="alien_predator")

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=cfg.trainer.max_epochs, 
                     logger=tensorboard_logger,
                     callbacks=[checkpoint_callback, early_stop_callback],
                     )

    # Train the model
    trainer.fit(model, data_module)

    # Evaluate the model on the held out test set ⚡⚡
    trainer.test(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()

