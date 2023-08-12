import argparse
import data.utils as utils
import gc
import logging
import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
import warnings
import yaml

from data.dataset import SegmentationDataset
from segmentation.model import create_segmentation_model
from training.utils import set_seed
from pytorch_lightning.callbacks import (
    EarlyStopping, 
    GradientAccumulationScheduler,
    ModelCheckpoint, 
    StochasticWeightAveraging
)
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy, fbeta_score
from torch.utils.data import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

warnings.filterwarnings("ignore") 


class SegmentationModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = create_segmentation_model(config["model"])
        self.logit_tracker = []
        self.label_tracker = []

    def configure_optimizers(self):
        lr = self.config["optimizer"]["lr"]
        optimizer = torch.optim.AdamW(
            self.parameters(),
            **self.config["optimizer"]
        )
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            **self.config["scheduler"],
        )
        lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
    
    def compute_losses(self, logits, y):
        pos_weight = torch.tensor([self.config["losses"]["bce"]["pos_weight"]]).to(self.device)
        bce_loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="mean")
        return bce_loss

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"].float()
        logits = self.model(x)
        loss = self.compute_losses(logits, y)
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("lr", lr, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"].float()
        logits = self.model(x)
        loss = self.compute_losses(logits, y)
        self.logit_tracker.append(logits)
        self.label_tracker.append(y)
        self.log("valid_loss", loss, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        logits = torch.cat(self.logit_tracker)
        labels = torch.cat(self.label_tracker)
        valid_f1 = fbeta_score(logits, labels.long(), task="multilabel", threshold=0.3, beta=1.)
        valid_accuracy = accuracy(logits, labels.long(), task="multilabel", threshold=0.3)
        self.log("valid_f1", valid_f1, on_step=False, on_epoch=True)
        self.log("valid_accuracy", valid_accuracy, on_step=False, on_epoch=True)
        self.logit_tracker.clear()
        self.label_tracker.clear()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/romainlhardy/kaggle/rsna-abdominal-trauma/configs/segmentation/config.yaml", required=True)
    args = parser.parse_args()
    return args


def train(fold, config):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if config["seed"] is not None:
        set_seed(config["seed"])

    df = utils.data_split_segmentation(os.path.join(utils.DATA_DIR, "data_split_segmentation.csv"))
    df_train = df[df.fold != fold]
    df_valid = df[df.fold == fold]

    image_size = config["model"]["data"]["image_size"]
    train_dataset = SegmentationDataset(
        df_train,
        image_size=image_size,
        split="train"
    )
    valid_dataset = SegmentationDataset(
        df_valid,
        image_size=image_size,
        split="validation"
    )
    
    logging.info(f"[FOLD {fold}]")
    logging.info(f"Training images: {len(train_dataset)}")
    logging.info(f"Validation images: {len(valid_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["train_batch_size"], 
        sampler=None,
        shuffle=True,
        num_workers=config["num_workers"]
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        shuffle=False, 
        batch_size=config["valid_batch_size"],
        num_workers=config["num_workers"]
    )

    model_config = config["model"]
    name = f"{model_config['model']['encoder_params']['model_name'].replace('/', '_')}__image_size_{image_size}__seed_{config['seed']}__fold_{fold}"
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        monitor="valid_f1",
        dirpath=config["output_dir"],
        mode="max",
        filename=name,
        save_top_k=1,
        verbose=1,
    )

    early_stopping_callback = EarlyStopping(monitor="valid_f1", **config["early_stopping"])
    accumulate_callback = GradientAccumulationScheduler(**config["accumulate"])
    swa_callback = StochasticWeightAveraging(**config["swa"])

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback, accumulate_callback, swa_callback],
        logger=WandbLogger(name=name, project=config["wandb_project"], save_dir=f"/home/romainlhardy/kaggle/contrails/logs"),
        **config["trainer"],
    )
    trainer.logger.log_hyperparams(config)

    model = SegmentationModule(config["model"])

    trainer.fit(model, train_dataloader, valid_dataloader)

    wandb.finish()
    del trainer, model
    gc.collect()


def main():
    torch.set_float32_matmul_precision("medium")
    args = parse_args()
    with open(args.config, "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for fold in config["folds"]:
        train(fold, config)
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()