import argparse
import data.utils as utils
import gc
import logging
import os
import pytorch_lightning as pl
import random
import torch
import torch.nn.functional as F
import wandb
import warnings
import yaml

from data.dataset import SegmentationTrainingDataset
from segmentation.model import create_segmentation_model
from segmentation_models_pytorch.losses import DiceLoss
from training.utils import set_seed
from pytorch_lightning.callbacks import (
    EarlyStopping, 
    GradientAccumulationScheduler,
    ModelCheckpoint, 
    StochasticWeightAveraging
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

warnings.filterwarnings("ignore") 


def mixup(x, y, clip=[0.0, 1.0]):
    """Mix-up augmentation."""
    indices = torch.randperm(x.size(0))
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    alpha = random.uniform(clip[0], clip[1])
    x = alpha * x + (1.0 - alpha) * x_shuffled
    return x, y, y_shuffled, alpha


class SegmentationModule(pl.LightningModule):
    """PyTorch Lightning module for training a segmentation model."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = create_segmentation_model(config["model"])
        self.loss_weights = self.config["losses"]["weights"]
        self.prob_mixup = self.config["losses"]["prob_mixup"]
        self.criteria = self.config["losses"]["criteria"]

    def configure_optimizers(self):
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
    
    def compute_loss(self, logits, y, criterion):
        if criterion["type"] == "bce":
            pos_weight = torch.tensor([criterion[f"pos_weight"]]).to(self.device)
            return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="mean")
        elif criterion["type"] == "dice":
            return DiceLoss(mode="multilabel", from_logits=True, smooth=criterion["label_smoothing"], ignore_index=-100)(logits, y)
        else:
            raise ValueError(f"Criterion type `{criterion['type']}` is not supported.")
        
    def compute_losses(self, logits, y):
        losses = []
        for criterion in self.criteria.values():
            losses.append(self.compute_loss(logits, y, criterion))
        return losses
    
    def step(self, batch, do_mixup=False):
        x = batch["image"]
        y = batch["mask"]
        if do_mixup and random.uniform(0, 1) < self.prob_mixup:
            x, y, y_shuffled, alpha = mixup(x, y)
            logits = self.model(x)
            losses_0 = self.compute_losses(logits, y)
            losses_1 = self.compute_losses(logits, y_shuffled)
            losses = [alpha * l0 + (1.0 - alpha) * l1 for l0, l1 in zip(losses_0, losses_1)]
        else:
            logits = self.model(x)
            losses = self.compute_losses(logits, y)
        return losses

    def training_step(self, batch, batch_idx):
        losses = self.step(batch, do_mixup=True)
        loss = sum([w * l for w, l in zip(self.loss_weights, losses)])
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("lr", lr, on_step=True, on_epoch=False)
        for loss_type, l in zip(self.criteria, losses):
            self.log(f"train_loss_{loss_type}", l, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        losses = self.step(batch)
        loss = sum([w * l for w, l in zip(self.loss_weights, losses)])
        self.log("valid_loss", loss, on_step=False, on_epoch=True)
        for loss_type, l in zip(self.criteria, losses):
            self.log(f"valid_loss_{loss_type}", l, on_step=False, on_epoch=True)
        

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/romainlhardy/kaggle/rsna-abdominal-trauma/configs/segmentation/config.yaml", required=True)
    args = parser.parse_args()
    return args


def train(fold, config):
    """Trains a segmentation model on a specified fold."""
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

    train_dataset = SegmentationTrainingDataset(df_train, split="train")
    valid_dataset = SegmentationTrainingDataset(df_valid, split="validation")
    
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
    name = f"{model_config['model']['encoder_params']['encoder_name'].replace('/', '_')}__{'_'.join([str(d) for d in utils.SEGMENTATION_DIM])}__seed_{config['seed']}__fold_{fold}"
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        monitor="valid_loss",
        dirpath=config["output_dir"],
        mode="min",
        filename=name,
        save_top_k=1,
        verbose=1,
    )

    early_stopping_callback = EarlyStopping(monitor="valid_loss", **config["early_stopping"])
    accumulate_callback = GradientAccumulationScheduler(**config["accumulate"])
    swa_callback = StochasticWeightAveraging(**config["swa"])

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback, accumulate_callback, swa_callback],
        logger=WandbLogger(name=name, project=config["wandb_project"], save_dir=f"/home/romainlhardy/kaggle/rsna-abdominal-trauma/logs"),
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