import argparse
import data.utils as utils
import gc
import logging
import numpy as np
import os
import pytorch_lightning as pl
import random
import torch
import torch.nn.functional as F
import wandb
import warnings
import yaml

from data.dataset import SliceClassificationTrainingDataset, INJURY_CATEGORIES
from pytorch_lightning.callbacks import (
    EarlyStopping, 
    GradientAccumulationScheduler,
    ModelCheckpoint, 
    StochasticWeightAveraging
)
from pytorch_lightning.loggers import WandbLogger
from scripts.segmentation.train import SegmentationModule
from sklearn.utils import compute_class_weight
from slice_classification.model import create_slice_classification_model
from training.utils import set_seed
from torchmetrics.functional import accuracy, auroc, fbeta_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

warnings.filterwarnings("ignore") 


def mixup(x, y_list, clip=[0.0, 1.0]):
    """Mix-up augmentation."""
    indices = torch.randperm(x.size(0))
    x_shuffled = x[indices]
    y_list_shuffled = [y[indices] for y in y_list]
    alpha = random.uniform(clip[0], clip[1])
    x = alpha * x + (1.0 - alpha) * x_shuffled
    return x, y_list, y_list_shuffled, alpha


class SliceClassificationModule(pl.LightningModule):
    """PyTorch Lightning module for training a slice-level classification model."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = create_slice_classification_model(config["model"])
        self.clf_criteria = self.config["losses"]["classification"]
        self.seg_criteria = self.config["losses"]["segmentation"]
        self.alpha = self.config["losses"]["alpha"]
        self.prob_mixup = self.config["losses"]["prob_mixup"]
        self.logits_tracker = []
        self.labels_tracker = []

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

    def compute_auroc(self, logits_list, y_list):
        num_classes = [2, 2, 3, 4, 4, 4]
        roc_list = []
        for logits, y, nc in zip(logits_list, y_list, num_classes):
            roc_list.append(
                auroc(logits, y, task="multiclass", num_classes=nc)
            )
        return sum(roc_list) / len(roc_list)

    def compute_clf_loss(self, logits, y, criterion, split="train"):
        if criterion["type"] == "bce":
            pos_weight = torch.tensor([criterion[f"{split}_weight"]]).to(self.device)
            return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="mean")
        elif criterion["type"] == "ce":
            weight = torch.tensor(criterion[f"{split}_weight"]).to(self.device)
            label_smoothing = float(criterion["label_smoothing"])
            return F.cross_entropy(logits, y, weight=weight, label_smoothing=label_smoothing, ignore_index=-100, reduction="mean")
        else:
            raise ValueError(f"Criterion type `{criterion['type']}` is not supported.")

    def compute_clf_losses(self, logits_list, y_list, split="train"):
        losses = []
        for logits, y, criterion in zip(logits_list, y_list, self.clf_criteria.values()):
            losses.append(self.compute_clf_loss(logits, y, criterion, split))
        return losses

    def compute_seg_loss(self, logits, y, criterion, split="train"):
        pos_weight = torch.tensor([criterion[f"{split}_weight"]]).to(self.device)
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="mean")

    def compute_seg_losses(self, logits_list, y_list, split="train"):
        losses = []
        for logits, y, criterion in zip(logits_list, y_list, self.seg_criteria.values()):
            losses.append(self.compute_seg_loss(logits, y, criterion, split))
        return losses

    def compute_losses(self, logits_list, y_list, split="train"):
        clf_losses = self.compute_clf_losses(logits_list[:-1], y_list[:-1], split)
        seg_losses = self.compute_seg_losses([logits_list[-1]], [y_list[-1]], split)
        return clf_losses + seg_losses

    def compute_any_injury(self, logits_list):
        probs_list = [F.softmax(logits, dim=-1) for logits in logits_list]
        healthy = torch.cat([
            probs_list[0][:, :1],
            probs_list[1][:, :1],
            probs_list[2][:, :1],
            probs_list[3][:, :1],
            probs_list[4][:, :1]
        ], dim=-1)
        any_injury = (1.0 - healthy).max(dim=-1)[0]
        any_injury = torch.log(torch.cat([(1.0 - any_injury)[..., None], any_injury[..., None]], dim=-1))
        logits_list = [any_injury] + logits_list
        return logits_list

    def step(self, batch, split="train", do_mixup=False):
        x = batch["image"]
        y_list = batch["labels"]
        if do_mixup and random.uniform(0, 1) < self.prob_mixup:
            x, y_list, y_list_shuffled, alpha = mixup(x, y_list)
            logits_list = self.model(x)
            logits_list = self.compute_any_injury(logits_list)
            losses_0 = self.compute_losses(logits_list, y_list, split)
            losses_1 = self.compute_losses(logits_list, y_list_shuffled, split)
            losses = [alpha * l0 + (1.0 - alpha) * l1 for l0, l1 in zip(losses_0, losses_1)]
        else:
            logits_list = self.model(x)
            logits_list = self.compute_any_injury(logits_list)
            losses = self.compute_losses(logits_list, y_list, split)
        if split == "valid":
            self.logits_tracker.append(logits_list[:-1])
            self.labels_tracker.append(y_list)
        return losses

    def training_step(self, batch, batch_idx):
        losses = self.step(batch, split="train", do_mixup=True)
        clf_loss, seg_loss = sum(losses[:-1]) / len(losses[:-1]), losses[-1]
        loss = self.alpha * clf_loss + (1.0 - self.alpha) * seg_loss
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_loss_inj", clf_loss, on_step=False, on_epoch=True)
        self.log("train_loss_seg", seg_loss, on_step=False, on_epoch=True)
        self.log("lr", lr, on_step=True, on_epoch=False)
        for i, name in zip(range(6), ["any", "extravasation", "bowel", "liver", "spleen", "kidney"]):
            self.log(f"train_loss_{name}", losses[:-1][i], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        losses = self.step(batch, split="valid", do_mixup=False)
        clf_loss, seg_loss = sum(losses[:-1]) / len(losses[:-1]), losses[-1]
        loss = self.alpha * clf_loss + (1.0 - self.alpha) * seg_loss
        self.log("valid_loss", loss, on_step=False, on_epoch=True)
        self.log("valid_loss_inj", clf_loss, on_step=False, on_epoch=True)
        self.log("valid_loss_seg", seg_loss, on_step=False, on_epoch=True)
        for i, name in zip(range(6), ["any", "extravasation", "bowel", "liver", "spleen", "kidney"]):
            self.log(f"valid_loss_{name}", losses[:-1][i], on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        logits_list = [torch.cat([x[i] for x in self.logits_tracker]).type(torch.half) for i in range(len(self.clf_criteria))]
        y_list = [torch.cat([x[i] for x in self.labels_tracker]) for i in range(len(self.clf_criteria))]
        roc = self.compute_auroc(logits_list, y_list)
        self.log("valid_auroc", roc, on_step=False, on_epoch=True)
        self.logits_tracker.clear()
        self.labels_tracker.clear()

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/romainlhardy/kaggle/rsna-abdominal-trauma/configs/slice_classification/config.yaml", required=True)
    args = parser.parse_args()
    return args


def train(fold, config):
    """Trains a slice-level classification model on a specified fold."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if config["seed"] is not None:
        set_seed(config["seed"])

    df = utils.data_split_slice(os.path.join(utils.DATA_DIR, "data_split_slice.csv"))
    df_train = df[df.fold != fold]
    df_valid = df[df.fold == fold]

    image_size = config["model"]["data"]["image_size"]
    num_channels = config["model"]["data"]["num_channels"]
    train_dataset = SliceClassificationTrainingDataset(
        df_train,
        image_size=image_size,
        num_channels=num_channels,
        split="train"
    )
    valid_dataset = SliceClassificationTrainingDataset(
        df_valid,
        image_size=image_size,
        num_channels=num_channels,
        split="validation"
    )
    
    logging.info(f"[FOLD {fold}]")
    logging.info(f"Training images: {len(train_dataset)}")
    logging.info(f"Validation images: {len(valid_dataset)}")

    def collate_fn(batch):
        return {
            "image": torch.stack([b["image"] for b in batch], dim=0),
            "labels": [
                torch.tensor(np.stack([b["labels"][i] for b in batch])) for i in range(len(INJURY_CATEGORIES) + 1)
            ],
        }
    
    sampler = None
    if config["use_sampler"]:
        classes = df_train.any_injury.unique()
        labels = df_train.any_injury
        class_weights = compute_class_weight("balanced", classes=classes, y=labels)
        sampler = WeightedRandomSampler(
            labels.map(dict([(k, v) for k, v in zip(classes, class_weights)])).values,
            num_samples=len(train_dataset),
            replacement=True
        )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["train_batch_size"], 
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=config["num_workers"],
        collate_fn=collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        shuffle=False, 
        batch_size=config["valid_batch_size"],
        num_workers=config["num_workers"],
        collate_fn=collate_fn
    )

    model_config = config["model"]
    name = f"{model_config['model']['encoder_name'].replace('/', '_')}__image_size_{image_size}__seed_{config['seed']}__fold_{fold}"
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

    model = SliceClassificationModule(model_config)

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