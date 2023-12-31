import argparse
import data.utils as utils
import gc
import numpy as np
import os
import pandas as pd
import torch
import yaml

from data.dataset import ScanClassificationTrainingDataset, INJURY_CATEGORIES
from train import ScanClassificationModule
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
from torch.nn.functional import softmax

from tqdm.auto import tqdm


def load_model(module, config, checkpoint_path):
    """Loads a scan-level classification model checkpoint."""
    model = module(config)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    return model


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/romainlhardy/kaggle/rsna-abdominal-trauma/configs/scan_classification/config.yaml", required=True)
    args = parser.parse_args()
    return args


def collate_fn(batch):
    """Collate function for batching individual instances."""
    return {
        "features": torch.stack([b["features"] for b in batch], dim=0),
        "labels": [torch.tensor([b["labels"][i] for b in batch]) for i in range(len(INJURY_CATEGORIES) + 1)],
        "mask": torch.stack([b["mask"] for b in batch], dim=0)
    }


def labels_to_weights(labels, map):
    """Maps injury labels to the competition weights."""
    mapped = labels.copy()
    for k, v in map.items():
        mapped[labels == k] = v
    return mapped


def get_sample_weights(labels, category):
    """Returns the sample weights for a given injury category."""
    if category == "any":
        map = {0: 1, 1: 6}
    elif category == "extravasation":
        map = {0: 1, 1: 6}
    elif category == "bowel":
        map = {0: 1, 1: 2}
    elif category == "liver":
        map = {0: 1, 1: 2, 2: 4}
    elif category == "spleen":
        map = {0: 1, 1: 2, 2: 4}
    elif category == "kidney":
        map = {0: 1, 1: 2, 2: 4}
    else:
        raise ValueError(f"Category `{category}` is not supported.")
    return labels_to_weights(labels, map)


def one_hot_labels(labels):
    """One-hot encodes a vector of labels."""
    n = len(np.unique(labels))
    return np.eye(n)[labels]


def trauma_metric(predictions, labels):
    """Computes the competition metric."""
    losses = []
    for p, l, category in zip(predictions, labels, INJURY_CATEGORIES):
        l_ = one_hot_labels(l)
        print(f"Category `{category}` distribution: {np.round(np.mean(l_, axis=0), 4)}")
        sample_weights = get_sample_weights(l, category)
        losses.append(log_loss(l_, p, sample_weight=sample_weights))
    return losses, np.mean(losses)


def infer(config):
    """Performs inference on held-out data."""
    df = utils.data_split_scan("/home/romainlhardy/kaggle/rsna-abdominal-trauma/data/data_split_scan.csv")

    models = config["models"]
    device = config["device"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    metric_tracker = []
    for name, model in models.items():
        config_path = model["config_path"]
        checkpoint_paths = model["checkpoint_paths"]
        folds = model["folds"]

        print(f"Loading slice classification model `{name}` with configuration: {config_path}")
        with open(config_path, "rb") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        for (checkpoint_path, fold) in zip(checkpoint_paths, folds):
            print(f"Checkpoint: {checkpoint_path}")
            model = load_model(ScanClassificationModule, config["model"], checkpoint_path)
            model.to(device)
            model.eval()
            
            df_ = df[df.fold == fold]
            data_params = config["model"]["data"]
            dataset = ScanClassificationTrainingDataset(
                df_,
                time_dim=data_params["time_dim"],
                p_shuffle=0.0,
                split="validation"
            )
            dataloader = DataLoader(
                dataset, 
                shuffle=False, 
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=collate_fn
            )

            predictions = [[] for _ in range(len(INJURY_CATEGORIES))]
            labels = [[] for _ in range(len(INJURY_CATEGORIES))]

            debug = [[] for _ in range(len(INJURY_CATEGORIES))]

            with torch.no_grad():
                for i, batch in tqdm(enumerate(dataloader)):
                    x = batch["features"].to(device)
                    mask = batch["mask"].to(device)
                    y_list = [y.to(device) for y in batch["labels"]]

                    logits_list = model.model(x, mask)
                    probs_list = [softmax(logits, dim=1) for logits in logits_list[:-1]]
                    healthy = torch.cat([
                        probs_list[0][:, :1],
                        probs_list[1][:, :1],
                        probs_list[2][:, :1],
                        probs_list[3][:, :1],
                        probs_list[4][:, :1]
                    ], dim=1)
                    any_injury = (1.0 - healthy).max(dim=1)[0]
                    any_injury = torch.cat([(1.0 - any_injury)[..., None], any_injury[..., None]], dim=1)

                    probs_list = [any_injury] + probs_list
                    for j, (probs, y) in enumerate(zip(probs_list, y_list[:-1])):
                        predictions[j].append(probs.cpu().numpy())
                        labels[j].append(y.cpu().numpy())

            predictions = [np.concatenate(p) for p in predictions]
            labels = [np.concatenate(l) for l in labels]

            _, metric = trauma_metric(predictions, labels)
            metric_tracker.append(metric)

            del model, dataset, dataloader
            torch.cuda.empty_cache()
            gc.collect()

    print(f"Fold-averaged trauma metric: {np.mean(metric_tracker):.04f}")


def main():
    args = parse_args()
    with open(args.config, "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    infer(config)
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()