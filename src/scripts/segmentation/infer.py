import argparse
import data.utils as utils
import gc
import numpy as np
import pandas as pd
import torch
import yaml

from data.dataset import SegmentationInferenceDataset
from train import SegmentationModule
from torch.utils.data import DataLoader


def load_model(module, config, checkpoint_path):
    model = module(config)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/romainlhardy/kaggle/rsna-abdominal-trauma/configs/segmentation/config.yaml", required=True)
    args = parser.parse_args()
    return args


def infer(config):
    df = utils.data_split_classification("/home/romainlhardy/kaggle/rsna-abdominal-trauma/data/data_split_classification.csv")

    models = config["models"]
    device = config["device"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    predictions = []
    for name, model in models.items():
        config_path = model["config_path"]
        checkpoint_paths = model["checkpoint_paths"]

        print(f"Loading segmentation model `{name}` with configuration: {config_path}")
        with open(config_path, "rb") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        for checkpoint_path in checkpoint_paths:
            print(f"Checkpoint: {checkpoint_path}")
            model = load_model(SegmentationModule, config["model"], checkpoint_path)
            model.to(device)
            model.eval()
            
            image_size = config["model"]["data"]["image_size"]
            
            dataset = SegmentationInferenceDataset(
                df,
                image_size=image_size,
                split="validation"
            )
            dataloader = DataLoader(
                dataset, 
                shuffle=False, 
                batch_size=batch_size,
                num_workers=num_workers
            )

            p = []
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    x = batch["image"].to(device)
                    logits = model.model(x)
                    probs = torch.sigmoid(logits)
                    p.append(probs.cpu().numpy())
            p = np.concatenate(p, axis=0)
            predictions.append(p)

            del model, dataset, dataloader
            torch.cuda.empty_cache()
            gc.collect()

            
    predictions = np.mean(predictions, axis=0)
    results = pd.DataFrame({
        "patient_id": df.patient_id.values,
        "series_id": df.series_id.values,
        "slice_idx": df.slice_idx.values,
        "predicted_organ_label": [str([1.] + list(p)) for p in predictions],
    })
    df = df[[c for c in df.columns if "predicted_organ_label" not in c]]
    df = df.merge(results, on=["patient_id", "series_id", "slice_idx"], how="inner")
    df.to_csv("/home/romainlhardy/kaggle/rsna-abdominal-trauma/data/data_split_classification.csv", index=None)


def main():
    args = parse_args()
    with open(args.config, "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    infer(config)
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()