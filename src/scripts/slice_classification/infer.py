import argparse
import data.utils as utils
import gc
import numpy as np
import os
import pandas as pd
import random
import torch
import yaml

from data.dataset import SliceClassificationInferenceDataset
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from train import SliceClassificationModule

from tqdm.auto import tqdm


def load_model(module, config, checkpoint_path):
    """Loads a slice-level classification model checkpoint."""
    model = module(config)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    return model


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/romainlhardy/kaggle/rsna-abdominal-trauma/configs/slice_classification/config.yaml", required=True)
    args = parser.parse_args()
    return args


def infer(config):
    """Performs inference on held-out data."""
    df = utils.data_split_slice("/home/romainlhardy/kaggle/rsna-abdominal-trauma/data/data_split_slice.csv")
    mask_meta = pd.read_csv(config["mask_metadata"]).set_index(["patient_id", "series_id"])

    models = config["models"]
    features_dir = config["features_dir"]
    device = config["device"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    features_meta = []
    predictions_list = [[] for _ in range(5)]
    for name, model in models.items():
        config_path = model["config_path"]
        checkpoint_paths = model["checkpoint_paths"]
        folds = model["folds"]

        print(f"Loading slice classification model `{name}` with configuration: {config_path}")
        with open(config_path, "rb") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        for checkpoint_path, fold in zip(checkpoint_paths, folds):
            print(f"Checkpoint: {checkpoint_path}")
            model = load_model(SliceClassificationModule, config["model"], checkpoint_path)
            model.to(device)
            model.eval()

            df_fold = df[df.fold == fold]
            keys = list(df_fold.groupby(["patient_id", "series_id"]).groups.keys())
            meta = []
            for patient_id, series_id in keys:
                patient_path = os.path.join(utils.DATA_DIR, "train_images", str(patient_id), str(series_id))
                sorted_slices = sorted([int(p.split(".")[0]) for p in os.listdir(patient_path)])
                r = mask_meta.loc[patient_id, series_id]
                mask_dir, mask_bounds = r.mask_dir, r.mask_bounds
                for i, slice_idx in enumerate(sorted_slices):
                    meta.append({
                        "patient_id": patient_id,
                        "series_id": series_id,
                        "slice_idx": slice_idx,
                        "position": i,
                        "mask_dir": mask_dir,
                        "mask_bounds": mask_bounds
                    })
            df_fold = pd.DataFrame(meta)
            
            image_size = config["model"]["data"]["image_size"]
            num_channels = config["model"]["data"]["num_channels"]
            dataset = SliceClassificationInferenceDataset(
                df_fold,
                image_size=image_size,
                num_channels=num_channels,
                split="validation"
            )
            dataloader = DataLoader(
                dataset, 
                shuffle=False, 
                batch_size=batch_size,
                num_workers=num_workers
            )

            with torch.no_grad():
                for i, batch in tqdm(enumerate(dataloader)):
                    x = batch["image"].to(device)
                    features = model.model.forward_features(x)
                    logits_list = model.model.head(features)[:-1]
                    for j, f in enumerate(features):
                        row = df_fold.iloc[i * batch_size + j]
                        hash = random.getrandbits(128)
                        save_path = os.path.join(features_dir, "features", f"{hash}.npy")
                        np.save(save_path, f.cpu().numpy())
                        features_meta.append({
                            "patient_id": row.patient_id,
                            "series_id": row.series_id,
                            "slice_idx": row.slice_idx,
                            "features_path": save_path,
                            "model_checkpoint_path": checkpoint_path,
                            "fold": fold
                        })
                    for j, logits in enumerate(logits_list):
                        probs = softmax(logits, dim=-1)
                        predictions_list[j].append(probs.cpu().numpy())

            del model, dataset, dataloader
            torch.cuda.empty_cache()
            gc.collect()

    df_meta = pd.DataFrame(features_meta)
    df_meta.to_csv(os.path.join(features_dir, "metadata.csv"), index=None)


def main():
    args = parse_args()
    with open(args.config, "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    infer(config)
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()