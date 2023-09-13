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
    model = module(config)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/romainlhardy/kaggle/rsna-abdominal-trauma/configs/slice_classification/config.yaml", required=True)
    args = parser.parse_args()
    return args


def infer(config):
    df = utils.data_split_slice("/home/romainlhardy/kaggle/rsna-abdominal-trauma/data/data_split_slice.csv")

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
                for i, slice_idx in enumerate(sorted_slices):
                    meta.append({
                        "patient_id": patient_id,
                        "series_id": series_id,
                        "slice_idx": slice_idx,
                        "position": i
                    })
            df_fold = pd.DataFrame(meta)
            
            image_size = config["model"]["data"]["image_size"]
            
            dataset = SliceClassificationInferenceDataset(
                df_fold,
                image_size=image_size,
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

    predictions_list = [np.concatenate(predictions, axis=0) for predictions in predictions_list]
    results = pd.DataFrame({
        "patient_id": df_meta.patient_id.values,
        "series_id": df_meta.series_id.values,
        "slice_idx": df_meta.slice_idx.values,
        "fold": df_meta.fold.values,
        # "pred_any_injury": [str(list(p)) for p in predictions_list[0]],
        "pred_extravasation_injury": [str(list(p)) for p in predictions_list[0]],
        "pred_bowel_injury": [str(list(p)) for p in predictions_list[1]],
        "pred_liver_injury": [str(list(p)) for p in predictions_list[2]],
        "pred_spleen_injury": [str(list(p)) for p in predictions_list[3]],
        "pred_kidney_injury": [str(list(p)) for p in predictions_list[4]],
    })
    results.to_csv("/home/romainlhardy/kaggle/rsna-abdominal-trauma/data/pred_slice_classification.csv", index=None)


def main():
    args = parse_args()
    with open(args.config, "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    infer(config)
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()