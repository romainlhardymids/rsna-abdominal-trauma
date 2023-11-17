import argparse
import data.utils as utils
import gc
import numpy as np
import os
import pandas as pd
import torch
import yaml

from data.dataset import SegmentationInferenceDataset
from train import SegmentationModule
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def load_model(module, config, checkpoint_path):
    """Loads a segmentation model checkpoint."""
    model = module(config)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    return model


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/romainlhardy/kaggle/rsna-abdominal-trauma/configs/segmentation/config.yaml", required=True)
    args = parser.parse_args()
    return args


def get_bounds(n, s):
    """Returns image bounds for non-zero segmentation predictions."""
    nx, ny, nz = n
    sx, sy, sz = s
    m = len(nx)
    if m == 0:
        x0, x1 = 0, 0
        y0, y1 = 0, 0
        z0, z1 = 0, 0
    else:
        x0, x1 = int(0.8 * nx[int(0.001 * m)]), int(1.2 * nx[int(0.999 * m)])
        y0, y1 = int(0.8 * ny[int(0.001 * m)]), int(1.2 * ny[int(0.999 * m)])
        z0, z1 = int(nz[int(0.001 * m)]), int(nz[int(0.999 * m)])

    xx0, xx1 = int(x0 * sx / utils.SEGMENTATION_DIM[1]), int(x1 * sx / utils.SEGMENTATION_DIM[1])
    yy0, yy1 = int(y0 * sy / utils.SEGMENTATION_DIM[2]), int(y1 * sy / utils.SEGMENTATION_DIM[2])
    zz0, zz1 = int(z0 * sz / utils.SEGMENTATION_DIM[0]), int(z1 * sz / utils.SEGMENTATION_DIM[0])

    x0, x1 = max(x0, 0), min(x1, utils.SEGMENTATION_DIM[1])
    y0, y1 = max(y0, 0), min(y1, utils.SEGMENTATION_DIM[2])
    z0, z1 = max(z0, 0), min(z1, utils.SEGMENTATION_DIM[0])

    xx0, xx1 = max(xx0, 0), min(xx1, sx.item())
    yy0, yy1 = max(yy0, 0), min(yy1, sy.item())
    zz0, zz1 = max(zz0, 0), min(zz1, sz.item())
    return [(x0, x1), (y0, y1), (z0, z1)], [(xx0, xx1), (yy0, yy1), (zz0, zz1)]


def segmentation_infer_batch(batch, models, thresholds=[0.3, 0.1]):
    """Performs inference on a batch of images."""
    logits = sum([model.model(batch["image"].to(model.device)) for model in models]) / len(models)
    probs = torch.sigmoid(logits)
    bounds = []
    masks = []
    for p, sx, sy, sz in zip(probs, batch["sx"], batch["sy"], batch["sz"]):
        patient_bounds = []
        for organ_id in range(1, len(utils.SEGMENTATION_LABELS)):
            p0 = p[organ_id] > thresholds[0]
            p1 = p[organ_id] > thresholds[1]
            nz, nx, ny = torch.nonzero(p0, as_tuple=True)
            n = len(nx)
            if n == 0:
                nz, nx, ny = torch.nonzero(p1, as_tuple=True)
            nx, ny, nz = torch.sort(nx)[0], torch.sort(ny)[0], torch.sort(nz)[0]
            b, bb = get_bounds((nx, ny, nz), (sx, sy, sz))
            patient_bounds.append([b, bb])
        bounds.append(patient_bounds)
        masks.append(p.cpu().numpy().astype(np.float16))
    return bounds, masks


def infer(config):
    """Performs inference on held-out data."""
    df = pd.read_csv("/home/romainlhardy/kaggle/rsna-abdominal-trauma/data/train_series_meta.csv")
    df = df[["patient_id", "series_id"]]

    backbones = config["models"]
    device = config["device"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    save_dir = config["save_dir"]

    models = []
    for name, arch in backbones.items():
        config_path = arch["config_path"]
        checkpoint_paths = arch["checkpoint_paths"]
        print(f"Loading segmentation model `{name}` with configuration: {config_path}")
        with open(config_path, "rb") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for checkpoint_path in checkpoint_paths:
            print(f"Checkpoint: {checkpoint_path}")
            model = load_model(SegmentationModule, config["model"], checkpoint_path)
            model.to(device)
            model.eval()
            models.append(model)

    dataset = SegmentationInferenceDataset(df, split="validation")

    dataloader = DataLoader(
        dataset, 
        shuffle=False, 
        batch_size=batch_size,
        num_workers=num_workers
    )

    meta = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            bounds, masks = segmentation_infer_batch(batch, models)
            for j, (b, m) in enumerate(zip(bounds, masks)):
                row = df.iloc[i * batch_size + j]
                patient_id, series_id = row.patient_id, row.series_id
                mask_dir = os.path.join(save_dir, "masks", str(patient_id), str(series_id))
                os.makedirs(mask_dir, exist_ok=True)
                for k in range(utils.SEGMENTATION_DIM[1]):
                    mask_path = os.path.join(mask_dir, f"mask_{k}.npy")
                    np.save(mask_path, m[:, :, k])
                meta.append({
                    "patient_id": row.patient_id,
                    "series_id": row.series_id,
                    "mask_bounds": b,
                    "mask_dir": mask_dir
                })

    torch.cuda.empty_cache()
    gc.collect()

    df_meta = pd.DataFrame(meta)
    df_meta.to_csv(os.path.join(save_dir, "metadata.csv"), index=None)


def main():
    args = parse_args()
    with open(args.config, "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    infer(config)
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()