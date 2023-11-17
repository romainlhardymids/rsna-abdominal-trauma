import functools
import math
import multiprocessing as mp
import nibabel as nib
import numpy as np
import os
import pandas as pd
import pydicom as dicom
import random

from ast import literal_eval
from monai.transforms import Resize
from scipy.ndimage import zoom
from sklearn.model_selection import GroupKFold


# Global variables
DATA_DIR = "/home/romainlhardy/kaggle/rsna-abdominal-trauma/data"
FOLDS = 5
SEGMENTATION_DIM = [128, 224, 224]
SEGMENTATION_LABELS = ["background", "liver", "spleen", "left kidney", "right kidney", "bowel"]
SEGMENTATION_THRESHOLD = 0.9
SAMPLE_GLOBAL_NEGATIVE_PATIENTS = 1.0
SAMPLE_EXTRAVASATION_INJURY_PATIENTS = 1.0
SAMPLE_BOWEL_INJURY_PATIENTS = 1.0
SAMPLE_LIVER_LOW_INJURY_PATIENTS = 1.0
SAMPLE_LIVER_HIGH_INJURY_PATIENTS = 1.0
SAMPLE_SPLEEN_LOW_INJURY_PATIENTS = 1.0
SAMPLE_SPLEEN_HIGH_INJURY_PATIENTS = 1.0
SAMPLE_KIDNEY_LOW_INJURY_PATIENTS = 1.0
SAMPLE_KIDNEY_HIGH_INJURY_PATIENTS = 1.0
SAMPLE_NEGATIVE_SLICES = 0.25


def normalize_min_max(image, eps=1.0e-8):
    """Applies min-max normalization to an input image."""
    return (image - image.min()) / (image.max() - image.min() + eps)


def convert_to_uint8(image):
    """Converts a [0, 1] float array to a [0, 255] uint8 array."""
    return (image * 255.).astype(np.uint8)


def get_spacing(slices):
    """Returns the vertical spacing between two slices."""
    assert len(slices) >= 2
    vertical_spacing = abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    pixel_spacing = list(slices[0].PixelSpacing)
    spacing = list(map(float, [vertical_spacing] + pixel_spacing))
    return np.array(spacing)


def resample_scan(scan, old_spacing, new_spacing, order=3, mode="nearest"):
    """Resamples a 3D scan using a new spacing."""
    r = old_spacing / new_spacing
    new_shape = np.round(scan.shape * r)
    scale_factor = new_shape / scan.shape
    scan = zoom(scan, scale_factor, order=order, mode=mode)
    return scan


def rescale_slice_to_array(slice):
    """Preprocesses a DICOM slice to a NumPy array."""
    image = slice.pixel_array
    if slice.PixelRepresentation == 1:
        bit_shift = slice.BitsAllocated - slice.BitsStored
        dtype = image.dtype
        image = (image << bit_shift).astype(dtype) >> bit_shift
    slope, intercept = slice.get("RescaleSlope", 1.), slice.get("RescaleIntercept", 0.)
    center, width = slice.WindowCenter, slice.WindowWidth
    low, high = center - width / 2, center + width / 2
    image = np.clip(image * slope + intercept, low, high)
    return image


def get_scan_path(patient_id, series_id, split="train"):
    """Returns the path to a directory where a patient's scan is stored."""
    return os.path.join(DATA_DIR, f"{split}_images", str(patient_id), str(series_id))


def get_slice_path(scan_path, slice_idx):
    """Returns the path to a DICOM slice for a given patient."""
    return os.path.join(scan_path, f"{slice_idx}.dcm")


def get_segmentation_path(patient_id, series_id):
    """Returns the path to a segmentation for a given patient."""
    return os.path.join(DATA_DIR, "segmentations", f"{str(series_id)}.nii")


def get_segmentation_numpy_paths(save_dir, patient_id, series_id):
    """Returns the paths to the preprocessed segmentation images and labels."""
    scan_dir = os.path.join(save_dir, "scans", str(patient_id), str(series_id))
    os.makedirs(scan_dir, exist_ok=True)
    image_path = os.path.join(scan_dir, "image.npy")
    mask_path = os.path.join(scan_dir, "mask.npy")
    return image_path, mask_path


def get_sorted_slices(scan_path):
    """Sorts the slice paths for a given scan."""
    return sorted([int(p.split(".")[0]) for p in os.listdir(scan_path)])


def sample_indices(n, m):
    """Samples m equally spaced indices from a list of length n."""
    return np.quantile(list(range(n)), np.linspace(0., 1., m)).round().astype(int)


def load_scan(scan_path):
    """Loads and preprocesses a full patient scan."""
    sorted_slices = get_sorted_slices(scan_path)
    slice_paths = [os.path.join(scan_path, f"{str(i)}.dcm") for i in sorted_slices]
    slices = [dicom.dcmread(p) for p in slice_paths]
    scan = np.stack([rescale_slice_to_array(slice) for slice in slices], axis=0)
    return scan


def load_segmentation(segmentation_path):
    """Loads a segmentation mask."""
    mask = nib.load(segmentation_path).get_fdata()
    mask = np.transpose(mask, (1, 0, 2))
    mask = np.rot90(mask, 1, (1, 2))
    mask = mask[::-1]
    mask = np.transpose(mask, (1, 0, 2)).astype(np.uint8)
    return np.eye(len(SEGMENTATION_LABELS))[mask].transpose((3, 0, 1, 2))


def flip_scan(slices):
    """Returns True if the scan is vertically flipped, and False otherwise."""
    x0, x1 = slices
    if x1.ImagePositionPatient[2] > x0.ImagePositionPatient[2]:
        return True
    return False


def resize_volume(x):
    """Resizes a 3D volume to the dimensions of the segmentator."""
    x = Resize(SEGMENTATION_DIM, mode="trilinear")(x).numpy().astype(np.uint8)
    return x


def segmentation_to_numpy_helper(inputs, save_dir):
    """Helper function for preprocessing the segmentation images and masks."""
    patient_id, series_id = inputs

    # Load scan
    scan_path = get_scan_path(patient_id, series_id, split="train")
    sorted_slices = get_sorted_slices(scan_path)
    indices = sample_indices(len(sorted_slices), SEGMENTATION_DIM[0])
    slices = [dicom.dcmread(get_slice_path(scan_path, sorted_slices[i])) for i in indices]
    image = np.stack([rescale_slice_to_array(slice) for slice in slices], axis=0)
    image = convert_to_uint8(normalize_min_max(image))
    if image.ndim < 4:
        image = image[None, :].repeat(3, 0)
    image = resize_volume(image)

    # Load segmentation
    segmentation_path = get_segmentation_path(patient_id, series_id)
    mask = load_segmentation(segmentation_path)
    mask = mask[:, indices]
    mask = resize_volume(mask)

    # Invert if needed
    flip = flip_scan(slices[:2])
    if flip:
        image = image[:, ::-1]
        mask = mask[:, ::-1]

    # Save to file
    image_path, mask_path = get_segmentation_numpy_paths(save_dir, patient_id, series_id)
    np.save(image_path, image)
    np.save(mask_path, mask)

    return [{
        "patient_id": patient_id,
        "series_id": series_id,
        "image_path": image_path,
        "mask_path": mask_path,
        "flip": flip
    }]


def segmentation_to_numpy():
    """Converts all of the segmentation images and masks to NumPy arrays."""
    save_dir = os.path.join(DATA_DIR, "segmentations_numpy")
    os.makedirs(save_dir, exist_ok=True)
    series = [int(s.split(".")[0]) for s in os.listdir(os.path.join(DATA_DIR, "segmentations"))]
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).merge(pd.read_csv(os.path.join(DATA_DIR, "train_series_meta.csv")), on=["patient_id"], how="inner")
    df = df[df.series_id.isin(series)]
    with mp.Pool(8) as pool:
        meta = sum(pool.map(functools.partial(segmentation_to_numpy_helper, save_dir=save_dir), zip(df.patient_id.values, df.series_id.values)), [])
    meta = pd.DataFrame(meta)
    meta.to_csv(os.path.join(save_dir, "metadata.csv"), index=None)
    return meta


def label_helper(inputs, save_dir):
    """Extracts slice-level segmentation labels for all of the segmentation scans."""
    patient_id, series_id = inputs
    patient_path = os.path.join(DATA_DIR, "train_images", str(patient_id), str(series_id))
    segmentation_path = os.path.join(DATA_DIR, "segmentations", f"{str(series_id)}.nii")
    sorted_slices = get_sorted_slices(patient_path)
    mask = load_segmentation(segmentation_path)
    meta = []
    for i, slice_idx in enumerate(sorted_slices):
        hash = random.getrandbits(128)
        save_path = os.path.join(save_dir, f"{hash}.npy")
        np.save(save_path, mask[i].astype(np.uint8))
        meta.append({
            "patient_id": patient_id,
            "series_id": series_id,
            "slice_idx": slice_idx,
            "position": i,
            "label": [int(j in np.unique(mask[i])) for j in range(len(SEGMENTATION_LABELS))],
            "segmentation_path": save_path
        })
    return meta


def data_split_segmentation(path):
    """Creates a cross-validation split for the segmentation task."""
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(os.path.join(DATA_DIR, "segmentations_numpy", "metadata.csv"))
        kf = GroupKFold(FOLDS)
        for n, (_, valid_idx) in enumerate(kf.split(df, groups=df.patient_id)):
            df.loc[valid_idx, "fold"] = n
        df["fold"] = df["fold"].astype(int)
        df.to_csv(path, index=None)
    return df


def sample_patients(patients, sample_frac):
    """Samples a fraction of patients from a given list."""
    return np.random.choice(patients, math.floor(sample_frac * len(patients)), replace=False)


def sample_negative_slices(df):
    """Samples slices with no visible injuries."""
    negative_patients = sample_patients(df[df.any_injury == 0].patient_id.unique(), SAMPLE_GLOBAL_NEGATIVE_PATIENTS)
    negative_slices = df[df.patient_id.isin(negative_patients)].groupby(["patient_id", "series_id"]).sample(frac=SAMPLE_NEGATIVE_SLICES, replace=False)
    return negative_slices


def sample_extravasation_slices(df):
    """Samples slices with visible extravasation injuries."""
    extravasation_patients = sample_patients(df[df.extravasation_injury == 1].patient_id.unique(), SAMPLE_EXTRAVASATION_INJURY_PATIENTS)
    extravasation_slices = df[df.patient_id.isin(extravasation_patients)]
    extravasation_slices = pd.concat([
        extravasation_slices[extravasation_slices.injury_class == 1],
        extravasation_slices[extravasation_slices.injury_class == 0].groupby(["patient_id", "series_id"]).sample(frac=SAMPLE_NEGATIVE_SLICES, replace=False)
    ], axis=0)
    return extravasation_slices


def sample_bowel_slices(df):
    """Samples slices with visible bowel injuries."""
    bowel_patients = sample_patients(df[df.bowel_injury == 1].patient_id.unique(), SAMPLE_BOWEL_INJURY_PATIENTS)
    bowel_slices = df[df.patient_id.isin(bowel_patients)]
    bowel_slices = pd.concat([
        bowel_slices[bowel_slices.injury_class == 2],
        bowel_slices[bowel_slices.injury_class == 0].groupby(["patient_id", "series_id"]).sample(frac=SAMPLE_NEGATIVE_SLICES, replace=False)
    ], axis=0)
    return bowel_slices


def sample_liver_low_slices(df):
    """Samples slices with visible low-grade liver injuries."""
    liver_low_patients = sample_patients(df[df.liver_low == 1].patient_id.unique(), SAMPLE_LIVER_LOW_INJURY_PATIENTS)
    liver_low_slices = df[df.patient_id.isin(liver_low_patients)]
    liver_low_slices = pd.concat([
        liver_low_slices[liver_low_slices.liver_low == 1],
        liver_low_slices[liver_low_slices.liver_low == 0].groupby(["patient_id", "series_id"]).sample(frac=SAMPLE_NEGATIVE_SLICES, replace=False)
    ], axis=0)
    return liver_low_slices


def sample_liver_high_slices(df):
    """Samples slices with visible high-grade liver injuries."""
    liver_high_patients = sample_patients(df[df.liver_high == 1].patient_id.unique(), SAMPLE_LIVER_HIGH_INJURY_PATIENTS)
    liver_high_slices = df[df.patient_id.isin(liver_high_patients)]
    liver_high_slices = pd.concat([
        liver_high_slices[liver_high_slices.liver_high == 1],
        liver_high_slices[liver_high_slices.liver_high == 0].groupby(["patient_id", "series_id"]).sample(frac=SAMPLE_NEGATIVE_SLICES, replace=False)
    ], axis=0)
    return liver_high_slices


def sample_spleen_low_slices(df):
    """Samples slices with visible low-grade spleen injuries."""
    spleen_low_patients = sample_patients(df[df.spleen_low == 1].patient_id.unique(), SAMPLE_SPLEEN_LOW_INJURY_PATIENTS)
    spleen_low_slices = df[df.patient_id.isin(spleen_low_patients)]
    spleen_low_slices = pd.concat([
        spleen_low_slices[spleen_low_slices.spleen_low == 1],
        spleen_low_slices[spleen_low_slices.spleen_low == 0].groupby(["patient_id", "series_id"]).sample(frac=SAMPLE_NEGATIVE_SLICES, replace=False)
    ], axis=0)
    return spleen_low_slices


def sample_spleen_high_slices(df):
    """Samples slices with visible high-grade spleen injuries."""
    spleen_high_patients = sample_patients(df[df.spleen_high == 1].patient_id.unique(), SAMPLE_SPLEEN_HIGH_INJURY_PATIENTS)
    spleen_high_slices = df[df.patient_id.isin(spleen_high_patients)]
    spleen_high_slices = pd.concat([
        spleen_high_slices[spleen_high_slices.spleen_high == 1],
        spleen_high_slices[spleen_high_slices.spleen_high == 0].groupby(["patient_id", "series_id"]).sample(frac=SAMPLE_NEGATIVE_SLICES, replace=False)
    ], axis=0)
    return spleen_high_slices


def sample_kidney_low_slices(df):
    """Samples slices with visible low-grade kidney injuries."""
    kidney_low_patients = sample_patients(df[df.kidney_low == 1].patient_id.unique(), SAMPLE_KIDNEY_LOW_INJURY_PATIENTS)
    kidney_low_slices = df[df.patient_id.isin(kidney_low_patients)]
    kidney_low_slices = pd.concat([
        kidney_low_slices[kidney_low_slices.kidney_low == 1],
        kidney_low_slices[kidney_low_slices.kidney_low == 0].groupby(["patient_id", "series_id"]).sample(frac=SAMPLE_NEGATIVE_SLICES, replace=False)
    ], axis=0)
    return kidney_low_slices


def sample_kidney_high_slices(df):
    """Samples slices with visible high-grade kidney injuries."""
    kidney_high_patients = sample_patients(df[df.kidney_high == 1].patient_id.unique(), SAMPLE_KIDNEY_HIGH_INJURY_PATIENTS)
    kidney_high_slices = df[df.patient_id.isin(kidney_high_patients)]
    kidney_high_slices = pd.concat([
        kidney_high_slices[kidney_high_slices.kidney_high == 1],
        kidney_high_slices[kidney_high_slices.kidney_high == 0].groupby(["patient_id", "series_id"]).sample(frac=SAMPLE_NEGATIVE_SLICES, replace=False)
    ], axis=0)
    return kidney_high_slices


def sample_slices(df):
    """Samples slices for the slice-level classification task."""
    slices = pd.concat([
        sample_negative_slices(df),
        sample_extravasation_slices(df),
        sample_bowel_slices(df),
        sample_liver_low_slices(df),
        sample_liver_high_slices(df),
        sample_spleen_low_slices(df),
        sample_spleen_high_slices(df),
        sample_kidney_low_slices(df),
        sample_kidney_high_slices(df)
    ], axis=0).drop_duplicates(subset=["patient_id", "series_id", "slice_idx"]).reset_index(drop=True)
    return slices


def parse_organ_label(organ_label):
    """Parses an organ label list."""
    parsed = literal_eval(organ_label.replace("nan", "None"))
    return [x if x is not None else 0.0 for x in parsed]


def split_organ_labels(organ_labels):
    """Splits organ labels and converts them to a NumPy array."""
    split_labels = []
    for l in organ_labels:
        split_labels.append(parse_organ_label(l))
    return np.array(split_labels)


def correct_slice_labels(df):
    """Converts scan-level labels to slice level labels using the predicted visibility of each organ."""
    df["liver_low"] = df["liver_low"] * (df["seg_pred_liver"] > SEGMENTATION_THRESHOLD)
    df["liver_high"] = df["liver_high"] * (df["seg_pred_liver"] > SEGMENTATION_THRESHOLD)
    df["spleen_low"] = df["spleen_low"] * (df["seg_pred_spleen"] > SEGMENTATION_THRESHOLD)
    df["spleen_high"] = df["spleen_high"] * (df["seg_pred_spleen"] > SEGMENTATION_THRESHOLD)
    df["kidney_low"] = df["kidney_low"] * ((df["seg_pred_left_kidney"] > SEGMENTATION_THRESHOLD) | (df["seg_pred_right_kidney"] > SEGMENTATION_THRESHOLD))
    df["kidney_high"] = df["kidney_high"] * ((df["seg_pred_left_kidney"] > SEGMENTATION_THRESHOLD) | (df["seg_pred_right_kidney"] > SEGMENTATION_THRESHOLD))
    return df


def data_split_classification(path):
    """Creates a cross-validation split for the classification tasks."""
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).merge(pd.read_csv(os.path.join(DATA_DIR, "train_series_meta.csv")), on=["patient_id"], how="inner")
        meta = []
        for patient_id, series_id in zip(df.patient_id.values, df.series_id.values):
            patient_path = os.path.join(DATA_DIR, "train_images", str(patient_id), str(series_id))
            sorted_slices = sorted_slices(patient_path)
            for i, slice_idx in enumerate(sorted_slices):
                meta.append({
                    "patient_id": patient_id,
                    "series_id": series_id,
                    "slice_idx": slice_idx,
                    "position": i
                })
        df = df.merge(pd.DataFrame(meta), on=["patient_id", "series_id"], how="inner")
        df.to_csv(path, index=None)
    return df


def data_split_slice(path):
    """Creates a cross-validation split for the slice-level classification task."""
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = data_split_classification(os.path.join(DATA_DIR, "data_split_classification.csv"))
        split_labels = split_organ_labels(df["predicted_organ_label"].values)
        for i, label in enumerate(SEGMENTATION_LABELS[1:]):
            df[f"seg_pred_{label.replace(' ', '_')}"] = split_labels[:, i + 1]
        df = correct_slice_labels(df)
        image_labels = pd.read_csv(os.path.join(DATA_DIR, "image_level_labels.csv")).rename(columns={"instance_number": "slice_idx"})
        mask_pred = pd.read_csv("/home/romainlhardy/kaggle/rsna-abdominal-trauma/data/mask_predictions/metadata.csv")
        df = df.merge(image_labels, on=["patient_id", "series_id", "slice_idx"], how="left")
        df = df.merge(mask_pred, on=["patient_id", "series_id"], how="inner")
        df["injury_name"] = df["injury_name"].fillna("Other")
        df["injury_class"] = df["injury_name"].map({
            "Other": 0,
            "Active_Extravasation": 1,
            "Bowel": 2
        })
        df = sample_slices(df)
        kf = GroupKFold(FOLDS)
        for n, (_, valid_idx) in enumerate(kf.split(df, groups=df.patient_id)):
            df.loc[valid_idx, "fold"] = n
        df["fold"] = df["fold"].astype(int)
        df.to_csv(path, index=None)
    return df


def data_split_scan(path, models=[]):
    """Creates a cross-validation split for the scan-level classification task."""
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = data_split_classification(os.path.join(DATA_DIR, "data_split_classification.csv"))
        slice_features = None
        for model in models:
            sf = pd.read_csv(os.path.join(DATA_DIR, "slice_features", model, "metadata.csv"))[["patient_id", "series_id", "slice_idx", "features_path"]]
            sf = sf.rename(columns={"features_path": f"{model}_features_path"})
            if slice_features is None:
                slice_features = sf
            else:
                slice_features = slice_features.merge(sf, on=["patient_id", "series_id", "slice_idx"], how="inner")
        df = df.merge(slice_features, on=["patient_id", "series_id", "slice_idx"], how="inner")
        kf = GroupKFold(FOLDS)
        for n, (_, valid_idx) in enumerate(kf.split(df, groups=df.patient_id)):
            df.loc[valid_idx, "fold"] = n
        df["fold"] = df["fold"].astype(int)
        df.to_csv(path, index=None)
    return df
