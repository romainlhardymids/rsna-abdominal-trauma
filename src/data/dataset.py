import albumentations as A
import cv2
import data.utils as utils
import monai.transforms as transforms
import numpy as np
import os
import pydicom as dicom
import random
import torch

from albumentations.pytorch import ToTensorV2
from ast import literal_eval
from scipy import ndimage
from torch.utils.data import Dataset


INJURY_CATEGORIES = [
    "any",
    "extravasation",
    "bowel",
    "liver",
    "spleen",
    "kidney"
]


def get_extravasation_label(row):
    """Returns the extravasation injury label for a given slice."""
    return int(int(row.injury_class) == 1)


def get_bowel_label(row):
    """Returns the bowel injury label of the patient."""
    seg_pred_bowel = row.seg_pred_bowel
    if seg_pred_bowel < utils.SEGMENTATION_THRESHOLD:
        return 0
    elif row.injury_class != 2:
        return 1
    else:
        return 2


def get_liver_label(row):
    """Returns the liver injury label for a given slice."""
    seg_pred_liver = row.seg_pred_liver
    if seg_pred_liver < utils.SEGMENTATION_THRESHOLD:
        return 0
    elif row.liver_healthy == 1:
        return 1
    elif row.liver_low == 1:
        return 2
    elif row.liver_high == 1:
        return 3


def get_spleen_label(row):
    """Returns the spleen injury label for a given slice."""
    seg_pred_spleen = row.seg_pred_spleen
    if seg_pred_spleen < utils.SEGMENTATION_THRESHOLD:
        return 0
    elif row.spleen_healthy == 1:
        return 1
    elif row.spleen_low == 1:
        return 2
    elif row.spleen_high == 1:
        return 3


def get_kidney_label(row):
    """Returns the kidney injury label for a given slice."""
    seg_pred_left_kidney = row.seg_pred_left_kidney
    seg_pred_right_kidney = row.seg_pred_right_kidney
    if seg_pred_left_kidney < utils.SEGMENTATION_THRESHOLD and seg_pred_right_kidney < utils.SEGMENTATION_THRESHOLD:
        return 0
    elif row.kidney_healthy == 1:
        return 1
    elif row.kidney_low == 1:
        return 2
    elif row.kidney_high == 1:
        return 3


def get_organ_segmentation_label(row):
    """Returns the organ segmentation label for a given slice."""
    return np.array(utils.parse_organ_label(row.predicted_organ_label)[1:])


def get_slice_classification_labels(row):
    """Returns all slice-level labels."""
    labels = [
        get_extravasation_label(row),
        get_bowel_label(row),
        get_liver_label(row),
        get_spleen_label(row),
        get_kidney_label(row),
        get_organ_segmentation_label(row)
    ]
    global_label = 1 if (labels[0] == 1 or any([l > 1 for l in labels[1:-1]])) else 0
    labels = [global_label] + labels
    return labels


def get_scan_classification_labels(rows):
    """Returns all scan-level labels."""
    first_row = rows.iloc[0]
    labels = [
        first_row.any_injury,
        first_row.extravasation_injury,
        first_row.bowel_injury,
        0 if first_row.liver_healthy else (1 if first_row.liver_low else 2),
        0 if first_row.spleen_healthy else (1 if first_row.spleen_low else 2),
        0 if first_row.kidney_healthy else (1 if first_row.kidney_low else 2)
    ]
    segmentation_labels = np.stack([utils.parse_organ_label(row.predicted_organ_label)[1:] for _, row in rows.iterrows()])
    labels += [segmentation_labels]
    return labels


def crop_image(image):
    """Center crop an input image around the largest non-zero blob."""
    thresh = (image > 0.0).astype(np.uint8)
    blobs, _ = ndimage.label(thresh)
    counts = np.bincount(blobs.ravel().astype(np.int32))
    counts[0] = 0
    mask = (blobs == counts.argmax()).astype(np.uint8)
    mask = cv2.dilate(mask, kernel=np.ones((3, 3)))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(sorted(contours, key=cv2.contourArea, reverse=True)[0])
    cropped = (image * mask)[y : y + h, x : x + w]
    return cropped


class SegmentationTrainingDataset(Dataset):
    """Training dataset for the segmentation task."""
    def __init__(self, df, split="train"):
        self.df = df
        self.split = split
        self.transform = get_segmentation_transform(self.split)

    def __len__(self):
        return len(self.df)
    
    def apply_transform(self, image, mask):
        tf = self.transform({"image": image, "mask": mask})
        image, mask = tf["image"], tf["mask"]
        image = image / 127.5 - 1.0
        return image, mask

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image, mask = np.load(row.image_path), np.load(row.mask_path)
        image, mask = self.apply_transform(image, mask)
        sample = {
            "image": torch.tensor(image).float(),
            "mask": torch.tensor(mask).float()
        }
        return sample
    

class SegmentationInferenceDataset(Dataset):
    """Inference dataset for the segmentation task."""
    def __init__(self, df, split="train"):
        self.df = df
        self.split = split
        self.transform = get_segmentation_transform(self.split)

    def __len__(self):
        return len(self.df)
    
    def load_scan(self, patient_id, series_id):
        scan_path = utils.get_scan_path(patient_id, series_id, split="train")
        sorted_slices = utils.get_sorted_slices(scan_path)
        indices = utils.sample_indices(len(sorted_slices), utils.SEGMENTATION_DIM[0])
        slices = [dicom.dcmread(utils.get_slice_path(scan_path, sorted_slices[i])) for i in indices]
        image = np.stack([utils.rescale_slice_to_array(slice) for slice in slices], axis=0)
        image = utils.convert_to_uint8(utils.normalize_min_max(image))
        sz = len(sorted_slices)
        sx, sy = image.shape[1], image.shape[2]
        if image.ndim < 4:
            image = image[None, :].repeat(3, 0)
        image = utils.resize_volume(image)
        flip = utils.flip_scan(slices[:2])
        if flip:
            image = image[:, ::-1]
        return image, (sz, sx, sy)
    
    def apply_transform(self, image):
        tf = self.transform({"image": image})
        image = tf["image"]
        image = image / 127.5 - 1.0
        return image

    def __getitem__(self, i):
        row = self.df.iloc[i]
        patient_id, series_id = row.patient_id, row.series_id
        image, shape = self.load_scan(patient_id, series_id)
        image = self.apply_transform(image)
        sample = {
            "image": torch.tensor(image).float(),
            "sx": shape[1],
            "sy": shape[2],
            "sz": shape[0]
        }
        return sample


class SliceClassificationTrainingDataset(Dataset):
    """Training dataset for the slice-level classification task."""
    def __init__(
        self, 
        df, 
        image_size=384, 
        num_channels=5, 
        split="train"
    ):
        self.df = df
        self.image_size = image_size
        self.num_channels = num_channels
        self.split = split
        self.transform = get_slice_classification_transform(num_channels, self.split)

    def __len__(self):
        return len(self.df)

    def resize(self, image):
        image = A.Compose([
            A.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_LINEAR, always_apply=True)
        ])(image=image)["image"]
        return image
    
    def apply_transform(self, image, mask):
        tf = self.transform(image=image, mask=mask)
        image, mask = tf["image"], tf["mask"]
        return image, mask
    
    def clean_bounds(self, bounds):
        x0, x1 = min([b[0][0] for b in bounds]), max([b[0][1] for b in bounds])
        y0, y1 = min([b[1][0] for b in bounds]), max([b[1][1] for b in bounds])
        return (x0, x1), (y0, y1)
    
    def crop(self, x, bounds, i):
        (x0, x1), (y0, y1) = self.clean_bounds([b[i] for b in bounds])
        return x[x0 : x1, y0 : y1]

    def __getitem__(self, i):
        row = self.df.iloc[i]
        patient_id, series_id, slice_idx = list(map(lambda x: str(int(x)), [row.patient_id, row.series_id, row.slice_idx]))
        mask_dir = row.mask_dir
        mask_bounds = literal_eval(row.mask_bounds)
        scan_path = utils.get_scan_path(patient_id, series_id, split="train")
        sorted_slices = utils.get_sorted_slices(scan_path)
        flip = utils.flip_scan([dicom.dcmread(utils.get_slice_path(scan_path, sorted_slices[i])) for i in range(2)])
        channels = []
        for j in range(-self.num_channels // 2 + 1, self.num_channels // 2 + 1):
            try:
                s = int(slice_idx) + j
                slice_path = os.path.join(utils.DATA_DIR, "train_images", patient_id, series_id, f"{str(s)}.dcm")
                c = dicom.dcmread(slice_path)
            except:
                s = int(slice_idx)
                slice_path = os.path.join(utils.DATA_DIR, "train_images", patient_id, series_id, f"{str(s)}.dcm")
                c = dicom.dcmread(slice_path)
            c = utils.normalize_min_max(utils.rescale_slice_to_array(c))
            c = self.crop(c, mask_bounds, i=1)
            c = self.resize(c)
            channels.append(c)
        image = np.stack(channels, axis=0).astype(np.float32)
        z = int(sorted_slices.index(int(slice_idx)) * utils.SEGMENTATION_DIM[0] / len(sorted_slices))
        if flip:
            z = utils.SEGMENTATION_DIM[0] - z
        z = min(utils.SEGMENTATION_DIM[0] - 1, z)
        mask = np.load(os.path.join(mask_dir, f"mask_{z}.npy"))[1:].max(axis=0).astype(np.float32)
        mask = self.crop(mask, mask_bounds, i=0)
        mask = self.resize(mask)
        image = np.transpose(image, (1, 2, 0))
        image, mask = self.apply_transform(image, mask)
        mask = 2.0 * mask - 1.0
        image = torch.cat([image, mask[None, :]], axis=0)
        labels = get_slice_classification_labels(row)
        sample = {
            "image": image,
            "labels": labels
        }
        return sample
    

class SliceClassificationInferenceDataset(Dataset):
    """Inference dataset for the slice-level classification task."""
    def __init__(
        self, 
        df, 
        image_size=384, 
        num_channels=5, 
        split="train"
    ):
        self.df = df
        self.image_size = image_size
        self.num_channels = num_channels
        self.split = split
        self.transform = get_slice_classification_transform(num_channels, self.split)

    def __len__(self):
        return len(self.df)

    def resize(self, image):
        image = A.Compose([
            A.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_LINEAR, always_apply=True)
        ])(image=image)["image"]
        return image
    
    def apply_transform(self, image, mask):
        tf = self.transform(image=image, mask=mask)
        image, mask = tf["image"], tf["mask"]
        return image, mask
    
    def clean_bounds(self, bounds):
        x0, x1 = min([b[0][0] for b in bounds]), max([b[0][1] for b in bounds])
        y0, y1 = min([b[1][0] for b in bounds]), max([b[1][1] for b in bounds])
        return (x0, x1), (y0, y1)
    
    def crop(self, x, bounds, i):
        (x0, x1), (y0, y1) = self.clean_bounds([b[i] for b in bounds])
        return x[x0 : x1, y0 : y1]

    def __getitem__(self, i):
        row = self.df.iloc[i]
        patient_id, series_id, slice_idx = list(map(lambda x: str(int(x)), [row.patient_id, row.series_id, row.slice_idx]))
        mask_dir = row.mask_dir
        mask_bounds = literal_eval(row.mask_bounds)
        scan_path = utils.get_scan_path(patient_id, series_id, split="train")
        sorted_slices = utils.get_sorted_slices(scan_path)
        flip = utils.flip_scan([dicom.dcmread(utils.get_slice_path(scan_path, sorted_slices[i])) for i in range(2)])
        channels = []
        for j in range(-self.num_channels // 2 + 1, self.num_channels // 2 + 1):
            try:
                s = int(slice_idx) + j
                slice_path = os.path.join(utils.DATA_DIR, "train_images", patient_id, series_id, f"{str(s)}.dcm")
                c = dicom.dcmread(slice_path)
            except:
                s = int(slice_idx)
                slice_path = os.path.join(utils.DATA_DIR, "train_images", patient_id, series_id, f"{str(s)}.dcm")
                c = dicom.dcmread(slice_path)
            c = utils.normalize_min_max(utils.rescale_slice_to_array(c))
            c = self.crop(c, mask_bounds, i=1)
            c = self.resize(c)
            channels.append(c)
        image = np.stack(channels, axis=0).astype(np.float32)
        z = int(sorted_slices.index(int(slice_idx)) * utils.SEGMENTATION_DIM[0] / len(sorted_slices))
        if flip:
            z = utils.SEGMENTATION_DIM[0] - z
        z = min(utils.SEGMENTATION_DIM[0] - 1, z)
        mask = np.load(os.path.join(mask_dir, f"mask_{z}.npy"))[1:].max(axis=0).astype(np.float32)
        mask = self.crop(mask, mask_bounds, i=0)
        mask = self.resize(mask)
        image = np.transpose(image, (1, 2, 0))
        image, mask = self.apply_transform(image, mask)
        mask = 2.0 * mask - 1.0
        image = torch.cat([image, mask[None, :]], axis=0)
        sample = {
            "image": image
        }
        return sample
    

class ScanClassificationTrainingDataset(Dataset):
    """Training dataset for the scan-level classification task."""
    def __init__(
        self, 
        df, 
        time_dim, 
        split="train"
    ):
        self.df = df
        self.scans = list(df.groupby(["patient_id", "series_id"]).groups.keys())
        self.feature_cols = [c for c in self.df if "features_path" in c]
        self.time_dim = time_dim
        self.split = split
        self.transform = get_scan_classification_transform(self.split)

    def __len__(self):
        return len(self.scans)
    
    def load_features(self, feature_paths):
        return np.stack([np.load(path) for path in feature_paths], axis=0)

    def apply_transform(self, features):
        tf = self.transform(image=features)
        return tf["image"]

    def __getitem__(self, i):
        patient_id, series_id = self.scans[i]
        rows = self.df[(self.df.patient_id == patient_id) & (self.df.series_id == series_id)].sort_values(by="slice_idx", ascending=True)
        features = []
        for feature_col in self.feature_cols:
            features.append(self.load_features(rows[feature_col].values))
        features = np.concatenate(features, axis=1)
        labels = get_scan_classification_labels(rows)
        t = features.shape[0]
        mask = np.ones((self.time_dim,))
        features = self.apply_transform(features)
        if t > self.time_dim:
            features = cv2.resize(features, (features.shape[-1], self.time_dim), interpolation=cv2.INTER_LINEAR)
            labels[-1] = cv2.resize(labels[-1], (labels[-1].shape[-1], self.time_dim), interpolation=cv2.INTER_LINEAR)
        else:
            pad_dim = ((0, self.time_dim - t), (0, 0))
            features = np.pad(features, pad_dim, mode="constant")
            labels[-1] = np.pad(labels[-1], pad_dim, mode="constant")
            mask[t:] = 0.0
        sample = {
            "features": torch.tensor(features, dtype=torch.float32),
            "labels": labels,
            "mask": torch.tensor(mask, dtype=torch.float16)
        }
        return sample


def get_segmentation_transform(split="train"):
    """Returns image transformations for the 3D segmentation model."""
    if split == "train":
        return transforms.Compose([
            transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
            transforms.RandAffined(keys=["image", "mask"], translate_range=[int(x * y) for x, y in zip(utils.SEGMENTATION_DIM, [0.3, 0.3, 0.3])], padding_mode="zeros", prob=0.7),
            transforms.RandGridDistortiond(keys=["image", "mask"], prob=0.5, distort_limit=[-0.01, 0.01], mode="nearest"),    
        ])

    else:
        return transforms.Compose([])


def get_slice_classification_transform(num_channels, split="train"):
    """Returns image transformations for the slice classification model."""
    if split == "train":
        return A.Compose([
            A.RandomBrightness(limit=0.05, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=25, border_mode=4, p=0.8),
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=3)
            ], p=0.5),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4),
                A.GridDistortion(num_steps=5, distort_limit=0.03)
            ], p=0.5),
            A.OneOf([
                A.Cutout(max_h_size=32, max_w_size=32, num_holes=3),
                A.PixelDropout(dropout_prob=0.1)
            ], p=0.5),
            A.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels, max_pixel_value=1.0, always_apply=True),
            ToTensorV2(always_apply=True)
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels, max_pixel_value=1.0, always_apply=True),
            ToTensorV2(always_apply=True)
        ])


def get_scan_classification_transform(split="train"):
    """Returns image transformations for the scan classification model."""
    if split == "train":
        return A.Compose([
            A.OneOf([
                A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, fill_value=0.0),
                A.PixelDropout(dropout_prob=0.1, drop_value=0.0),
                A.GaussNoise(var_limit=1.0, mean=0.0)
            ], p=0.8)
        ])
    else:
        return A.Compose([])