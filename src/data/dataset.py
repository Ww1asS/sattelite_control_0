import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import get_train_transforms, get_val_transforms


def read_image_unicode(path, flags):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


class LEVIRCDDataset(Dataset):
    def __init__(self, root, split="train", transform=None, img_size=256):
        assert split in ("train", "val", "test")

        self.root = root
        self.split = split
        self.dir_A = os.path.join(root, split, "A")
        self.dir_B = os.path.join(root, split, "B")
        self.dir_L = os.path.join(root, split, "label")

        for p in (self.dir_A, self.dir_B, self.dir_L):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Directory not found: {p}")

        self.filenames = sorted(
            f for f in os.listdir(self.dir_A)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        )

        if len(self.filenames) == 0:
            raise RuntimeError(f"No images in {self.dir_A}")

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(img_size)
        else:
            self.transform = get_val_transforms(img_size)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        img_A = read_image_unicode(os.path.join(self.dir_A, fname), cv2.IMREAD_COLOR)
        img_B = read_image_unicode(os.path.join(self.dir_B, fname), cv2.IMREAD_COLOR)
        mask = read_image_unicode(os.path.join(self.dir_L, fname), cv2.IMREAD_GRAYSCALE)

        if img_A is None:
            raise FileNotFoundError(f"Cannot read: {os.path.join(self.dir_A, fname)}")
        if img_B is None:
            raise FileNotFoundError(f"Cannot read: {os.path.join(self.dir_B, fname)}")
        if mask is None:
            raise FileNotFoundError(f"Cannot read: {os.path.join(self.dir_L, fname)}")

        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
        mask = (mask > 127).astype(np.float32)

        augmented = self.transform(image=img_A, image2=img_B, mask=mask)

        img_A_t = augmented["image"]       # [3, H, W]
        img_B_t = augmented["image2"]      # [3, H, W]
        mask_t = augmented["mask"].unsqueeze(0).float()  # [1, H, W]

        # Склеиваем два изображения по каналам: [6, H, W]
        # Модель сама разделит это на два [3, H, W] тензора
        x = torch.cat([img_A_t, img_B_t], dim=0)

        return x, mask_t