import os
import numpy as np
from torch.utils.data import Dataset
import glob
import PIL.Image

factor_sizes = [5, 6, 6, 6, 6, 6, 6]
factor_names = [
    "lighting_intensity",
    "lighting_x-dir",
    "lighting_y-dir",
    "lighting_z-dir",
    "camera_x-pos",
    "camera_y-pos",
    "camera_z-pos",
]


class Falor3D(Dataset):
    """Falor3D dataset."""

    def __init__(self, root, train=True, transform=None):
        self.image_files = sorted(glob.glob(os.path.join(root, "*.png")))
        self.transform = transform
        self.factor_bases = np.array(np.prod(factor_sizes) / np.cumprod(factor_sizes), np.int32)
        self.index = 0

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # index = np.random.randint(7)
        index = self.index
        factor_size = factor_sizes[index]
        index_arr = self.factor_bases[index] * np.arange(factor_size) + (idx % self.factor_bases[0])
        image_names = [self.image_files[t] for t in index_arr]
        images = []
        for image_file in image_names:
            assert isinstance(image_file, str), image_file
            img = PIL.Image.open(image_file)
            img = np.array(img, dtype=np.float32) / 255.0
            if self.transform:
                img = self.transform(img)
            images.append(img)
        # images = np.array(images)
        return images, index
