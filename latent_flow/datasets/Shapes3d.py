import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

_FACTORS_IN_ORDER = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]
_NUM_VALUES_PER_FACTOR = {"floor_hue": 10, "wall_hue": 10, "object_hue": 10, "scale": 8, "shape": 4, "orientation": 15}


def get_index(factors):
    """Converts factors to indices in range(num_data)
    Args:
      factors: np array shape [6,batch_size].
               factors[i]=factors[i,:] takes integer values in
               range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

    Returns:
      indices: np array shape [batch_size].
    """
    indices = 0
    base = 1
    for factor, name in reversed(list(enumerate(_FACTORS_IN_ORDER))):
        indices += factors[factor] * base
        base *= _NUM_VALUES_PER_FACTOR[name]
    return indices


class Shapes3D(Dataset):
    """3D-Shapes dataset."""

    def __init__(self, root, train=True, transform=None):
        datapath = root
        assert os.path.exists(datapath), "You need to download the data first!"
        data = h5py.File(datapath, "r")
        # file = open(datapath, 'rb')
        # data = pickle.load(file)
        # data = np.load(datapath, mmap_mode='r+',allow_pickle=True)
        self.images = np.array(data["images"][:])  # convert to tensor and place to RAM
        self.images = np.transpose(self.images, [0, 3, 1, 2]) / 255.0  # images in range of [0,1]
        self.labels = np.array(data["labels"][:])  # convert to tensor and place to RAM
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        factors = np.zeros([len(_FACTORS_IN_ORDER), 1], dtype=np.int32)
        for factor, name in enumerate(_FACTORS_IN_ORDER):
            num_choices = _NUM_VALUES_PER_FACTOR[name]
            factors[factor] = np.random.choice(num_choices, 1)
        ind = get_index(factors)
        image = self.images[ind]
        if self.transform:
            image = self.transform(image)
        return image, factors

    def sequence_by_index(self, index, factors):
        if index == 4:
            index = 5
        img_seq = []
        for i in range(0, 8):
            factors[index] = i
            ind = get_index(factors)
            image = self.images[ind]
            if self.transform:
                image = self.transform(image)
            img_seq.append(image)
        img_seq = torch.FloatTensor(img_seq)
        return img_seq
