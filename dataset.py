import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader

import torchvision.datasets as dset
import torchvision.transforms as transforms



class Dataset():
    def __init__(self, root, model_config, train="train"):
        super().__init__()
        self.dataroot = root
        self.train = train
        self.image_size = model_config["image_size"]
        self.scaled = True # imagefolder normalization already scale images.

    def get_dataset(self):
        dataset = dset.ImageFolder(root=self.dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(self.image_size),
                                transforms.CenterCrop(self.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        return dataset

    def get_data_scaler(self, x):
        """Data normalizer. Assume data are always in [0, 1]."""
        if not self.scaled:
            # Rescale to [-1, 1]
            return lambda x: x * 2. - 1.
        else:
            return lambda x: x


    def get_data_inverse_scaler(self, x):
        """Inverse data normalizer."""
        if not self.scaled:
            # Rescale [-1, 1] to [0, 1]
            return lambda x: (x + 1.) / 2.
        else:
            return lambda x: x

