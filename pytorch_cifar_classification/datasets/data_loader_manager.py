from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch


class DataLoaderManager:
    def __init__(self, batch_size, valid_size, number_of_workers):
        # Set data transform process
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        # Load train_data and test_data
        train_data = datasets.CIFAR10('../../data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10('../../data', train=False, download=True, transform=transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
                                                        num_workers=number_of_workers)
        self.valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler,
                                                        num_workers=number_of_workers)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=number_of_workers)

    def get_train_loader(self):
        return self.train_loader

    def get_valid_loader(self):
        return self.valid_loader

    def get_test_loader(self):
        return self.test_loader
