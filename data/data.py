from torchvision import transforms
from dataloader import PathImageFolder
import numpy as np
from torch.utils.data import SubsetRandomSampler
import torch


class Data:
    def __init__(self, train_path, test_path, batch_size, valid_size):
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.train_transform = None
        self.test_transform = None
        self.valid_size = valid_size

    def get_trainset(self):
        # Defning proper transform
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Loading data
        train_data = PathImageFolder(self.train_path, transform=self.train_transform)

        # Getting data size
        data_size = len(train_data)

        # Creating index list
        index_list = list(range(data_size))

        # Shuffling index_list
        np.random.shuffle(index_list)

        # Defining splitter
        splitter = int(np.floor(self.valid_size*data_size))

        # Splitting index_list
        valid_list, train_list = index_list[:splitter], index_list[splitter:]

        # Creating samples
        train_sampler, valid_sampler = SubsetRandomSampler(train_list), SubsetRandomSampler(valid_list)

        # Creating data loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, sampler=valid_sampler)

        return train_loader, valid_loader



    def get_testset(self, shuffle):
        # Defining a transform method
        self.test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # Loading test images
        test_data = PathImageFolder(self.test_path, transform=self.test_transform)

        # Getting data loader
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=shuffle)

        return test_loader


def get_class_name(path, level):
    return path.split('\\')[level]


