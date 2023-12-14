import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from data_prepare.label_split import label_transformer


class StreetImageDataset(Dataset):
    def __init__(self, labels, img_dir):
        self.img_labels = labels
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 2])
        embedding = torch.load(img_path.replace(".jpg", ".pt"))

        label = self.img_labels.iloc[idx, 5]

        return embedding, label


def get_dataset(label_file, test_size, seed, EMB_DIR):
    train_data, val_data, test_data = label_transformer(label_file, test_size, seed)
    train_dataSet = StreetImageDataset(train_data, EMB_DIR)
    val_dataSet = StreetImageDataset(val_data, EMB_DIR)
    test_dataSet = StreetImageDataset(test_data, EMB_DIR)
    return train_dataSet, val_dataSet, test_dataSet



