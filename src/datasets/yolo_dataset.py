import yaml
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image
from typing import Union


class YoloDataset(Dataset):

    def __init__(self, data_filepath: Union[str, Path], transform=None):

        data_filepath = Path(data_filepath)

        # read yaml
        with open(data_filepath) as f:
            self.data_yaml: dict = yaml.safe_load(f)
        self.data_path: Path = data_filepath.parent

        imgs_path = self.data_path / "train" / "images"
        labels_path = self.data_path / "train" / "labels"

        # save image paths and image labels
        self.img_paths = [filepath for filepath in imgs_path.iterdir() if filepath.is_file()]
        self.img_labels = [
            pd.read_csv(filepath, delimiter=' ').to_numpy()
            for filepath in labels_path.iterdir()
            if filepath.is_file()
        ]
        if len(self.img_paths) != len(self.img_labels):
            raise ValueError(f"Number of images {len(self.img_paths)} and labels {len(self.img_labels)} do not match")

        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = read_image(self.img_paths[idx])
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
