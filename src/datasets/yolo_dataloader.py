import torch

from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class YoloDataLoader(DataLoader):

    def __init__(self, dataset: Dataset, **kwargs):
        super().__init__(dataset, **kwargs)

        # custom collate function
        self.collate_fn = self.collate_batch

    @staticmethod
    def collate_batch(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        """
        Collates a batch of (image, labels) pairings

        Parameters
        ----------
        batch:
            list of tuples/pairs of (image, labels) tensors

        Returns
        -------
        batch_images:
            tensor of size (batch_size, channels, height, width)
        batch_labels:
            tensor of size number_of_objects_in_batch x (image_index_into_batch, label_index, x, y, w, h)
        """

        batch_images_list, batch_labels_list = zip(*batch)

        # collate images
        batch_images = torch.stack(batch_images_list)

        # collate labels while inserting a new column for batch image index
        batch_labels_list_new_column = []
        for i, image_labels in enumerate(batch_labels_list):
            batch_index = torch.full((len(image_labels), 1), fill_value=i)
            image_labels_new_column = torch.hstack((batch_index, image_labels))
            batch_labels_list_new_column.append(image_labels_new_column)
        batch_labels = torch.cat(batch_labels_list_new_column)

        return batch_images, batch_labels
