from datasets.yolo_dataloader import YoloDataLoader
from datasets.yolo_dataset import YoloDataset
from torch.utils.data import Dataset, DataLoader


def test_yolo_dataloader():
    # use actual downloaded data; could update to use simplified unit test data
    data_filepath = "data/nba1022/data.yaml"
    batch_size = 4

    # act
    dataset = YoloDataset(data_filepath)
    data_loader = YoloDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # get expected number of objects in 1st batch
    n_objects = 0
    for i in range(batch_size):
        _, labels = dataset[i]
        n_objects += len(labels)

    # assert
    assert isinstance(dataset, Dataset)
    assert isinstance(data_loader, DataLoader)

    for images, labels in data_loader:
        assert images.shape == (batch_size, 3, 720, 1280)  # channels x height x length
        assert labels.shape == (n_objects, 6)  # number of objects in 1st batch x (image index, class label, x, y, w, h)
        break
