from datasets.yolo_dataset import YoloDataset
from torch.utils.data import Dataset


def test_yolo_dataset():
    # use actual downloaded data; could update to use simplified unit test data
    data_filepath = "data/nba1022/data.yaml"

    # act
    dataset = YoloDataset(data_filepath)

    # assert
    assert isinstance(dataset, Dataset)
    assert dataset.data_yaml is not None
    assert dataset.data_path is not None
    assert dataset.img_paths is not None
    assert dataset.img_labels is not None
    assert len(dataset) >= 50  # gdrive often limits downloads to 50 which may be the size of a prototype test set
    assert len(dataset.names) == 9

    img, label = dataset[0]
    assert img.shape == (3, 720, 1280)  # channels x height x length
    assert label.shape == (13, 5)  # number of objects x (class label, x, y, w, h)
