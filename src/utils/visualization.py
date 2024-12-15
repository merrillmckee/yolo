import matplotlib.pyplot as plt

from datasets.yolo_dataset import YoloDataset
from matplotlib.pyplot import Axes
from matplotlib.patches import Rectangle


def draw_box(
        ax: Axes,
        xy: tuple[float, float],
        wh: tuple[float, float],
        imgsz: tuple[int, int],
        name: str = None,
):
    """

    Parameters
    ----------
    ax:
        matplotlib axis
    xy:
        normalized 0-to-1 xy coordinates of upper-left of box
    wh:
        normalized 0-to-1 wh size of box
    imgsz:
        image size as (rows, cols)
    name:
        string name of this class

    Returns
    -------
    Draws a rectangle to the given axis
    """

    rows, cols = imgsz
    ulxy = [xy[0] * cols, xy[1] * rows]

    rect = Rectangle(
        xy=ulxy,
        width=wh[0] * cols,
        height=wh[1] * rows,
        linewidth=1,
        edgecolor='cyan',
        facecolor='none',
    )
    ax.add_patch(rect)
    ax.text(ulxy[0], ulxy[1], name, color='magenta')


def visualize_dataset(dataset: YoloDataset):

    for img, labels in dataset:
        fig, ax = plt.subplots()
        img = img.permute(1, 2, 0)
        plt.imshow(img)

        for label in labels:
            label_index, x, y, w, h = label
            label_index = int(label_index)

            ulx = (x - w / 2.0)
            uly = (y - h / 2.0)
            name = dataset.names[label_index]
            draw_box(ax, [ulx, uly], [w, h], img.shape[:2], name)

        plt.axis('equal')
        plt.show()

        # cleanup
        plt.close()


if __name__ == "__main__":
    data_filepath = "../../data/nba1022/data.yaml"
    dataset = YoloDataset(data_filepath)

    # visualize
    visualize_dataset(dataset)
