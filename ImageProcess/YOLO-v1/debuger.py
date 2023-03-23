import pathlib
from collections import defaultdict

import PIL.Image

import torch
import torch.utils.data

import torchvision


def show(sample):
    import matplotlib.pyplot as plt

    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)
    image = F.convert_dtype(image, torch.uint8)
    # annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)
    box = torch.tensor([x['bbox'] for x in target])
    box = torchvision.ops.box_convert(box, 'xywh', 'xyxy')
    # boxes = box.unsqueeze(0)
    boxes = box
    annotated_image = draw_bounding_boxes(image, boxes, colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()

    fig.show()