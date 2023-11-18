import pathlib
import rich.progress
import torch
from pillow_lut import load_cube_file
from rich.console import Console
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation, VOCDetection
from torchvision.transforms import ToTensor, Compose, Lambda, CenterCrop, Resize
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from collections import Counter

import aca as style
from util.sortgrid import sort_image_grid

style.use()
lut = load_cube_file(str(style.lut_path))

datasets_root = pathlib.Path("/Users/daniilhayrapetyan/Documents/datasets")
output_dir = pathlib.Path("outputs")
output_dir.mkdir(exist_ok=True)

dataset = VOCSegmentation(str(datasets_root / "PASCAL_SEGMENT"),
                          # download=True,
                          transform=Compose([
                            Lambda(lambda x: x.filter(lut)),
                            # Resize(256),
                            # CenterCrop(256),
                            ToTensor(),
                          ]),
                          target_transform=Compose([
                            # Lambda(lambda x: x.filter(lut)),
                            # Resize(256),
                            # CenterCrop(256),
                            ToTensor(),
                          ]),
                          )

dataset_detection = VOCDetection(str(datasets_root / "PASCAL_SEGMENT"),
                                  # download=True,
                                  transform=Compose([
                                    Lambda(lambda x: x.filter(lut)),
                                    # Resize(256),
                                    # CenterCrop(256),
                                    ToTensor(),
                                  ]),)

console = Console()

skip_images = 1

for filename in dataset.images:
    if filename in dataset_detection.images:
        skip_images -= 1
        if skip_images < 0:
            break

sample = dataset[dataset.images.index(filename)]
sample2 = dataset_detection[dataset_detection.images.index(filename)]

image, mask = sample
mask = torch.logical_not(mask == torch.mode(mask.flatten()).values)

annotation = sample2[1]["annotation"]
# bndboxes = [
#
# ]
image_masked = draw_segmentation_masks((image * 255).byte(), mask,
                                       alpha=0.5,
                                       colors="white")


def bbox_as_4f(obj):
    xmin = obj["bndbox"]["xmin"]
    xmax = obj["bndbox"]["xmax"]
    ymin = obj["bndbox"]["ymin"]
    ymax = obj["bndbox"]["ymax"]
    return int(xmin), int(ymin), int(xmax), int(ymax)


image_bboxed = draw_bounding_boxes((image * 255).byte(),
                                   boxes=torch.tensor(
                                       [bbox_as_4f(obj) for obj in annotation["object"]]
                                   ),
                                   labels=[obj["name"] for obj in annotation["object"]],
                                   width=2,
                                   colors="white")

to_pil_image(image).save(output_dir / "example-classification.png")
to_pil_image(image_masked).save(output_dir / "example-segmentation.png")
to_pil_image(image_bboxed).save(output_dir / "example-detection.png")
