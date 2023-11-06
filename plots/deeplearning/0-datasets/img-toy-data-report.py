import pathlib

from torchvision.datasets import CIFAR100
import rich.progress
import torch
from pillow_lut import load_cube_file
from rich.console import Console
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10, MNIST
from torchvision.transforms import ToTensor, Compose, Lambda
from torchvision.transforms.functional import to_pil_image

import aca as style
from util.sortgrid import sort_image_grid

style.use()
lut = load_cube_file(str(style.lut_path))

datasets_root = pathlib.Path("/Users/daniilhayrapetyan/Documents/datasets")

datasets = {
    "CIFAR10": CIFAR10(str(datasets_root / "CIFAR10"),
                        download=True,
                        transform=Compose([
                            Lambda(lambda x: x.filter(lut)),
                            ToTensor()
                        ])),
    "CIFAR100": CIFAR100(str(datasets_root / "CIFAR100"),
                         download=True,
                         transform=Compose([
                             Lambda(lambda x: x.filter(lut)),
                             ToTensor()
                         ])),
    "MNIST": MNIST(str(datasets_root / "MNIST"),
                   download=True,
                   transform=Compose([
                       ToTensor()
                   ]))
}


console = Console()
for key in datasets:
    dataloader = DataLoader(datasets[key])

    n_rows = 5
    n_cols = 10

    images = []
    for i, (img, label) in zip(
            rich.progress.track(list(range(n_rows * n_cols)), console=console),
            dataloader):
        if i % n_cols == 0:
            images.append(list())
        images[-1].append(img)

    console.print(f"DATASET : {key}")
    console.print(f"\t SIZE : {len(datasets[key])}")
    console.print(f"\t SHAPE: {images[0][0].shape}")

    images = sort_image_grid(images)
    result = torch.cat([torch.cat(batch, dim=-1) for batch in images], dim=-2)
    to_pil_image(result[0]).save(f"outputs/{key}.png")
