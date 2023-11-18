import pathlib

import PIL.Image
import rich.progress
import torch
from pillow_lut import load_cube_file
from rich.console import Console
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10, MNIST, VOCDetection
from torchvision.transforms import ToTensor, Compose, Lambda, CenterCrop, Resize
from torchvision.transforms.functional import to_pil_image

import aca as style
import numpy as np
from util.sortgrid import sort_image_grid

style.use()
lut = load_cube_file(str(style.lut_path))

datasets_root = pathlib.Path("/Users/daniilhayrapetyan/Documents/datasets")

def additional_filter(image):
    return image
    image = np.array(image).astype(np.float32)
    image_des = np.mean(image, axis=-1, keepdims=True)
    image_out = np.minimum(image_des, image)
    print(image.shape)
    return PIL.Image.fromarray(image_out.astype(np.uint8))


datasets = {
    "80MIL": CIFAR10(str(datasets_root / "CIFAR10"),
                     download=True,
                     transform=Compose([
                         Lambda(lambda x: additional_filter(x)),
                         Lambda(lambda x: x.filter(lut)),
                         ToTensor()
                     ])),
    "CIFAR10": CIFAR10(str(datasets_root / "CIFAR10"),
                        download=True,
                        transform=Compose([
                            Lambda(lambda x: additional_filter(x)),
                            Lambda(lambda x: x.filter(lut)),
                            ToTensor()
                        ])),
    "CIFAR100": CIFAR100(str(datasets_root / "CIFAR100"),
                         download=True,
                         transform=Compose([
                             Lambda(lambda x: additional_filter(x)),
                             Lambda(lambda x: x.filter(lut)),
                             ToTensor()
                         ])),
    "MNIST": MNIST(str(datasets_root / "MNIST"),
                   download=True,
                   transform=Compose([
                       ToTensor()
                   ])),
    "PASCAL": VOCDetection(str(datasets_root / "PASCAL"),
                           download=True,
                           transform=Compose([
                               Lambda(lambda x: additional_filter(x)),
                               Lambda(lambda x: x.filter(lut)),
                               Resize(256),
                               CenterCrop(256),
                               ToTensor(),
                           ]))
}

sizes = {
    "80MIL": (16, 36),
    "CIFAR10": (8, 16),
    "CIFAR100": (8, 16),
    "MNIST": (10, 20),
    "PASCAL": (3, 6)
}


console = Console()
for key in datasets:
    dataloader = DataLoader(datasets[key])

    n_rows, n_cols = sizes[key]

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
