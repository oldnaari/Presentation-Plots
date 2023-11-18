from typing import List
import torch
import itertools
import functools
import numpy as np
import tqdm
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import normalize


def _preprocess_image(tensor: torch.Tensor, mean: torch.tensor):
    return (tensor - mean).numpy().reshape(-1)


def _kernel(image1: np.ndarray, image2: np.ndarray):
    n_channels = image2.shape[0]
    image1_mean_color = np.mean(image1.reshape((n_channels, -1)), axis=-1)
    image2_mean_color = np.mean(image2.reshape((n_channels, -1)), axis=-1)
    return np.dot(image2_mean_color, image1_mean_color)


def sort_image_grid(data: List[List[torch.Tensor]],
                    n_sorting_steps=10,
                    ):
    algorithm = KernelPCA(n_components=2, kernel=_kernel)

    mean = torch.mean(torch.stack(list(itertools.chain(*data)))[:, 0], dim=(0, 2, 3), keepdim=True)[0]
    assert mean.shape[1:] == (1, 1)

    data_flattened_np = [_preprocess_image(tensor, mean) for tensor in itertools.chain(*data)]
    algorithm.fit(data_flattened_np)

    color = functools.partial(torch.mean, dim=(-1, -2))

    for c in tqdm.tqdm([0, 1] * n_sorting_steps):
        for i, row in enumerate(data):
            data[i] = list(sorted(row, key=lambda x: algorithm.transform(_preprocess_image(x, mean)[None])[0, c]))
        data = list(zip(*data))

    return data