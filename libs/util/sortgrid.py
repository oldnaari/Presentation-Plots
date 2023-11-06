from typing import List
import torch
import itertools
import functools


def sort_image_grid(data: List[List[torch.Tensor]]):

    color = functools.partial(torch.mean, dim=(-1, -2))

    key = lambda x: torch.mean(x)

    for _ in range(10):
        for i, row in enumerate(data):
            data[i] = list(sorted(row, key=key))
        data = list(zip(*data))

    return data