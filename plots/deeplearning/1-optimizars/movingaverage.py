import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from dataclasses import dataclass
from typing import List, Callable, Tuple
import rich.console
import rich.progress
import pathlib
import aca as style
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plot
import io
from PIL import Image
import numpy as np
from matplotlib.patches import Rectangle



style.use()
save_root = pathlib.Path("outputs")
save_root.mkdir(exist_ok=True)
far_x = 10

