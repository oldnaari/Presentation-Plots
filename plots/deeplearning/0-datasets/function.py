from matplotlib.pyplot import plot
import aca as style
import numpy as np
import skimage.draw
from PIL import ImageColor
import matplotlib as mpl
import pathlib

style.use()
output_dir = pathlib.Path("outputs")


functions = {
    "gaussian": lambda x: -0.8 * np.exp(- x * x * 4),
    "sigmoid": lambda x: 0.8 * np.exp(-x) / (np.exp(-x) + 1),
}

for function_name, function in functions.items():
    size = 512
    radius = size * 3 / 256

    canvas = np.zeros([size, size, 4], dtype=np.uint8)

    xs = np.linspace(-1.0, 1.0, 1500)
    ys = -0.8 * np.exp(- xs * xs * 4)

    for x, y in zip(xs, ys):
        j = int(size * (x + 1) / 2)
        i = int(size * (y + 1) / 2)
        idx_i, idx_j = skimage.draw.disk((i, j), radius, shape=(size, size))
        color = next(iter(mpl.rcParams["axes.prop_cycle"]))["color"]
        color = ImageColor.getrgb(color) + (255, )
        skimage.draw.set_color(canvas, (idx_i, idx_j), color, alpha=0.1)

    skimage.io.imsave(output_dir / f"{function_name}.png", canvas)
