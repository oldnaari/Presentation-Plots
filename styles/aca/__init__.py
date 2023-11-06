import matplotlib.pyplot as plot
import matplotlib.font_manager as font_manager
import pathlib


def use():
    plot.style.use("aca.aca")


lut_path: pathlib.Path = (pathlib.Path(__file__).parent / "aca.CUBE").absolute()
