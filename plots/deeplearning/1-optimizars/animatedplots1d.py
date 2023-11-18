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
final_render = False


@dataclass
class Report:
    timesteps: torch.Tensor
    trajectory: torch.Tensor
    momentums: torch.Tensor
    plotter: "AnimationPlotter"

    @torch.no_grad()
    def save_as_gif(self, pathname: pathlib.Path, console):
        frames = []

        level_prev = None
        fps = 25
        frame_interval = 1.0 / fps

        last_t = -10

        for i, t in enumerate(rich.progress.track(self.timesteps, console=console)):
            if last_t + frame_interval > t:
                continue

            last_t = t
            plot.figure(figsize=(8, 8))
            plot.xlim(*self.plotter.xlims)
            plot.ylim(*self.plotter.ylims)
            plot.plot(self.plotter.xs,
                      self.plotter.f(self.plotter.xs))
            plot.plot(self.trajectory[:i + 1],
                      self.plotter.f(self.trajectory[:i + 1]),
                      lw=3,
                      c="black",
                      zorder=4)
            level = self.plotter.energy(self.trajectory[i], self.momentums[i]).item()
            plot.gca().add_patch(
                Rectangle((-far_x, level), 2 * far_x, level + far_x,
                          edgecolor='white',
                          facecolor='C2',
                          alpha=0.4))

            if level_prev is not None:
                level_prev = level + (level_prev - level) / 3
                height = level - level_prev
                plot.gca().add_patch(
                    Rectangle((-far_x, level_prev), 2 * far_x, height,
                              edgecolor='white',
                              facecolor='white',
                              alpha=1,
                              zorder=3))

            plot.plot((-far_x, far_x), (level, level), lw=2, c="black", zorder=3)

            plot.scatter(self.trajectory[i], self.plotter.f(self.trajectory[i]),
                         c="black",
                         zorder=5)
            img_buf = io.BytesIO()
            plot.savefig(img_buf, format='png', dpi=300 if final_render else 50)
            im = Image.open(img_buf)
            frames.append(np.array(im))
            img_buf.close()

            level_prev = level

        clip = ImageSequenceClip(
            sequence=frames,
            fps=fps,
        )
        clip.write_gif(pathname)
        return clip


@dataclass
class AnimationPlotter:
    f: Callable = None
    energy: Callable = None
    xs: torch.Tensor = None

    points: torch.nn.Parameter = None
    optimizer: torch.optim.Optimizer = None
    xlims: Tuple[float, float] = (-2.25, 2.25)
    ylims: Tuple[float, float] = (-0.25, 4.25)

    duration: float = 1.5

    def __post_init__(self):
        if self.energy is None:
            self.energy = lambda x, m: self.f(x)

    def render_animation(self):
        timesteps = torch.linspace(0.0, self.duration, int(self.duration * 100))

        values = []
        momentums = []

        for _ in timesteps:
            state = self.optimizer.state[self.points]
            if "momentum_buffer" in state and isinstance(state["momentum_buffer"], torch.Tensor):
                momentums.append(state["momentum_buffer"].detach().clone())
            else:
                momentums.append(torch.zeros_like(self.points))

            values.append(self.points.detach().clone())
            self.optimizer.zero_grad()
            loss = self.f(self.points)
            loss.backward()
            self.optimizer.step()

        return Report(timesteps, torch.stack(values), torch.stack(momentums), plotter=self)


console = rich.console.Console()

# region Gradient Descent
starting_point = torch.nn.Parameter(torch.Tensor([-1.5]))
animator = AnimationPlotter(
    f=lambda x: x ** 2,
    xs=torch.linspace(-3.0, 3.0, 1000),
    points=starting_point,
    duration=4.0,
    optimizer=torch.optim.SGD([starting_point], lr=0.01)
)
report = animator.render_animation()
report.save_as_gif(save_root / "gradient-descent-1d.gif", console=console)
# endregion

# region Momentum
starting_point = torch.nn.Parameter(torch.Tensor([-1.5]))
optimizer = torch.optim.SGD([starting_point], lr=0.01, momentum=0.98)
f = lambda x: x ** 2
lr = optimizer.param_groups[0]["lr"]
momentum = optimizer.param_groups[0]["momentum"]
energy = lambda x, m: (f(x) + 0.5 * m ** 2 * lr / momentum)
animator = AnimationPlotter(
    f=f,
    energy=energy,
    xs=torch.linspace(-3.0, 3.0, 1000),
    points=starting_point,
    duration=4.0,
    optimizer=optimizer,
)
report = animator.render_animation()
report.save_as_gif(save_root / "momentum-1d.gif", console=console)
# endregion

# region Signed
starting_point = torch.nn.Parameter(torch.Tensor([-1.5]))
animator = AnimationPlotter(
    f=lambda x: -torch.exp(- 4 * x ** 2),
    xs=torch.linspace(-3.0, 3.0, 1000),
    xlims=(-2.0, 2.0),
    ylims=(-2.0, 2.0),
    points=starting_point,
    duration=4.0,
    optimizer=torch.optim.Rprop([starting_point], lr=0.001, etas=(0.9999, 1.0001))
)
report = animator.render_animation()
report.save_as_gif(save_root / "signed-1d.gif", console=console)
# endregion


# region RMSProp
starting_point = torch.nn.Parameter(torch.Tensor([-1.5]))
animator = AnimationPlotter(
    f=lambda x: -torch.exp(- 4 * x ** 2),
    xs=torch.linspace(-3.0, 3.0, 1000),
    xlims=(-2.0, 2.0),
    ylims=(-2.0, 2.0),
    points=starting_point,
    duration=4.0,
    optimizer=torch.optim.RMSprop([starting_point], lr=0.001)
)
report = animator.render_animation()
report.save_as_gif(save_root / "rmsprop-1d.gif", console=console)
# endregion
