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


@dataclass
class Report:
    timesteps: torch.Tensor
    trajectory: torch.Tensor
    plotter: "AnimationPlotter"

    @torch.no_grad()
    def save_as_gif(self, pathname: pathlib.Path, console):
        snapshots = []

        level_prev = None
        for i, t in enumerate(rich.progress.track(self.timesteps, console=console)):
            plot.figure(figsize=(8, 8))
            plot.xlim(*self.plotter.xlims)
            plot.ylim(*self.plotter.ylims)
            plot.plot(self.plotter.xs,
                      self.plotter.f(self.plotter.xs))
            plot.plot(self.trajectory[:i],
                      self.plotter.f(self.trajectory[:i]),
                      lw=3,
                      c="white")
            level = self.plotter.energy(self.trajectory[i]).item()
            plot.gca().add_patch(
                Rectangle((-far_x, level), 2 * far_x, level + far_x,
                          edgecolor='white',
                          facecolor='C2',
                          alpha=0.4))

            if level_prev is not None:
                plot.gca().add_patch(
                    Rectangle((-far_x, level), 2 * far_x, level_prev - level,
                              # edgecolor='white',
                              facecolor='white',
                              alpha=1))

            plot.plot((-far_x, far_x), (level, level), lw=2, c="black")

            plot.scatter(self.trajectory[i], self.plotter.f(self.trajectory[i]),
                         c="white")
            img_buf = io.BytesIO()
            plot.savefig(img_buf, format='png', dpi=300)
            im = Image.open(img_buf)
            snapshots.append(to_tensor(im))
            img_buf.close()

            level_prev = level

        snapshots = torch.stack(snapshots)

        frames = []
        fps = 25
        frame_timesteps = list(torch.arange(0, self.plotter.duration, 1.0 / fps).data)

        canvas = snapshots[0]
        for t, snapshot in zip(self.timesteps,
                               rich.progress.track(snapshots, console=console)):
            c1 = canvas[:3]
            a1 = canvas[[3]]
            f = 0.4
            c2 = snapshot[:3]
            a2 = snapshot[[3]]
            a = a1 + a2 * f - a1 * a2 * f
            ac = -c1 * a1 * a2 * f + a1 * c1 + a2 * c2 * f
            c = torch.where(a1 == 0, c2, c1 + a2 * f * (c2 - c1) / a)
            canvas = torch.cat([c, a])
            if t > frame_timesteps[0]:
                frames.append(np.array(to_pil_image(canvas)))
                frame_timesteps.pop(0)
                if len(frame_timesteps) == 0:
                    break

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
            self.energy = self.f

    def render_animation(self):
        timesteps = torch.linspace(0.0, self.duration, int(self.duration * 50))

        values = []

        for _ in timesteps:
            values.append(self.points.detach().clone())
            self.optimizer.zero_grad()
            loss = self.f(self.points)
            loss.backward()
            self.optimizer.step()

        return Report(timesteps, torch.stack(values), plotter=self)


console = rich.console.Console()

starting_point = torch.nn.Parameter(torch.Tensor([-1.5]))
animator = AnimationPlotter(
    f=lambda x: x ** 2,
    xs=torch.linspace(-3.0, 3.0, 1000),
    points=starting_point,
    duration=4.0,
    optimizer=torch.optim.SGD([starting_point], lr=0.002)
)
report = animator.render_animation()
report.save_as_gif(save_root / "gradient-descent-1d.gif", console=console)

055023232