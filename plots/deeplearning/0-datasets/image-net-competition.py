import matplotlib.pyplot as plot
import pandas as pd
import pathlib

from styles import aca as style

output_dir = pathlib.Path("outputs")
output_dir.mkdir(exist_ok=True)

style.use()

df = pd.read_csv("ImageNetWinners.csv",
                 skipinitialspace=True)
df.rename(columns=lambda x: x.strip(" \t"), inplace=True)
y = df["TOP 5 ERROR RATE %"]
x = df["YEAR"]
model_names = df["WINNER"]

plot.figure(figsize=(12.8, 4.8))
plot.title("ImageNet Competition Winners")
plot.xlabel("year")
plot.ylabel("top 5 error rate")

plot.plot(x, y, label="winning model score")
plot.scatter(x, y)
plot.plot((x.min(), x.max()), (5.1, 5.1), label="human score")

plot.legend()

offset_x = 0.2
offset_y = 0.2

for i, txt in enumerate(model_names):

    plot.annotate(txt.strip(" \t"), (x[i] + offset_x, y[i] + offset_y))
plot.savefig(output_dir / "ImageNet-Competition.png", format='png', dpi=300)


