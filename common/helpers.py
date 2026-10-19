"""Shared helpers: data loading, a tiny Report logger, and figure saving."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: saves PNGs without a display
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "figure.figsize": (7, 4.5),
})


def load(name: str) -> pd.DataFrame:
    """Load a CSV from the shared data/ folder."""
    return pd.read_csv(DATA / name)


class Report:
    """Prints text AND stores it so it ends up in outputs/results_python.txt."""

    def __init__(self, script_file: str):
        self.out = Path(script_file).resolve().parents[1] / "outputs"
        self.out.mkdir(exist_ok=True)
        self.lines: list[str] = []

    def p(self, *args) -> None:
        text = " ".join(str(a) for a in args)
        print(text)
        self.lines.append(text)

    def h(self, title: str) -> None:
        self.p("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)

    def save_fig(self, fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(self.out / name, dpi=130)
        plt.close(fig)

    def finish(self) -> None:
        (self.out / "results_python.txt").write_text("\n".join(self.lines), encoding="utf-8")
