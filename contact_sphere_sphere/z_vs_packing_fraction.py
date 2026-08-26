import math
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from matplotlib.lines import Line2D


BASE_FOLDER = Path(__file__).resolve().parent
PACKING_FOLDER = BASE_FOLDER
CONTACT_FILE = BASE_FOLDER / "sphere_contact_mean_std_sph_p_rod.txt"
PACKING_FILE = PACKING_FOLDER / "packing_fraction_mean_std_n_spheres_per_rod.txt"
OUTPUT_FILE = BASE_FOLDER / "z_vs_packing_fraction_review.pdf"

COLORS = {"rod": "green", "cross": "purple", "star": "orange"}
MARKERS = {"0.0": "o", "0.4": "s", "1.0": "^"}
SERIES_ORDER = [
    "rod0.0", "cross0.0", "star0.0",
    "rod0.4", "cross0.4", "star0.4",
    "rod1.0", "cross1.0", "star1.0",
]


def parse_series_file(filename):
    """Read repeated ``# series`` blocks containing x, mean, and std columns."""
    data = {}
    current_series = None

    with filename.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line[1:].strip().split()
                current_series = parts[0].lower() if parts else None
                if current_series is not None:
                    data.setdefault(current_series, {"x": [], "mean": [], "std": []})
                continue
            if current_series is None:
                continue

            parts = line.replace(",", " ").replace(";", " ").split()
            if len(parts) < 3:
                continue
            try:
                x, mean, std = (float(value) for value in parts[:3])
            except ValueError:
                continue
            data[current_series]["x"].append(x)
            data[current_series]["mean"].append(mean)
            data[current_series]["std"].append(std)

    return data


def split_series_name(name):
    for family in ("rod", "cross", "star"):
        if name.startswith(family):
            return family, name[len(family):]
    raise ValueError(f"Unknown series name: {name}")


def match_series(contact_series, packing_series):
    """Join Z and packing data by sphere count, retaining finite paired values."""
    packing_by_x = {
        x: (mean, std)
        for x, mean, std in zip(
            packing_series["x"], packing_series["mean"], packing_series["std"]
        )
    }
    joined = []
    for x, z, z_std in zip(
        contact_series["x"], contact_series["mean"], contact_series["std"]
    ):
        if x not in packing_by_x:
            continue
        packing, packing_std = packing_by_x[x]
        if all(math.isfinite(value) for value in (packing, packing_std, z, z_std)):
            joined.append((packing, packing_std, z, z_std))
    return np.asarray(joined, dtype=float)


def add_semantic_legends(ax):
    type_legend = ax.legend(
        handles=[
            Line2D([], [], color=COLORS["rod"], marker="s", linestyle="none", label="Rod"),
            Line2D([], [], color=COLORS["cross"], marker="s", linestyle="none", label="Cross"),
            Line2D([], [], color=COLORS["star"], marker="s", linestyle="none", label="Star"),
        ],
        title="Type", loc="upper left", bbox_to_anchor=(0.00, 1.01),
        fontsize=18, title_fontsize=20, frameon=False,
    )
    ax.add_artist(type_legend)
    ax.legend(
        handles=[
            Line2D([], [], color="black", marker=MARKERS["0.0"], linestyle="none", label="0.0"),
            Line2D([], [], color="black", marker=MARKERS["0.4"], linestyle="none", label="0.4"),
            Line2D([], [], color="black", marker=MARKERS["1.0"], linestyle="none", label="1.0"),
        ],
        title="Friction", loc="upper left", bbox_to_anchor=(0.16, 1.01),
        fontsize=18, title_fontsize=20, frameon=False,
    )


def plot_z_vs_packing_fraction(contact_data, packing_data):
    fig, ax = plt.subplots(figsize=(16, 9))
    plotted = 0

    for name in SERIES_ORDER:
        if name not in contact_data or name not in packing_data:
            print(f"Warning: missing contact or packing data for {name}; skipping.")
            continue
        matched = match_series(contact_data[name], packing_data[name])
        if not matched.size:
            print(f"Warning: no paired data for {name}; skipping.")
            continue

        family, friction = split_series_name(name)
        ax.errorbar(
            matched[:, 0], matched[:, 2],
            xerr=matched[:, 1], #yerr=matched[:, 3],
            fmt=MARKERS[friction], linestyle="none", color=COLORS[family],
            markerfacecolor=COLORS[family], markeredgecolor="black",
            markeredgewidth=1.5, markersize=10, ecolor=COLORS[family],
            elinewidth=1.5, capsize=3, capthick=1.5,
            label="_nolegend_", zorder=3,
        )
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No paired contact and packing-fraction series were found.")

    ax.set_xlabel("Packing fraction")
    ax.set_ylabel("Contact Number $Z_\mathrm{sphere}$")
    ax.grid(alpha=0.25)
    add_semantic_legends(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    print(f"Plot saved to {OUTPUT_FILE}")


def main():
    contact_data = parse_series_file(CONTACT_FILE)
    packing_data = parse_series_file(PACKING_FILE)
    plot_z_vs_packing_fraction(contact_data, packing_data)


if __name__ == "__main__":
    main()