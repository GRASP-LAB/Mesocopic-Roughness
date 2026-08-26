"""Read the combined simulation-time file produced by RCS_analysis.py and plot it against friction (mu)."""
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
from matplotlib.lines import Line2D  # type: ignore

COMBINED_PLOT_FOLDER = Path(
    "."
)
TIME_FILE = COMBINED_PLOT_FOLDER / "simulation_time_mean_std_n_spheres_per_rod_all.txt"
OUTPUT_FILE = COMBINED_PLOT_FOLDER / "simulation_time_vs_mu.pdf"

TYPE_COLORS = {"rod": "green", "cross": "purple", "star": "orange"}
FRICTION_MARKERS = {"0.0": "o", "0.4": "s", "1.0": "^"}
TYPE_ORDER = ["rod", "cross", "star"]


def parse_time_file(filename):
    """Read the "# series" blocks (n_spheres_per_rod, mean, std) written by save_plot_information_from_df_with_y_std."""
    data = {}
    current_series = None

    with filename.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line[1:].strip().split()
                current_series = parts[0] if parts else None
                if current_series is not None:
                    data.setdefault(current_series, {"n_spheres_per_rod": [], "mean": [], "std": []})
                continue
            if current_series is None:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                n_spheres, mean, std = (float(value) for value in parts[:3])
            except ValueError:
                continue
            data[current_series]["n_spheres_per_rod"].append(n_spheres)
            data[current_series]["mean"].append(mean)
            data[current_series]["std"].append(std)

    return data


def split_series_name(name):
    match = re.match(r"([a-zA-Z]+)([0-9.]+)", name)
    if not match:
        raise ValueError(f"Unknown series name: {name}")
    return match.group(1), match.group(2)


def build_dataframe(time_data):
    rows = []
    for series_name, series in time_data.items():
        family, friction = split_series_name(series_name)
        for n_spheres, mean, std in zip(series["n_spheres_per_rod"], series["mean"], series["std"]):
            mu_meso = n_to_mu_meso(n_spheres)
            rows.append({
                "type": family,
                "friction": float(friction),
                "n_spheres_per_rod": n_spheres,
                "mu_meso": mu_meso,
                "mean_time": mean,
                "std_time": std,
            })
    return pd.DataFrame(rows)


def n_to_mu_meso(n_value):
    if not isinstance(n_value, (int, float)) or not math.isfinite(n_value) or n_value <= 1:
        return float("nan")
    inside = max(0.0, 1.0 - (2.0 / (n_value - 1.0)) ** 2)
    furrow_parameter = math.sqrt(inside)
    return 1.0 - furrow_parameter


def friction_to_marker(friction):
    friction_key = f"{float(friction):.1f}"
    return FRICTION_MARKERS.get(friction_key, "o")


def plot_time_vs_mu(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    for (family, friction), group in df.groupby(["type", "friction"]):
        group = group.sort_values("mu_meso")
        group = group[group["mu_meso"].notna()]
        if group.empty:
            continue
        color = TYPE_COLORS.get(family, "black")
        marker = friction_to_marker(friction)

        # Keep one curve per (type, friction) while preserving style convention:
        # color = type, marker = friction level.
        ax.errorbar(
            group["mu_meso"], group["mean_time"],# yerr=group["std_time"],
            fmt=marker, linestyle="", color=color,
            markerfacecolor=color, markeredgecolor="black", markeredgewidth=1.0,
            markersize=9, ecolor="black", elinewidth=1.2, capsize=3, capthick=1.2,
            linewidth=1.4, alpha=0.9, zorder=2,
        )

    ax.set_xlabel("Meso roughness")
    ax.set_ylabel("Simulation Time (hours)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)

    type_legend = ax.legend(
        handles=[
            Line2D([], [], color=TYPE_COLORS[t], marker="s", linestyle="none", label=t.capitalize())
            for t in TYPE_ORDER
        ],
        title="Type", loc="lower left", bbox_to_anchor=(0.00, -0.01), frameon=False,
    )
    ax.add_artist(type_legend)

    friction_legend = ax.legend(
        handles=[
            Line2D([], [], color="black", marker=FRICTION_MARKERS["0.0"], linestyle="none", label="0.0"),
            Line2D([], [], color="black", marker=FRICTION_MARKERS["0.4"], linestyle="none", label="0.4"),
            Line2D([], [], color="black", marker=FRICTION_MARKERS["1.0"], linestyle="none", label="1.0"),
        ],
        title="Friction", loc="lower left", bbox_to_anchor=(0.16, -0.01), frameon=False,
    )
    ax.add_artist(friction_legend)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    print(f"Plot saved to {OUTPUT_FILE}")


def main():
    time_data = parse_time_file(TIME_FILE)
    df = build_dataframe(time_data)
    if df.empty:
        raise RuntimeError(f"No data parsed from {TIME_FILE}")
    plot_time_vs_mu(df)


if __name__ == "__main__":
    main()
