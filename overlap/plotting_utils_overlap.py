import math
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
from pathlib import Path
from matplotlib.cm import get_cmap

# --------------------------------------------------
# Screen/figure size (inches)
DIM_SCREEN = (4096/320, 1800/160)  # (width, height)
# --------------------------------------------------

# Typography (your settings)
plt.rcParams['font.size'] = 20           # default
plt.rcParams['axes.titlesize'] = 42
plt.rcParams['axes.labelsize'] = 36
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['legend.fontsize'] = 32

# Facet order and base colors (as requested)
order = [("rod", "green"), ("cross", "purple"), ("star", "orange")]
colors: Dict[str, str] = {k: v for k, v in order}


def _read_loose_table(file: Path, expected_cols) -> pd.DataFrame:
    # 1) try TSV
    try:
        df = pd.read_csv(file, sep="\t", engine="python", on_bad_lines="skip")
        return df
    except Exception:
        pass
    # 2) try whitespace-delimited
    try:
        df = pd.read_csv(file, delim_whitespace=True, engine="python", on_bad_lines="skip")
        return df
    except Exception:
        pass

    # 3) manual fallback: normalize separators per line
    rows = []
    with open(file, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        header_parts = [p.strip().lower() for p in header.replace(",", "\t").replace(";", "\t").split()]
        if len(header_parts) < 3 or "facet" not in header.lower():
            header_parts = expected_cols
        for line in f:
            line = line.strip()
            if not line:
                continue
            norm = line.replace(",", "\t").replace(";", "\t")
            parts = [p for p in norm.split() if p]
            if len(parts) < min(3, len(expected_cols)):
                continue
            rows.append(parts[:len(expected_cols)])
    df = pd.DataFrame(rows, columns=expected_cols[:len(rows[0])])
    return df

def _coerce_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    expected_cols = ["facet", "hue", "bin_center", "mean_density", "error", "n_seeds"]
    # normalize header names
    df.columns = [str(c).strip().lower() for c in df.columns]
    # best-effort renames (spaces → underscores)
    rename_map = {}
    for c in df.columns:
        cu = c.replace(" ", "_")
        if cu != c:
            rename_map[c] = cu
    if rename_map:
        df = df.rename(columns=rename_map)

    # ensure all expected columns exist (create if missing)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan

    # types
    df["facet"] = df["facet"].astype(str).str.strip().str.lower()
    for c in ["hue", "bin_center", "mean_density", "error", "n_seeds"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # keep only known facets
    df = df[df["facet"].isin(["rod", "cross", "star"])]
    return df

def _impute_series_values(g: pd.DataFrame, cols=("mean_density", "error")) -> pd.DataFrame:
    # Sort by bin_center for interpolation
    g = g.sort_values("bin_center").copy()

    # Interpolate within the series (facet,hue) along bin_center.
    for c in cols:
        if c not in g:
            continue
        # linear interpolate where possible
        g[c] = g[c].interpolate(method="linear", limit_direction="both")

    # If error is still missing, try a local estimate:
    # - if multiple rows share the same bin_center, use their std as a proxy;
    # - otherwise fall back to a small epsilon so the point can be drawn.
    if "error" in g.columns:
        still_na = g["error"].isna()
        if still_na.any():
            # estimate per-bin std across duplicates if available
            est = (
                g.groupby("bin_center")["mean_density"]
                 .transform(lambda s: s.std(ddof=1) if len(s) > 1 else np.nan)
            )
            g.loc[still_na & est.notna(), "error"] = est[still_na & est.notna()]
            # final fallback
            g["error"] = g["error"].fillna(0.0)

    # Any mean_density still missing after interpolation → drop those rows
    g = g[~g["mean_density"].isna()]
    # Ensure non-negative error
    if "error" in g.columns:
        g.loc[g["error"] < 0, "error"] = 0.0
    return g

def _largest_remainder_integer_scaling(weights: np.ndarray, target: int) -> np.ndarray:
    """
    Scale non-negative weights so their integer-rounded values sum to target.
    Returns integer array summing exactly to target.
    """
    weights = np.nan_to_num(weights, nan=0.0)
    total = weights.sum()
    if total <= 0:
        # nothing to scale: distribute uniformly
        base = np.full(len(weights), target // max(1, len(weights)), dtype=int)
        remainder = target - base.sum()
        base[:remainder] += 1
        return base

    scaled = weights * (target / total)
    floored = np.floor(scaled).astype(int)
    remainder = target - floored.sum()
    if remainder > 0:
        # assign the largest fractional parts an extra 1
        frac_order = np.argsort(-(scaled - floored))
        floored[frac_order[:remainder]] += 1
    return floored

def _normalize_n_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (facet, hue), make sure the total n_seeds equals exactly 100.
    Keeps original counts in 'original_n_seeds' and writes normalized counts to 'n_seeds'.
    """
    df = df.copy()
    if "original_n_seeds" not in df.columns:
        df["original_n_seeds"] = df["n_seeds"]

    out = []
    for (facet, hue), g in df.groupby(["facet", "hue"], sort=False):
        counts = g["n_seeds"].to_numpy()
        # if all NaN or sum==0, use uniform distribution
        counts = np.nan_to_num(counts, nan=0.0)
        n_int = _largest_remainder_integer_scaling(counts, target=100)
        gi = g.copy()
        gi["n_seeds"] = n_int
        out.append(gi)
    return pd.concat(out, axis=0).sort_values(["facet", "hue", "bin_center"]).reset_index(drop=True)

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load loosely formatted tabular data; tolerate missing/garbled lines.
    Impute missing mean_density/error within each (facet,hue) along bin_center.
    Guarantee exactly 100 n_seeds per (facet,hue) after normalization.
    """
    file = Path(filepath)
    if not file.exists():
        raise FileNotFoundError(f"Could not find data file: {filepath}")

    expected_cols = ["facet", "hue", "bin_center", "mean_density", "error", "n_seeds"]
    raw = _read_loose_table(file, expected_cols=expected_cols)
    before_read = len(raw)

    df = _coerce_and_standardize(raw)

    # Drop rows missing both facet or hue or bin_center (can’t place them)
    df = df.dropna(subset=["facet", "hue", "bin_center"])

    # Impute per (facet,hue)
    parts = []
    for (facet, hue), g in df.groupby(["facet", "hue"], sort=True):
        gi = _impute_series_values(g, cols=("mean_density", "error"))
        parts.append(gi)
    df = pd.concat(parts, axis=0) if parts else pd.DataFrame(columns=expected_cols)

    # Clean oddities
    df = df[(df["bin_center"].notna()) & (df["mean_density"].notna())]
    df.loc[df["error"].isna(), "error"] = 0.0
    df.loc[df["error"] < 0, "error"] = 0.0

    # Ensure integer n_seeds where present
    df["n_seeds"] = pd.to_numeric(df["n_seeds"], errors="coerce")
    # If some rows lack n_seeds, set to 0 (they’ll be normalized next)
    df["n_seeds"] = df["n_seeds"].fillna(0).clip(lower=0)

    # Normalize counts to exactly 100 per (facet,hue)
    df = _normalize_n_seeds(df)

    # Final ordering
    df = df.sort_values(["facet", "hue", "bin_center"]).reset_index(drop=True)

    # Save snapshot
    df.to_csv("cleaned_overlap_density.csv", index=False)

    # Reporting
    kept = len(df)
    # Per-group check summary
    summary = (
        df.groupby(["facet", "hue"])["n_seeds"]
          .sum()
          .reset_index(name="n_seeds_total")
    )
    bad = summary[summary["n_seeds_total"] != 100]
    if not bad.empty:
        print("Warning: some groups did not normalize to 100 (unexpected):")
        print(bad.to_string(index=False))
    print(f"Loaded ~{before_read} raw lines, kept {kept} rows after cleaning and imputation.")
    print("Wrote cleaned_overlap_density.csv with normalized n_seeds (exactly 100 per facet,hue) and original_n_seeds preserved.")

    return df

def plot_group_symlog(ax, x, y, err, label, i_fmt,color):
    y = np.asarray(y, float)
    err = np.asarray(err, float)
    yerr_low  = np.minimum(err, y)     # don't go below 0
    yerr_high = err
    yerr = np.vstack([yerr_low, yerr_high])
    ax.errorbar(
        x, y, fmt=i_fmt, color=color, #label=label,
        markersize=14,
        markeredgecolor="black",
        markeredgewidth=1.5,
        elinewidth=2,
        capsize=6,
        alpha=0.95, errorevery=max(1, len(x)//150),
    )

from matplotlib.colors import to_rgb, to_hex

# base colors for facet (light variants for h==3)
# BASE_FACET_COLORS = {
#     "rod": "#66b637ff",    # green (light)
#     "cross": "#ef4de7ff",  # purple (you can pick another hex if you prefer)
#     "star": "#fcb424ff",   # orange (light)
# }
BASE_FACET_COLORS = {
    "rod": "green",    # green (light)
    "cross": "purple",  # purple (light)
    "star": "orange",   # orange (light)
}

def darken_color(hex_color, factor=0.6):
    """
    Return a darker version of hex_color.
    factor in (0,1] where smaller -> darker. e.g. 0.6 reduces brightness to 60%.
    """
    r, g, b = to_rgb(hex_color)
    r, g, b = (r * factor, g * factor, b * factor)
    return to_hex((r, g, b))


from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

def save_facet_pdf(fig, ax, df, facet, i_fmt='o'):
    sub = df[df["facet"] == facet].sort_values("bin_center")
    hue_vals = sorted(sub["hue"].unique())

    base   = BASE_FACET_COLORS.get(facet, "#000000")
    darker = darken_color(base, factor=0.66)
    darkest    = darken_color(base, factor=0.33)  # your “other” hue color

    for h in hue_vals:
        g = sub[sub["hue"] == h]
        g = g[g["mean_density"] != 0]

        mask = (
            (g["bin_center_renorm"] > 0) &
            (g["mean_density_renorm"] > 0) &
            (g["error_renorm"] >= 0)
        )
        g = g.loc[mask]
        
        if g.empty:
            continue

        if h == 3:
            col = base
        elif h>3:
            continue
        elif h == 4:
            col = darker
        else:
            col = darkest
        plot_group_symlog(
            ax,
            g["bin_center_renorm"].to_numpy(),
            g["mean_density_renorm"].to_numpy(),
            g["error_renorm"].to_numpy(),
            label=f"h={h}",
            color=col,
            i_fmt=i_fmt,
        )

    # ax.set_xscale("symlog", linthresh=1e-5, linscale=1.0)
    # ax.set_yscale("symlog", linthresh=1e-0, linscale=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_ylim(bottom=-0.25)
    ax.set_xlabel("Pairwise overlap (sphere volume fraction)")
    ax.set_ylabel("Mean density of overlaps")



    return fig

def plot_all_to_pdfs(df):
    fmt = ['o','o','o']

    fig, ax = plt.subplots(1, 1, figsize=DIM_SCREEN)

    for i, (facet, _) in enumerate(order):
        file_name = f"mean_density.pdf"
        fig = save_facet_pdf(fig, ax, df, facet, i_fmt=fmt[i])
        # save per-facet file if you prefer; here we overwrite file_name each loop
        fig.tight_layout()

    # Add legend (show facets mapped to marker shapes)
    ax.plot([], [], fmt[0], label='rod', color=BASE_FACET_COLORS["rod"])
    ax.plot([], [], fmt[1], label='cross', color=BASE_FACET_COLORS["cross"])
    ax.plot([], [], fmt[2], label='star', color=BASE_FACET_COLORS["star"])
    #plt.legend(loc="best", frameon=False, ncol=1,markerscale=2.5)
    fig.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close(fig)

def sphere_volume(radius):
    return (4/3) * np.pi * (radius ** 3)

def renormalization_factor(shape_type, r, n_minimal_sphere):
    individual_volume = sphere_volume(r)
    steinmetz2_volume = (16.0/3.0) * (r**3)
    steinmetz3_volume = 8.0 * (2.0 - np.sqrt(2.0)) * (r**3)
    rod_volume = np.pi * (r**2) * ((n_minimal_sphere - 1) * 2.0 * r) + individual_volume

    if shape_type == "rod":
        denom = rod_volume
    elif shape_type == "cross":
        denom = (2.0 * rod_volume - steinmetz2_volume)
    elif shape_type == "star":
        denom = (3.0 * rod_volume - 3.0 * steinmetz2_volume + steinmetz3_volume)
    else:
        raise ValueError(f"Unknown shape_type {shape_type}")

    return denom/individual_volume

if __name__ == "__main__":
    LIMIT_OVERLAP = 10*(5e-04)**2
    data_path = "overlap_density_meaned_distrib_new_data.txt"
    df_clean = load_and_clean(data_path)
    df_clean['furrow_param'] = df_clean['hue'].apply(
    lambda N: 1-math.sqrt((1.0 - (2.0 / (N - 1.0)))**0.5) if isinstance(N, (int, float)) and math.isfinite(N) and N > 1 else float('nan')
    )

    # renormalize ? because overlap was initially measured with respect to the volume of the particles
    r = 0.0025  # example
    n_minimal_sphere = 3  # example

    factor_map = {
        "rod": renormalization_factor("rod", r, n_minimal_sphere),
        "cross": renormalization_factor("cross", r, n_minimal_sphere),
        "star": renormalization_factor("star", r, n_minimal_sphere),
    }

    df_clean["renorm_factor"] = df_clean["facet"].map(factor_map)

    # Transform variable
    df_clean["bin_center_renorm"] = df_clean["bin_center"] * df_clean["renorm_factor"]

    # Apply Jacobian correction to PDF
    df_clean["mean_density_renorm"] = df_clean["mean_density"] / df_clean["renorm_factor"]
    df_clean["error_renorm"] = df_clean["error"] / df_clean["renorm_factor"]
    df_filt = df_clean[df_clean["bin_center_renorm"] >= LIMIT_OVERLAP].copy()
    plot_all_to_pdfs(df_filt)

    facet_colors = {
        "rod":   "green",   # green
        "cross": "purple",   # purple
        "star":  "orange",   # orange
    }
 
    markers = {
        "rod": "o",
        "cross": "o",
        "star": "o",
    }
    #mean - std 
    def weighted_hist_stats_mean_std(group: pd.DataFrame) -> pd.Series:
        """
        Calculate weighted statistics for a group of data.

        Parameters:
        group (pd.DataFrame): A DataFrame containing the data to analyze

        Returns:
        pd.Series: A Series containing the calculated statistics
        """
        # Extract the data from the group
        x = group["bin_center_renorm"].to_numpy(dtype=float)  # midpoint
        f = group["mean_density_renorm"].to_numpy(dtype=float)  # frequency (density)

        # Calculate the sum of frequencies
        sum_f = np.sum(f)

        # Handle the case where sum_f is zero to avoid division by zero
        if sum_f == 0:
            return pd.Series({
                "mean": np.nan,
                "std": np.nan,
                "sum_f": sum_f
            })

        # Calculate weighted mean and variance
        fx = f * x
        fx2 = f * x * x

        mean = fx.sum() / sum_f
        var = fx2.sum() / sum_f - mean**2

        # Ensure variance is non-negative for numerical stability
        std = np.sqrt(max(var, 0.0))

        # Return the calculated statistics as a Series
        return pd.Series({
            "mean": mean,
            "std": std,
            "sum_f": sum_f
        })
    
    # First, apply the function and create the stats DataFrame
    stats = (
        df_filt
        .groupby(["facet", "furrow_param"], sort=True)
        .apply(weighted_hist_stats_mean_std)
        .reset_index()
    )

    # Pivot the DataFrame for each statistic
    mean_pivoted = stats.pivot(index="facet", columns="furrow_param", values="mean")
    std_pivoted = stats.pivot(index="facet", columns="furrow_param", values="std")

    # Get unique furrow_param values for plotting
    hues_sorted = np.sort(mean_pivoted.columns.unique())

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=DIM_SCREEN)

    # Plot the mean values with error bars for each facet
    for facet in mean_pivoted.index:
        # Get the color and marker for this facet
        color = facet_colors[facet]
        marker = markers[facet]

        # Get the mean and standard deviation for this facet across all furrow_params
        means = mean_pivoted.loc[facet]
        errors = std_pivoted.loc[facet]

        # Plot the data with error bars
        ax.errorbar(
            x=hues_sorted,  # x positions (furrow_param values)
            y=means,        # y values (means)
            yerr=errors,    # error bars (std)
            label=facet,    # label for the legend (though we won't show it)
            marker=marker,  # marker style
            color=color,    # color
            linestyle="",  # line style
            markersize=10*2,
            markeredgewidth=2,
            markerfacecolor=color,
            markeredgecolor='black',
            elinewidth=2,
            ecolor='black',                      # or ecolor=color to match fill
            capsize=3,
            capthick=2,
            alpha=1.0,                           # optional
            zorder=3   
        )

    # Customize the plot
    ax.set_xlabel('Meso Roughness')  # label for the x-axis
    ax.set_ylabel('Mean Density')      # label for the y-axis
    # Set x-ticks to be the furrow_param values
    ax.set_xticks(hues_sorted)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_box_aspect(0.5)
    ax.set_ylim(top=1.5e-2)
    # Show the plot
    plt.tight_layout()
    plt.savefig("overlap_mean_std.pdf", bbox_inches="tight")
    plt.close()