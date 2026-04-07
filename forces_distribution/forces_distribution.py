import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FORCES_CACHE_DIR = "./forces_cache"
SAVE_PLOT_FOLDER = "./plots"
TYPES            = ["rod", "cross", "star"]
VALUE_COLS       = ["Fn", "Ft"]
BINS             = 50
FIT_RANGE        = (0.0, 8.0)   # <── plot & fit window

BASE_COLORS  = {"rod": "#2ca02c", "cross": "#9467bd", "star": "#ff7f0e"}
TYPE_MARKERS = {"rod": "o",       "cross": "s",        "star": "^"}

os.makedirs(SAVE_PLOT_FOLDER, exist_ok=True)
fit_results_dir = os.path.join(SAVE_PLOT_FOLDER, "fit_parameters")
os.makedirs(fit_results_dir, exist_ok=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def darken_color(hex_color: str, factor: float = 0.5) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b   = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return "#{:02x}{:02x}{:02x}".format(int(r*factor), int(g*factor), int(b*factor))


def fit_func_simple(f, a, beta):
    return a * np.exp(-beta * np.asarray(f, dtype=float))


def progress_bar(current, total, label="", bar_len=40):
    filled = int(bar_len * current / max(total, 1))
    bar    = "█" * filled + "░" * (bar_len - filled)
    pct    = 100.0 * current / max(total, 1)
    print(f"\r[{bar}] {pct:5.1f}%  {label}", end="", flush=True)
    if current >= total:
        print()


def normalize_by_group_seed_mean(df, value_col, group_cols, seed_col, normalized_col):
    key   = group_cols + [seed_col]
    means = df.groupby(key)[value_col].transform("mean")
    out   = df.copy()
    out[normalized_col] = out[value_col] / means
    return out


def histogram_by_seed_density(df, norm_col, group_cols, seed_col,
                               bins=50, x_range=(0.0, 8.0)):
    """
    Per-(group, seed) probability-density histogram restricted to x_range,
    then average ± std across seeds.
    """
    df = df.dropna(subset=[norm_col])
    df = df[np.isfinite(df[norm_col]) & (df[norm_col] > 0)]
    # restrict to plotting window
    df = df[(df[norm_col] >= x_range[0]) & (df[norm_col] <= x_range[1])]

    edges   = np.linspace(x_range[0], x_range[1], bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths  = np.diff(edges)

    records = []
    keys    = group_cols + [seed_col]
    groups  = list(df.groupby(keys))
    n_grp   = len(groups)

    for idx_g, (key_val, sub) in enumerate(groups, 1):
        progress_bar(idx_g, n_grp, label=f"histogramming group {idx_g}/{n_grp}")
        counts, _ = np.histogram(sub[norm_col].values, bins=edges)
        total      = counts.sum()
        if total == 0:
            continue
        density  = counts / (total * widths)
        key_dict = dict(zip(keys,
                            key_val if isinstance(key_val, tuple) else [key_val]))
        for bc, bw, dens in zip(centers, widths, density):
            records.append({**key_dict, "bin_center": bc,
                             "bin_width": bw, "density": dens})

    if not records:
        return pd.DataFrame()

    raw = pd.DataFrame(records)
    agg = (
        raw.groupby(group_cols + ["bin_center", "bin_width"])["density"]
           .agg(density_mean="mean", density_std="std")
           .reset_index()
    )
    return agg


def assign_colors(n_spheres_values, base):
    darker  = darken_color(base,  0.66)
    darkest = darken_color(base,  0.33)
    n       = len(n_spheres_values)
    cols    = [darkest] * n
    if n >= 1: cols[0] = base
    if n >= 2: cols[1] = darker
    return cols


# ─────────────────────────────────────────────
# STEP 1 – collect fit parameters
# ─────────────────────────────────────────────
fit_params_all = []   # list of dicts → DataFrame at the end

for value_col in VALUE_COLS:
    print(f"\n{'='*65}")
    print(f"  VALUE COLUMN : {value_col}")
    print(f"{'='*65}")

    # one combined figure per value_col
    fig_all, ax_all = plt.subplots(figsize=(10, 7))
    ax_all.set_title(f"{value_col} – all types  [normalised, range {FIT_RANGE}]")
    ax_all.set_xlabel(f"{value_col} / mean per seed")
    ax_all.set_ylabel("Probability density")
    ax_all.set_yscale("log")
    ax_all.grid(True, ls="--", alpha=0.3, which="both")

    for t in TYPES:
        parquet_path = os.path.join(FORCES_CACHE_DIR, f"forces_{t}.parquet")
        if not os.path.exists(parquet_path):
            print(f"  [skip] {parquet_path} not found")
            continue

        # ── 1. LOAD ──────────────────────────────────────────────
        print(f"\n  [load] {parquet_path} …", flush=True)
        chunk_size = 500_000
        raw_df     = pd.read_parquet(parquet_path, engine="pyarrow")
        n_rows     = len(raw_df)
        n_chunks   = max(1, n_rows // chunk_size)
        chunks     = []
        for ci in range(n_chunks):
            s, e = ci * chunk_size, min((ci+1) * chunk_size, n_rows)
            chunks.append(raw_df.iloc[s:e].copy())
            progress_bar(ci+1, n_chunks,
                         label=f"reading chunk {ci+1}/{n_chunks}  ({e:,}/{n_rows:,} rows)")
        forces_long = pd.concat(chunks, ignore_index=True)
        del raw_df, chunks
        gc.collect()
        print(f"  [load] done – {len(forces_long):,} rows", flush=True)

        # ── 2. LAST FRAME ─────────────────────────────────────────
        print(f"  [proc] Selecting last frame …", flush=True)
        if "frame_idx" not in forces_long.columns:
            forces_long["frame_idx"] = (
                forces_long["frame"]
                .str.extract(r"(\d+)")[0]
                .astype("Int64")
            )
        last_idx    = forces_long.groupby("row_id")["frame_idx"].transform("max")
        forces_last = forces_long[forces_long["frame_idx"] == last_idx].copy()
        del forces_long
        gc.collect()
        print(f"  [proc] {len(forces_last):,} rows after last-frame filter", flush=True)

        if value_col not in forces_last.columns:
            print(f"  [skip] Column {value_col} missing – skipping type={t}")
            continue

        # ── 3. NORMALISE ──────────────────────────────────────────
        print(f"  [proc] Normalising by per-seed mean …", flush=True)
        normalized    = normalize_by_group_seed_mean(
            forces_last, value_col=value_col,
            group_cols=["type", "n_spheres_per_rod"],
            seed_col="seed",
            normalized_col=f"{value_col}_norm",
        )
        norm_col_name = f"{value_col}_norm"

        # ── 4. HISTOGRAM (restricted to FIT_RANGE) ────────────────
        print(f"  [proc] Histogramming in range {FIT_RANGE} …", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            summary = histogram_by_seed_density(
                normalized,
                norm_col=norm_col_name,
                group_cols=["type", "n_spheres_per_rod"],
                seed_col="seed",
                bins=BINS,
                x_range=FIT_RANGE,
            )

        if summary.empty:
            print(f"  [warn] Empty summary for type={t}, {value_col}")
            continue

        # trim last bin, require nonzero std
        max_bc  = summary.groupby(["type", "n_spheres_per_rod"])["bin_center"].transform("max")
        summary = summary[summary["bin_center"] < max_bc].copy()
        summary = summary[summary["density_std"] > 0]

        sub = summary[summary["type"] == t].dropna(
                subset=["density_mean", "density_std"])
        if sub.empty:
            print(f"  [warn] No valid bins for type={t}")
            continue

        # ── 5. PER-TYPE FIGURE + FIT ──────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title(f"{value_col}  –  type = {t}   [range {FIT_RANGE}]")
        ax.set_xlabel(f"{value_col} / mean per seed")
        ax.set_ylabel("Probability density")
        ax.set_yscale("log")
        ax.grid(True, ls="--", alpha=0.3, which="both")
        ax.set_xlim(*FIT_RANGE)

        n_spheres_values = np.sort(sub["n_spheres_per_rod"].unique())
        colors           = assign_colors(n_spheres_values, BASE_COLORS.get(t, "#333333"))
        marker           = TYPE_MARKERS.get(t, "o")
        n_groups         = len(n_spheres_values)

        fit_file = os.path.join(fit_results_dir,
                                f"{t}_{value_col}_fit_parameters_simple.txt")
        with open(fit_file, "w") as f_out:
            f_out.write(f"# Fitting results – type={t}, col={value_col}, "
                        f"range={FIT_RANGE}\n")
            f_out.write("n_spheres\ta\tbeta\ta_err\tbeta_err\treduced_chi2\n")

            for idx_n, n_spheres in enumerate(n_spheres_values):
                progress_bar(idx_n+1, n_groups,
                             label=f"fitting n_spheres={int(n_spheres)}")

                sub_n = (sub[sub["n_spheres_per_rod"] == n_spheres]
                         .dropna(subset=["bin_center", "density_mean", "density_std"])
                         .copy())

                color = colors[idx_n]
                lbl   = f"n={int(n_spheres)}"

                # ── plot raw density + std band ──
                ax.errorbar(
                    sub_n["bin_center"], sub_n["density_mean"],
                    yerr=sub_n["density_std"],
                    fmt=marker, color=color, ms=5, lw=0,
                    elinewidth=1, capsize=2, alpha=0.8,
                    label=lbl, zorder=4,
                )
                ax_all.errorbar(
                    sub_n["bin_center"], sub_n["density_mean"],
                    yerr=sub_n["density_std"],
                    fmt=marker, color=color, ms=4, lw=0,
                    elinewidth=1, capsize=2, alpha=0.6,
                    label=f"{t}, n={int(n_spheres)}", zorder=4,
                )

                # ── fit on [1, FIT_RANGE[1]] ──────────────────────
                sub_fit = sub_n[sub_n["bin_center"] > 1.0]
                if len(sub_fit) < 3:
                    print(f"\n  [warn] Too few points >1 for n={int(n_spheres)}")
                    f_out.write(f"{int(n_spheres)}\tnan\tnan\tnan\tnan\tnan\n")
                    continue

                x_data = sub_fit["bin_center"].values.astype(float)
                y_data = sub_fit["density_mean"].values.astype(float)
                y_err  = np.maximum(sub_fit["density_std"].values.astype(float), 1e-10)

                try:
                    params, cov = curve_fit(
                        fit_func_simple, x_data, y_data,
                        p0=[float(np.mean(y_data)),
                            1.0 / max(float(np.mean(x_data)), 1e-6)],
                        sigma=y_err, absolute_sigma=True,
                        bounds=([0, 0], [float(max(y_data)) * 20, 200]),
                        maxfev=100_000,
                    )
                    a, beta = params
                    perr    = np.sqrt(np.diag(cov)) if cov is not None else [np.nan]*2
                    a_err, beta_err = perr

                    dof          = max(len(y_data) - 2, 1)
                    residuals    = y_data - fit_func_simple(x_data, a, beta)
                    reduced_chi2 = np.sum((residuals / y_err)**2) / dof

                    x_fit       = np.linspace(FIT_RANGE[0], FIT_RANGE[1], 300)
                    y_fit       = fit_func_simple(x_fit, a, beta)
                    y_fit_upper = fit_func_simple(x_fit, a + a_err, beta - abs(beta_err))
                    y_fit_lower = fit_func_simple(x_fit, a - a_err, beta + abs(beta_err))

                    ax.plot(x_fit, y_fit, color=color, ls="--", lw=2, zorder=5)
                    ax.fill_between(x_fit,
                                    np.clip(y_fit_lower, 1e-12, None),
                                    y_fit_upper,
                                    color=color, alpha=0.18, zorder=3)
                    ax_all.plot(x_fit, y_fit, color=color, ls="--", lw=1.5, zorder=5)

                    f_out.write(f"{int(n_spheres)}\t{a:.6f}\t{beta:.6f}\t"
                                f"{a_err:.6f}\t{beta_err:.6f}\t{reduced_chi2:.6f}\n")

                    fit_params_all.append(dict(
                        type=t, n_spheres=int(n_spheres), value_col=value_col,
                        a=a, a_err=a_err, beta=beta, beta_err=beta_err,
                        reduced_chi2=reduced_chi2,
                    ))
                    print(f"  [fit ] n={int(n_spheres):3d}  "
                          f"a={a:.4f}±{a_err:.4f}  "
                          f"β={beta:.4f}±{beta_err:.4f}  "
                          f"χ²_r={reduced_chi2:.3f}")

                except Exception as exc:
                    print(f"\n  [warn] Fit failed n={int(n_spheres)}: {exc}")
                    f_out.write(f"{int(n_spheres)}\tnan\tnan\tnan\tnan\tnan\n")

        ax.legend(title="n spheres / rod", fontsize=8)
        fig.tight_layout()
        out_path = os.path.join(SAVE_PLOT_FOLDER,
                                f"{value_col}_{t}_distribution_fit.pdf")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  [save] {out_path}")

        del forces_last, normalized, summary, sub
        gc.collect()

    # combined figure
    ax_all.legend(fontsize=6, ncol=3, title="type, n spheres")
    ax_all.set_xlim(*FIT_RANGE)
    fig_all.tight_layout()
    out_all = os.path.join(SAVE_PLOT_FOLDER, f"{value_col}_all_types_distribution.pdf")
    fig_all.savefig(out_all, dpi=150)
    plt.close(fig_all)
    print(f"\n  [save] combined → {out_all}")


# ─────────────────────────────────────────────
# STEP 2 – plot A and β vs n_spheres
# ─────────────────────────────────────────────
if not fit_params_all:
    print("\n[summary] No fit parameters collected – nothing to plot.")
else:
    df_params = pd.DataFrame(fit_params_all)
    csv_path  = os.path.join(SAVE_PLOT_FOLDER, "fit_parameters_all.csv")
    df_params.to_csv(csv_path, index=False)
    print(f"\n[summary] {len(df_params)} parameter rows saved → {csv_path}")

    for value_col in VALUE_COLS:
        sub_vc = df_params[df_params["value_col"] == value_col].copy()
        if sub_vc.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"Fit parameters vs n_spheres_per_rod  –  {value_col}",
            fontsize=13, fontweight="bold",
        )

        param_cfg = [
            ("a",    "a_err",    "Amplitude  A",          axes[0]),
            ("beta", "beta_err", "Decay rate  β",         axes[1]),
        ]

        legend_handles = []

        for t in TYPES:
            sub_t = sub_vc[sub_vc["type"] == t].sort_values("n_spheres")
            if sub_t.empty:
                continue

            base   = BASE_COLORS.get(t,  "#333333")
            marker = TYPE_MARKERS.get(t, "o")
            colors = assign_colors(sub_t["n_spheres"].values, base)

            for (pcol, ecol, ylabel, ax) in param_cfg:
                for i, (_, row) in enumerate(sub_t.iterrows()):
                    ax.errorbar(
                        row["n_spheres"], row[pcol],
                        yerr=row[ecol],
                        fmt=marker,
                        color=colors[i],
                        ms=10,
                        markeredgecolor="black",
                        markeredgewidth=1.2,
                        elinewidth=1.5,
                        ecolor=colors[i],
                        capsize=4,
                        capthick=1.5,
                        zorder=4,
                    )
                # connect points of same type with a thin line
                # ax.plot(
                #     sub_t["n_spheres"], sub_t[pcol],
                #     color=base, lw=1.2, ls="-", alpha=0.5, zorder=3,
                # )

                # annotate each point with its n_spheres label
                for _, row in sub_t.iterrows():
                    ax.annotate(
                        f"{t}\nn={int(row['n_spheres'])}",
                        xy=(row["n_spheres"], row[pcol]),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=7, color=base,
                    )

            # legend entry (one per type)
            legend_handles.append(
                Line2D([0], [0], marker=marker, color=base,
                       markeredgecolor="black", markeredgewidth=1,
                       linestyle="-", linewidth=1.2,
                       markersize=9, label=t)
            )

        for (pcol, ecol, ylabel, ax) in param_cfg:
            ax.set_xlabel("n spheres per rod", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, ls="--", alpha=0.35)
            # integer x-ticks
            all_n = sorted(df_params["n_spheres"].unique())
            ax.set_xticks(all_n)
            ax.set_xticklabels([str(n) for n in all_n])

        # deduplicate legend entries
        seen = set()
        unique_handles = []
        for h in legend_handles:
            if h.get_label() not in seen:
                seen.add(h.get_label())
                unique_handles.append(h)

        fig.legend(
            unique_handles, [h.get_label() for h in unique_handles],
            title="grain type", loc="lower center",
            ncol=len(TYPES), fontsize=10,
            bbox_to_anchor=(0.5, -0.04),
        )
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        out_fit = os.path.join(SAVE_PLOT_FOLDER,
                               f"{value_col}_fit_params_vs_n_spheres.pdf")
        fig.savefig(out_fit, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] fit-parameter plot → {out_fit}")

print("\n[done]")
