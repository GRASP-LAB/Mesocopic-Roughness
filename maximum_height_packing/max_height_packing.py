import csv
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

BASE_FOLDER = ""
SAVE_PLOT_FOLDER = BASE_FOLDER + ""
DATA_FILE = SAVE_PLOT_FOLDER + "maximum_height_mean_std_n_spheres_per_rod.txt"
PACK_FILE2 = SAVE_PLOT_FOLDER + "packing_fraction_mean_std_n_spheres_per_rod.txt"
CST = 6.66/4  # N v_inf / (4 lx ly 6R)


DIM_SCREEN =  4096/320,1800/160
plt.rcParams['font.size'] = 20  # Default font size for all text
plt.rcParams['axes.titlesize'] = 42  # Title font size
plt.rcParams['axes.labelsize'] = 36  # Axis labels (x, y)
plt.rcParams['xtick.labelsize'] = 30  # X-tick labels
plt.rcParams['ytick.labelsize'] = 30  # Y-tick labels
plt.rcParams['legend.fontsize'] = 20  # Legend font size


def N_to_furrow_param(series):
    xs = series["x"]  # a Python list of N values
    alphas = []
    for N in xs:
        # validity checks
        if not isinstance(N, (int, float)) or not math.isfinite(N) or N <= 1:
            alphas.append(float("nan"))
            continue
        #furrow relative depth      (1.0 - (2.0 / (N - 1.0)))**0.5 --> choose this one
        #furrow relative volume     (1.0 - (2.0 / (N - 1.0)))
        inside = (1.0 - (2.0 / (N - 1.0)))**(0.5)
        if inside < 0.0:
            inside = 0.0
        alphas.append(math.sqrt(inside))
    series["x"] = alphas  # overwrite with alpha
def parse_data(filename):
    """
    Parses the data file structured with blocks starting with '# type'
    followed by rows: x mean std
    Returns dict: {type: {'x': [...], 'y': [...], 'yerr': [...]}}
    """
    data = {}
    current_type = None
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # e.g. "# rod"
                parts = line[1:].strip().split()
                if parts:
                    current_type = parts[0].lower()
                    data.setdefault(current_type, {"x": [], "y": [], "yerr": []})
                continue
            # data row: x mean std
            if current_type is None:
                continue
            parts = line.split()
            if len(parts) >= 3:
                x, mean, std = float(parts[0]), float(parts[1]), float(parts[2])
                data[current_type]["x"].append(x)
                data[current_type]["y"].append(mean)
                data[current_type]["yerr"].append(std)
    return data
def parse_fit_params(filename):
    """
    Parses the fit parameters TSV with header:
    #Type c1 c1_error c2 c2_error chi2 dof
    Returns dict: {type: {'c1':..., 'c1_err':..., 'c2':..., 'c2_err':...}}
    """
    fits = {}
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = None
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                header = [h.strip() for h in row[0][1:].split()] + [h.strip() for h in row[1:]]
                continue
            # Some files might use single tabs; ensure we have expected columns
            if len(row) == 1 and "\t" in row[0]:
                row = row[0].split("\t")
            if len(row) < 6:
                continue
            t = row[0].strip().lower()
            c1 = float(row[1]); c1_err = float(row[2])
            c2 = float(row[3]); c2_err = float(row[4])
            fits[t] = {"c1": c1, "c1_error": c1_err, "c2": c2, "c2_error": c2_err}
    return fits
def parse_packing_file(filename):
    """
    Parse packing fraction file format:
    Blocks starting with "# type" followed by lines: alpha mean std
    Handles occasional malformed lines (skips them).
    Returns dict: {type: {'x': [...], 'y': [...], 'yerr': [...]}} where x is alpha already.
    """
    data = {}
    current = None
    with open(filename, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line[1:].strip().split()
                if parts:
                    current = parts[0].lower()
                    data.setdefault(current, {"x": [], "y": [], "yerr": []})
                continue
            # try to be robust: replace common bad separators
            cleaned = line.replace(";", " ").replace(",", " ").replace("\t", " ").strip()
            parts = cleaned.split()
            if len(parts) < 3:
                continue
            try:
                a = float(parts[0])
                mean = float(parts[1])
                std = float(parts[2])
            except Exception:
                continue
            if current is None:
                continue
            data[current]["x"].append(a)
            data[current]["y"].append(mean)
            data[current]["yerr"].append(std)
    return data
def linspace(a, b, n=201):
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]
def plot_dataset_on_ax(ax, name, series, fit_params, color):
    """
    series: dict-like containing lists/arrays series['x'], series['y'], series['yerr']
    fit_params: dict with keys "c1", "c1_error", "c2", "c2_error"
    color: color string
    """
    
    # convert x (furrow depth) to meso roughness
    x = 1-np.asarray(series["x"], dtype=float)
    y = np.asarray(series["y"], dtype=float)
    yerr = np.asarray(series.get("yerr", np.zeros_like(y)), dtype=float)

    # scatter + errorbars
    ax.errorbar(x, y, yerr=yerr,
                fmt='o', linestyle='none',           # no connecting line
                label=name,
                markersize=10,
                markeredgewidth=1.5,
                markerfacecolor=color,
                markeredgecolor='black',
                elinewidth=1.5,
                ecolor='black',                      # or ecolor=color to match fill
                capsize=3,
                capthick=1.5,
                alpha=1.0,                           # optional
                zorder=3                             # helps draw above grids
            )

    # fit params
    c1 = fit_params["c1"]
    c1_err = fit_params["c1_error"]
    c2 = fit_params["c2"]
    c2_err = fit_params["c2_error"]

    # line for fit
    x_line = np.linspace(np.nanmin(x[~np.isnan(x)]), np.nanmax(x[~np.isnan(x)]), 200)
    y_line = c1 * (1-x_line) + c2
    ax.plot(x_line, y_line, '-', color=color)  # main fit line
    ax.plot([], [], '-', color="grey",label="Fit")  # main fit line

    # 3) Combined envelope: both parameters shifted together (conservative)
    y_comb_plus  = (c1 + c1_err) * (1-x_line) + (c2 + c2_err)
    y_comb_minus = (c1 - c1_err) * (1-x_line) + (c2 - c2_err)
    # Use a slightly stronger alpha so the combined envelope is visible but not overwhelming
    ax.fill_between(x_line, y_comb_minus, y_comb_plus, color=color, alpha=0.07, linewidth=0)#,label=f"{name} ±1σ (combined)")
    ax.plot(x_line,[c1+c2]*len(x_line), '--', color=color)
    ax.set_xscale('log')
def packing_model(alpha, c1, c2, cst):
    """
    Vectorized packing fraction model:
      phi(alpha) = cst * ((alpha^2 + 3) / (c1*alpha + c2))

    alpha may be a scalar, list, or numpy array. c1, c2 should be numeric.
    Returns numpy array of same shape as alpha.
    """
    alpha = np.asarray(alpha, dtype=float)
    c1 = float(c1)
    c2 = float(c2)

    denom = (c1 * alpha + c2)            # vectorized
    # avoid division by zero: wherever denom == 0 set result to np.nan
    safe = denom != 0.0
    out = np.full(alpha.shape, np.nan, dtype=float)
    out[safe] = cst * ((alpha[safe]**2 + 3.0) / denom[safe])
    return out
def plot_packing_fraction(ax, pack_data, fits, colors, cst):
    """
    pack_data: dict like {'rod': {'x':[...],'y':[...],'yerr':[...], ...}, ...}
    fits: dict like {'rod': {'c1':..,'c1_error':..,'c2':..,'c2_error':..}, ...}
    colors: dict or mapping name->color
    """
    # Enforce plotting order: rod -> cross -> star (others, if any, after)
    priority = {'rod': 0, 'cross': 1, 'star': 2}
    ordered_names = sorted(pack_data.keys(), key=lambda n: priority.get(n, 99))

    for name in ordered_names:
        d = pack_data[name]

        N_to_furrow_param(d)  # uncomment if pack_data["x"] are n_spheres_per_rod

        # convert x (furrow depth) to meso roughness
        x = 1-np.asarray(d["x"], dtype=float)
        y = np.asarray(d["y"], dtype=float)
        yerr = np.asarray(d.get("yerr", np.zeros_like(y)), dtype=float)

        color = colors.get(name, None)

        ax.errorbar(
            x, y, yerr=yerr,
            fmt='o', linestyle='none',
            label=name,
            markersize=10,
            markeredgewidth=1.5,
            markerfacecolor=color,
            markeredgecolor='black',
            elinewidth=1.5,
            ecolor='black',
            capsize=3,
            capthick=1.5,
            alpha=1.0,
            zorder=3
        )

        # fit params
        if name not in fits:
            continue
        fp = fits[name]
        c1 = float(fp["c1"])
        c1_err = float(fp["c1_error"])
        c2 = float(fp["c2"])
        c2_err = float(fp["c2_error"])

        # x grid for model
        a_min = np.nanmin(x[~np.isnan(x)]) if x.size else 0.0
        a_max = np.nanmax(x[~np.isnan(x)]) if x.size else 1.0
        a_line = np.linspace(a_min, a_max, 300)

        # nominal model
        # use again meso roughness
        y_nom = packing_model(1-a_line, c1, c2, cst=cst)
        ax.plot(a_line, y_nom, '-', color=color)  # model line
        ax.plot([], [], '-', color="grey", label=f"Model")

        # envelope from +/- errors
        combos = [
            (c1 + c1_err, c2 + c2_err),
            (c1 + c1_err, c2 - c2_err),
            (c1 - c1_err, c2 + c2_err),
            (c1 - c1_err, c2 - c2_err),
        ]
        # use again meso roughness
        y_combo = np.vstack([packing_model(1-a_line, c1p, c2p, cst=cst) for (c1p, c2p) in combos])
        y_min = np.nanmin(y_combo, axis=0)
        y_max = np.nanmax(y_combo, axis=0)

        valid = ~np.isnan(y_min) & ~np.isnan(y_max)
        if np.any(valid):
            ax.fill_between(a_line[valid], y_min[valid], y_max[valid],
                            color=color, alpha=0.15, linewidth=0)
        ax.plot(a_line, packing_model([1]*len(a_line), c1, c2, cst=cst), '--', color=color)

    ax.set_xlabel("alpha")
    ax.set_ylabel("Packing fraction")
    ax.grid(alpha=0.25)
    ax.set_xscale('log')
def linear_model(x, c1, c2):
    """Linear fit function."""
    return c1 * (1- x) + c2   
def fit_data(data):
    """
    Perform a linear fit for each type separately.
    
    Parameters
    ----------
    data : dict
        {type: {"x": [...], "y": [...], "yerr": [...]}}

    Returns
    -------
    dict : {type: {"c1": ..., "c1_error": ..., "c2": ..., "c2_error": ..., "chi2": ..., "dof": ...}}
    """
    fits = {}
    for t, series in data.items():
        # Convert x to alpha (in place)
        N_to_furrow_param(series)
        # convert x (furrow depth) to meso roughness
        x = 1- np.asarray(series["x"], dtype=float)
        y = np.asarray(series["y"], dtype=float)
        yerr = np.asarray(series.get("yerr", np.ones_like(y)), dtype=float)

        # Initial guess for [c1, c2]
        p0 = [1.0, np.mean(y)]

        try:
            popt, pcov = curve_fit(
                linear_model, x, y, p0=p0, sigma=yerr,
                absolute_sigma=True, maxfev=10000
            )
        except RuntimeError:
            print(f"Fit failed for {t}")
            continue

        # Extract parameters and errors
        c1, c2 = popt
        perr = np.sqrt(np.diag(pcov))
        c1_err, c2_err = perr

        # Chi² / dof
        residuals = (y - linear_model(x, *popt)) / yerr
        chi2 = np.sum(residuals**2)
        dof = max(len(y) - len(popt), 1)

        fits[t] = {
            "c1": c1, "c1_error": c1_err,
            "c2": c2, "c2_error": c2_err,
            "chi2": chi2, "dof": dof
        }

        print(f"Fit for {t}: c1={c1:.4f} ± {c1_err:.4f}, c2={c2:.4f} ± {c2_err:.4f}, chi²/dof={chi2/dof:.2f}")

    return fits

# --- main that draws everything into one plot ---
def main():
    # parse_data and parse_fit_params must produce the following shapes:
    data = parse_data(DATA_FILE)
    fits=fit_data(data)
    pack_data = parse_packing_file(PACK_FILE2)
    # Order and colors as requested: first rod (green), then cross (purple), then star (orange)
    order = [("rod", "green"), ("cross", "purple"), ("star", "orange")]
    colors = {k: v for k, v in order}

    # --- Single plot with all height datasets and their linear fits ---
    fig1, ax1 = plt.subplots(figsize=(DIM_SCREEN[0],DIM_SCREEN[1]))
    for name, color in order:
        if name not in data or name not in fits:
            print(f"Warning: missing data or fit for {name}; skipping.")
            continue
        plot_dataset_on_ax(ax1, name, data[name], fits[name], color)
    ax1.set_xlabel(r"Meso roughness")
    ax1.set_ylabel("Maximum height")
    #ax1.set_title("Maximum height vs Furrow Filling")
    ax1.grid(alpha=0.25)
    # legend
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #ax1.legend(by_label.values(), by_label.keys(), loc="best")


    # --- Packing fraction plot ---
    fig2, ax2 = plt.subplots(figsize=(DIM_SCREEN[0],DIM_SCREEN[1]))
    plot_packing_fraction(ax2, pack_data, fits, colors, cst=CST)
    ax2.set_xlabel(r"Meso roughness")
    ax2.set_ylabel("Packing fraction")
    #ax2.set_title("Packing fraction vs Furrow Filling")
    ax2.grid(alpha=0.25)
    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))
    #ax2.legend(by_label2.values(), by_label2.keys(), loc="lower right")
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(SAVE_PLOT_FOLDER+"max_height.pdf")
    fig2.savefig(SAVE_PLOT_FOLDER+"packing_fraction.pdf")

if __name__ == "__main__":
    main()