import math
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit, minimize  # type: ignore

# To reproduce article plot, play with norm_factor term in the plot_packing_series function.

BASE_FOLDER = ""
SAVE_PLOT_FOLDER = BASE_FOLDER + ""
DATA_FILE = SAVE_PLOT_FOLDER + "maximum_height_mean_std_n_spheres_per_rod.txt"
PACK_FILE = SAVE_PLOT_FOLDER + "packing_fraction_mean_std_n_spheres_per_rod.txt"
MU_STAR = 2.0 - math.sqrt(2.0)

# N v_inf / (4 lx ly)
CST_ROD = (3780 * 0.00031390550156249993) / (1200 * 4 * 0.1 * 0.1 * 6 * 0.0025)
CST_CROSS = (2268 * 0.0005277972656249998) / (1200 * 4 * 0.1 * 0.1 * 6 * 0.0025)
CST_STAR = (1620 * 0.0007301127374999999) / (1200 * 4 * 0.1 * 0.1 * 6 * 0.0025)

print(f"CST_ROD = {CST_ROD:.6f}, CST_CROSS = {CST_CROSS:.6f}, CST_STAR = {CST_STAR:.6f}")

DIM_SCREEN = 4096 / 320, 1800 / 160
plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 42
plt.rcParams['axes.labelsize'] = 36
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['legend.fontsize'] = 20

# Plot order and styling, keyed by "<family><friction>" series name (e.g. "rod0.4")
SERIES_ORDER = [
    "rod0.0", "cross0.0", "star0.0",
    "rod0.4", "cross0.4", "star0.4",
    "rod1.0", "cross1.0", "star1.0",
]
COLORS = {"rod": "green", "cross": "purple", "star": "orange"}
MARKERS = {"0.0": "o", "0.4": "s", "1.0": "^"}
CST_BY_FAMILY = {"rod": CST_ROD, "cross": CST_CROSS, "star": CST_STAR}
FRICTION_VALUES = {"0.0": 0.0, "0.4": 0.4, "1.0": 1.0}

# Empirical modulation of phi by microscopic friction: 1 + scale * mu_macro * exp(-mu_micro / scale)
MODULATION_SCALE = 0.31

# Choose independently whether to subtract and/or divide by the star reference.
SUBTRACT_REFERENCE = True
DIVIDE_REFERENCE = False

def split_series_name(name):
    """Split a series name like "cross0.4" into ("cross", "0.4")."""
    for family in ("rod", "cross", "star"):
        if name.startswith(family):
            return family, name[len(family):]
    return name, ""


def n_to_mu_meso(n_values):
    """Convert sphere-per-rod counts N into meso roughness µ_meso = 1 - furrow_parameter."""
    mu_meso = []
    for n in n_values:
        if not isinstance(n, (int, float)) or not math.isfinite(n) or n <= 1:
            mu_meso.append(float("nan"))
            continue
        inside = max(0.0, 1.0 - (2.0 / (n - 1.0)) ** 2)
        furrow_parameter = math.sqrt(inside)
        mu_meso.append(1.0 - furrow_parameter)
    return mu_meso


def parse_series_file(filename):
    """
    Parse a file structured as repeated blocks:
        # <type>
        x mean std
        ...
    Returns dict: {type: {"x": [...], "y": [...], "yerr": [...]}}
    """
    data = {}
    current_type = None
    with open(filename, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line[1:].strip().split()
                if parts:
                    current_type = parts[0].lower()
                    data.setdefault(current_type, {"x": [], "y": [], "yerr": []})
                continue
            if current_type is None:
                continue
            cleaned = line.replace(";", " ").replace(",", " ").replace("\t", " ")
            parts = cleaned.split()
            if len(parts) < 3:
                continue
            try:
                x, mean, std = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
            data[current_type]["x"].append(x)
            data[current_type]["y"].append(mean)
            data[current_type]["yerr"].append(std)
    return data


def piecewise_height_model(mu_meso, H1, HDelta, mu_star=MU_STAR):
    """Piecewise linear-constant height model with fixed transition roughness."""
    mu = np.asarray(mu_meso, dtype=float)
    linear_branch = (H1 + HDelta) - (HDelta / mu_star) * mu
    return np.where(mu <= mu_star, linear_branch, H1)


def packing_model(mu_meso, H1, HDelta, cst, mu_star=MU_STAR):
    """phi(mu_meso) = cst * (alpha^2 + 3) / H(mu_meso), with alpha = 1 - mu_meso."""
    mu_meso = np.asarray(mu_meso, dtype=float)
    alpha = 1.0 - mu_meso
    denom = piecewise_height_model(mu_meso, H1, HDelta, mu_star=mu_star)

    out = np.full(mu_meso.shape, np.nan, dtype=float)
    safe = denom != 0.0
    out[safe] = cst * ((alpha[safe] ** 2 + 3.0) / denom[safe])
    return out


def macro_roughness_of(fit):
    """mu_macro = 1 - HDelta / H1 for a given fit."""
    return 1.0 - fit["HDelta"] / fit["H1"]


def phi_0_of(fit, cst):
    """phi_0 = cst * 4 / (H1 + HDelta), i.e. packing_model at alpha = 1 (mu_meso = 0)."""
    return cst * 4.0 / (fit["H1"] + fit["HDelta"])


def packing_modulation_factor(mu_micro, scale_a=MODULATION_SCALE, scale_b=MODULATION_SCALE):
    """1 + scale_a * exp(-mu_micro / scale_b)."""
    return 1.0 + scale_a * math.exp(-mu_micro / scale_b)


def fit_modulation_scale(
    family, fits, pack_data, cst, phi_0, n_grid=200,
    bounds=((-5.0, 5.0), (0.02, 5.0)),
):
    """Find the per-family (scale_a, scale_b) that best collapse the friction-level
    packing curves (0.0, 0.4, 1.0) onto a single master curve, by minimizing the
    variance across friction levels of the normalized model curve over a common
    mu_meso grid."""
    level_info = []
    x_min, x_max = np.inf, -np.inf
    for level, mu_micro in FRICTION_VALUES.items():
        fit = fits.get(f"{family}{level}")
        series = pack_data.get(f"{family}{level}")
        if fit is None or series is None:
            continue
        x = np.asarray(series["x"], dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        x_min = min(x_min, np.nanmin(x))
        x_max = max(x_max, np.nanmax(x))
        level_info.append((mu_micro, fit))

    fallback = (MODULATION_SCALE, MODULATION_SCALE)
    if len(level_info) < 2 or not np.isfinite(x_min) or not np.isfinite(x_max) or x_min >= x_max:
        return fallback

    x_grid = np.linspace(x_min, x_max, n_grid)

    def objective(params):
        scale_a, scale_b = params
        curves = []
        for mu_micro, fit in level_info:
            mu_macro = macro_roughness_of(fit)
            modulation = packing_modulation_factor(mu_micro, scale_a, scale_b)
            model = packing_model(x_grid, fit["H1"], fit["HDelta"], cst=cst)
            curves.append(model / (phi_0 * modulation))
        return float(np.nansum(np.nanvar(np.vstack(curves), axis=0)))

    result = minimize(
        objective, x0=[MODULATION_SCALE, MODULATION_SCALE],
        bounds=bounds, method="L-BFGS-B",
    )
    return (float(result.x[0]), float(result.x[1])) if result.success else fallback


def normalize_observable(values, reference):
    """Apply the selected subtraction and division to an observable."""
    subtract_value = reference if SUBTRACT_REFERENCE else 0.0
    divide_value = reference if DIVIDE_REFERENCE else 1.0
    return (np.asarray(values, dtype=float) - subtract_value) / divide_value


def normalize_error(values, reference):
    """Scale an observable's uncertainty consistently with its normalization."""
    divide_value = abs(reference) if DIVIDE_REFERENCE else 1.0
    return np.asarray(values, dtype=float) / divide_value


def fit_data(data):
    """
    Fit the piecewise height model to each type's (mu_meso, y) series.
    Returns {type: {"H1", "H1_error", "HDelta", "HDelta_error", "chi2", "dof"}}
    """
    fits = {}
    for t, series in data.items():
        x = np.asarray(series["x"], dtype=float)
        y = np.asarray(series["y"], dtype=float)
        yerr = np.asarray(series.get("yerr", np.ones_like(y)), dtype=float)

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0.0)
        x, y, yerr = x[valid], y[valid], yerr[valid]

        if len(y) < 2:
            print(f"Not enough valid points to fit {t}")
            continue

        H1_guess = max(float(np.nanmin(y)), 0.0)
        HDelta_guess = max(float(np.nanmax(y) - np.nanmin(y)), 1e-6)

        try:
            popt, pcov = curve_fit(
                piecewise_height_model, x, y, p0=[H1_guess, HDelta_guess],
                sigma=yerr, absolute_sigma=True,
                bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=10000,
            )
        except RuntimeError:
            print(f"Fit failed for {t}")
            continue

        H1, HDelta = popt
        H1_err, HDelta_err = np.sqrt(np.diag(pcov))
        residuals = (y - piecewise_height_model(x, *popt)) / yerr
        chi2 = np.sum(residuals ** 2)
        dof = max(len(y) - len(popt), 1)

        fits[t] = {
            "H1": H1, "H1_error": H1_err,
            "HDelta": HDelta, "HDelta_error": HDelta_err,
            "chi2": chi2, "dof": dof,
        }
        print(
            f"Fit for {t}: H1={H1:.4f} ± {H1_err:.4f}, "
            f"HDelta={HDelta:.4f} ± {HDelta_err:.4f}, chi²/dof={chi2 / dof:.2f}"
        )
    return fits


def plot_height_series(
    ax, name, series, fit_params, color, marker, mu_micro, type,
    reference_fit,
):
    x = np.asarray(series["x"], dtype=float)
    y = np.asarray(series["y"], dtype=float)
    yerr = np.asarray(series.get("yerr", np.zeros_like(y)), dtype=float)

    reference = reference_fit["H1"] + reference_fit["HDelta"]
    y = normalize_observable(y, reference)
    yerr = normalize_error(yerr, reference)

    ax.errorbar(
        x, y, yerr=yerr, fmt=marker, linestyle='none', label="_nolegend_",
        markersize=10, markeredgewidth=1.5, markerfacecolor=color,
        markeredgecolor='black', elinewidth=1.5, ecolor='black',
        capsize=3, capthick=1.5, zorder=3,
    )

    H1, H1_err = fit_params["H1"], fit_params["H1_error"]
    HDelta, HDelta_err = fit_params["HDelta"], fit_params["HDelta_error"]

    x_valid = x[~np.isnan(x)]
    x_line = np.linspace(np.nanmin(x_valid), np.nanmax(x_valid), 200)
    height_model = piecewise_height_model(x_line, H1, HDelta)
    height_model = normalize_observable(height_model, reference)
    ax.plot(
        x_line, height_model,
        '-', color=color,
    )

    height_plus = piecewise_height_model(x_line, H1 + H1_err, HDelta + HDelta_err)
    height_minus = piecewise_height_model(
        x_line, max(H1 - H1_err, 0.0), max(HDelta - HDelta_err, 0.0),
    )
    height_plus = normalize_observable(height_plus, reference)
    height_minus = normalize_observable(height_minus, reference)
    ax.fill_between(x_line, height_minus, height_plus, color=color, alpha=0.07, linewidth=0)
    ax.set_xscale('log')

def plot_height_at_zero_inset(ax, fits):
    """Show fitted H_max at mu_meso=0 versus microscopic friction."""
    inset = ax.inset_axes([0.05, 0.25, 0.35, 0.35])
    inset_markers = {"0.0": "p", "0.4": "s", "1.0": "^"}

    for family, color in COLORS.items():
        family_labeled = False
        for level, mu_micro in FRICTION_VALUES.items():
            fit = fits.get(f"{family}{level}")
            if fit is None:
                continue

            height = fit["H1"] + fit["HDelta"]
            error = math.hypot(fit["H1_error"], fit["HDelta_error"])
            inset.errorbar(
                [mu_micro], [height], yerr=[error],
                color=color, marker=inset_markers[level], linewidth=1.5,
                capsize=2,
                label=family.capitalize() if not family_labeled else "_nolegend_",
            )
            family_labeled = True

    inset.set_title(r"$H_{\max}(0)$", fontsize=16)
    inset.set_xlabel(r"$\mu_{\mathrm{micro}}$", fontsize=13)
    inset.tick_params(axis="both", labelsize=12)
    inset.set_xticks(list(FRICTION_VALUES.values()))

def plot_packing_series(
    ax, name, series, fit_params, color, marker, cst, mu_micro, type,
    phi_0, scale_a=MODULATION_SCALE, scale_b=MODULATION_SCALE,
):
    """Collapse phi onto a single master curve by dividing out phi_0 and the
    friction modulation 1 + scale_a * mu_macro * exp(-mu_micro / scale_b)."""
    x = np.asarray(series["x"], dtype=float)
    y = np.asarray(series["y"], dtype=float)
    yerr = np.asarray(series.get("yerr", np.zeros_like(y)), dtype=float)

    mu_macro = macro_roughness_of(fit_params) if fit_params is not None else 0.0
    #Function to give Figure 7
    def figure7():
        return 1
    #Function to give Figure 8
    def figure8():
        if "rod" in name:
            normalization = packing_modulation_factor(mu_micro, scale_a, scale_b)
        elif "cross" in name:
            normalization = packing_modulation_factor(mu_micro, scale_a, scale_b)
        elif "star" in name:
            normalization = packing_modulation_factor(mu_micro, scale_a, scale_b)
        else:
            print("Unknown particle type: {}".format(name))
        return normalization
    #discussed with reviewer but ommited
    def f_hard_core_excluded_volume():
        if "rod" in name:
            normalization = packing_modulation_factor(mu_micro, scale_a, scale_b)*5.0
        elif "cross" in name:
            normalization = packing_modulation_factor(mu_micro, scale_a, scale_b)*4.4346
        elif "star" in name:
            normalization = packing_modulation_factor(mu_micro, scale_a, scale_b)*4.1283
        else:
            print("Unknown particle type: {}".format(name))
        return normalization
    #discussed with reviewer but ommited
    def f_mc_excluded_volume():
        if "rod" in name:
            normalization = packing_modulation_factor(mu_micro, scale_a, scale_b)*11.1594
        elif "cross" in name:
            normalization = packing_modulation_factor(mu_micro, scale_a, scale_b)*12.5048
        elif "star" in name:
            normalization = packing_modulation_factor(mu_micro, scale_a, scale_b)*12.8592
        else:
            print("Unknown particle type: {}".format(name))
        return normalization

    # change this by either figure7, figure8, f_hard_core_excluded_volume or f_mc_excluded_volume to normalize the data accordingly
    norm_factor = f_hard_core_excluded_volume()
    y = y / norm_factor
    yerr = yerr / abs(norm_factor)

    ax.errorbar(
        x, y, yerr=yerr, fmt=marker, linestyle='none', label="_nolegend_",
        markersize=10, markeredgewidth=1.5, markerfacecolor=color,
        markeredgecolor='black', elinewidth=1.5, ecolor='black',
        capsize=3, capthick=1.5, zorder=3,
    )

    if fit_params is None:
        return

    H1, H1_err = fit_params["H1"], fit_params["H1_error"]
    HDelta, HDelta_err = fit_params["HDelta"], fit_params["HDelta_error"]

    x_valid = x[~np.isnan(x)]
    x_line = np.linspace(np.nanmin(x_valid), np.nanmax(x_valid), 300) if x_valid.size else np.linspace(0.0, 1.0, 300)

    packing_nominal = packing_model(x_line, H1, HDelta, cst=cst)
    y_nom = packing_nominal / norm_factor
    ax.plot(x_line, y_nom, '-', color=color, zorder=2)

    combos = [
        (H1 + H1_err, HDelta + HDelta_err),
        (H1 + H1_err, max(HDelta - HDelta_err, 0.0)),
        (max(H1 - H1_err, 0.0), HDelta + HDelta_err),
        (max(H1 - H1_err, 0.0), max(HDelta - HDelta_err, 0.0)),
    ]
    y_combo = np.vstack([
        packing_model(x_line, H1p, HDeltap, cst=cst) / norm_factor
        for H1p, HDeltap in combos
    ])
    y_min = np.nanmin(y_combo, axis=0)
    y_max = np.nanmax(y_combo, axis=0)
    valid = ~np.isnan(y_min) & ~np.isnan(y_max)
    if np.any(valid):
        ax.fill_between(x_line[valid], y_min[valid], y_max[valid], color=color, alpha=0.15, linewidth=0)


def add_semantic_legends(ax):
    """Add separate side-by-side legends for type, friction, and model."""
    type_legend = ax.legend(
        handles=[
            Line2D([], [], color=COLORS["rod"], marker="s", linestyle="none", label="Rod"),
            Line2D([], [], color=COLORS["cross"], marker="s", linestyle="none", label="Cross"),
            Line2D([], [], color=COLORS["star"], marker="s", linestyle="none", label="Star"),
        ],
        title="Type", loc="lower left", bbox_to_anchor=(0.00, -0.01),
        fontsize=18, title_fontsize=20, frameon=False,
    )
    ax.add_artist(type_legend)

    friction_legend = ax.legend(
        handles=[
            Line2D([], [], color="black", marker=MARKERS["0.0"], linestyle="none", label="0.0"),
            Line2D([], [], color="black", marker=MARKERS["0.4"], linestyle="none", label="0.4"),
            Line2D([], [], color="black", marker=MARKERS["1.0"], linestyle="none", label="1.0"),
        ],
        title="Friction", loc="lower left", bbox_to_anchor=(0.16, -0.01),
        fontsize=18, title_fontsize=20, frameon=False,
    )
    ax.add_artist(friction_legend)

    ax.legend(
        handles=[Line2D([], [], color="black", linestyle="-", label="")],
        title="Model", loc="lower left", bbox_to_anchor=(0.32, -0.01),
        fontsize=18, frameon=False,
    )


def plot_macro_roughness(ax, fits):
    """Plot fitted macro roughness against microscopic friction."""
    for family, color in COLORS.items():
        x_values = []
        macro_values = []
        macro_errors = []
        for level, friction in FRICTION_VALUES.items():
            fit = fits.get(f"{family}{level}")
            if fit is None:
                continue
            H1 = fit["H1"]
            HDelta = fit["HDelta"]
            macro_value = 1.0 - HDelta / H1
            macro_error = math.sqrt(
                (fit["HDelta_error"] / H1) ** 2
                + (HDelta * fit["H1_error"] / H1 ** 2) ** 2
            )
            x_values.append(friction)
            macro_values.append(macro_value)
            macro_errors.append(macro_error)

        if x_values:
            ax.errorbar(
                x_values, macro_values, yerr=macro_errors,
                fmt="o", color=color, label=family.capitalize(),
                markersize=9, capsize=4, linewidth=2,
            )


def main():
    height_data = parse_series_file(DATA_FILE)
    pack_data = parse_series_file(PACK_FILE)

    for series in list(height_data.values()) + list(pack_data.values()):
        series["x"] = n_to_mu_meso(series["x"])

    fits = fit_data(height_data)

    print("phi_0 for friction=1.0:")
    phi_0_by_family = {}
    for family in ("rod", "cross", "star"):
        fit = fits.get(f"{family}1.0")
        if fit is None:
            print(f"  {family}: missing fit")
            continue
        phi_0 = phi_0_of(fit, CST_BY_FAMILY[family])
        H1_plus_HDelta_err = math.sqrt(fit["H1_error"] ** 2 + fit["HDelta_error"] ** 2)
        phi_0_error = abs(phi_0) * H1_plus_HDelta_err / (fit["H1"] + fit["HDelta"])
        phi_0_by_family[family] = phi_0
        print(f"  {family}: phi_0 = {phi_0:.6f} +/- {phi_0_error:.6f}")

    print("Fitted modulation scale per family:")
    modulation_scale_by_family = {}
    for family in ("rod", "cross", "star"):
        if family not in phi_0_by_family:
            modulation_scale_by_family[family] = (MODULATION_SCALE, MODULATION_SCALE)
            print(f"  {family}: missing phi_0, using default scale_a=scale_b={MODULATION_SCALE}")
            continue
        scale_a, scale_b = fit_modulation_scale(
            family, fits, pack_data, CST_BY_FAMILY[family], phi_0_by_family[family],
        )
        modulation_scale_by_family[family] = (scale_a, scale_b)
        print(f"  {family}: scale_a = {scale_a:.4f}, scale_b = {scale_b:.4f}")

    print("Macro roughness for all fitted series:")
    for name in SERIES_ORDER:
        fit = fits.get(name)
        if fit is None:
            print(f"  {name}: missing fit")
            continue
        H1 = fit["H1"]
        HDelta = fit["HDelta"]
        macro_roughness = 1.0 - HDelta / H1
        macro_roughness_error = math.sqrt(
            (fit["HDelta_error"] / H1) ** 2
            + (HDelta * fit["H1_error"] / H1 ** 2) ** 2
        )
        print(
            f"  {name}: mu_macro = {macro_roughness:.6f} "
            f"+/- {macro_roughness_error:.6f}"
        )

    fig1, ax1 = plt.subplots(figsize=DIM_SCREEN)
    for name in SERIES_ORDER:
        if name not in height_data or name not in fits:
            print(f"Warning: missing data or fit for {name}; skipping.")
            continue
        family, level = split_series_name(name)
        plot_height_series(
            ax1, name, height_data[name], fits[name], COLORS[family], MARKERS[level],
            mu_micro=level, type=family, reference_fit=fits[f"{family}{level}"],
        )
    ax1.set_ylabel(r"$H_{\max}(0) - H_{\max}(\mu_{\mathrm{meso}})$")
    plot_height_at_zero_inset(ax1, fits)
    ax1.set_xlabel(r"Meso roughness")
    ax1.set_ylabel(r"$(H_{\max} - H_{\max,\mathrm{ref}}) / H_{\max,\mathrm{ref}}$")
    ax1.set_xscale('log')

    fig2, ax2 = plt.subplots(figsize=DIM_SCREEN)
    for name in SERIES_ORDER:
        if name not in pack_data:
            print(f"Warning: missing packing data for {name}; skipping.")
            continue
        family, level = split_series_name(name)
        scale_a, scale_b = modulation_scale_by_family[family]
        plot_packing_series(
            ax2, name, pack_data[name], fits.get(name),
            COLORS[family], MARKERS[level], cst=CST_BY_FAMILY[family],
            mu_micro=FRICTION_VALUES[level], type=family,
            phi_0=phi_0_by_family[family],
            scale_a=scale_a, scale_b=scale_b,
        )
    ax2.set_xlabel(r"Meso roughness")
    ax2.set_ylabel(
        r"$\phi \, / \, \left[\phi_0 "
        "(1+a" 
        r"e^{-\mu_{\mathrm{micro}}/b})\right]$"
    )
    ax2.grid(alpha=0.25)
    ax2.set_xscale('log')

    add_semantic_legends(ax1)
    add_semantic_legends(ax2)

    fig3, ax3 = plt.subplots(figsize=DIM_SCREEN)
    plot_macro_roughness(ax3, fits)
    ax3.set_xlabel(r"Microscopic friction $\mu_{\mathrm{micro}}$")
    ax3.set_ylabel(r"Macro roughness $\mu_{\mathrm{macro}}$")
    ax3.set_xticks([0.0, 0.4, 1.0])
    ax3.grid(alpha=0.25)
    ax3.legend(loc="best", fontsize=20)

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    end = "pdf"
    fig1.savefig(SAVE_PLOT_FOLDER + "max_height." + end)
    fig2.savefig(SAVE_PLOT_FOLDER + "packing_fraction." + end)
    fig3.savefig(SAVE_PLOT_FOLDER + "macro_roughness." + end)


if __name__ == "__main__":
    main()
