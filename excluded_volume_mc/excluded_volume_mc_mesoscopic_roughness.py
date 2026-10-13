"""Mesoscopic-roughness Rod/Cross/Star particles (multi-sphere representation)
and their excluded volume, following Luyckx & Opsomer, "Impact of Particle
Roughness on Packing" (Granular Matter, in review2/excluded_mc).

Geometry (paper Section 2 / Appendix A)
----------------------------------------
Each shape is built from 1 (rod), 2 (cross) or 3 (star) mutually orthogonal
"arms" of total length L = 6R, each arm approximated by n overlapping spheres
of radius R with centers uniformly placed at

    r_i = -L/2 + R + i * (L - 2R)/(n - 1),   i = 0 .. n-1            (Eq. A1)

which for L = 6R reduces to r_i in [-2R, 2R] -- the same segment endpoints
used for the smooth (n -> inf) spherocylinder limit.  The furrow depth (the
gap left between two neighbouring spheres, measured from the arm axis) is

    f_n = R * sqrt(1 - (2/(n-1))**2)                                  (Eq. 1)

and the dimensionless mesoscopic roughness is mu_meso = 1 - f_n/R  (Eq. 2),
0 for a smooth surface (n -> inf) and 1 for spheres just touching (n = 3
gives the deepest furrow explored in the paper).

Non-uniform spacing (Appendix B) lets mu_meso be dialled continuously between
the values reachable by consecutive integers n and n+1: one extra sphere is
inserted asymmetrically inside one gap of the n-sphere base chain, at a
fractional position t in (0, 1) along that gap (t=0.5 reproduces the
symmetric insertion).  This file implements that same qualitative idea
(inserting one off-centre sphere per gap) with a single free fraction `t`,
because the closed-form offset in Appendix B (Eqs. B13/B15) is degenerate to
reproduce bit-exactly from the manuscript; the implementation below is exact
in its own (equivalent, length-conserving) local two-sphere geometry and
spans the same interval [mu_meso(n+1), mu_meso(n)].

Excluded volume
----------------
Because every shape here is an exact union of spheres of a single common
radius R, the excluded-volume Monte Carlo test of Jia, Xu & Zhu (Eq. 18)
collapses to a simple sphere-sphere overlap test: two configurations overlap
iff any sphere centre of the (randomly placed & oriented) second particle
lies within 2R of any sphere centre of the reference particle.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from excluded_volume_core import excluded_volume_mc, sphere_union_overlap

ARM_AXES = {
    'rod':   np.eye(3)[:1],
    'cross': np.eye(3)[:2],
    'star':  np.eye(3)[:3],
}


# ============================================================
# MESOSCOPIC ROUGHNESS GEOMETRY
# ============================================================

def mu_meso_uniform(n):
    """Dimensionless mesoscopic roughness of an n-sphere uniform arm (Eq. 2)."""
    return 1.0 - np.sqrt(1.0 - (2.0 / (n - 1)) ** 2)


def arm_offsets_uniform(n, R=1.0, L=None):
    """1D sphere-centre offsets along one arm axis, n spheres, uniform (Eq. A1)."""
    if L is None:
        L = 6 * R
    return np.linspace(-(L - 2 * R) / 2, (L - 2 * R) / 2, n)


def arm_offsets_nonuniform(n, t, gap_index=None, R=1.0, L=None):
    """1D sphere-centre offsets with one gap of the n-sphere base chain split
    asymmetrically by inserting an extra sphere at fraction t (in (0, 1)) of
    that gap's width (Appendix B idea; t=0.5 is the symmetric insertion,
    reproducing the (n+1)-sphere case only when the base chain is n=... ).
    Returns n+1 offsets and the achieved mu_meso, averaged furrow depth over
    all n gaps (n-1 base gaps, one of which is now split in two).
    """
    if L is None:
        L = 6 * R
    if not (0.0 < t < 1.0):
        raise ValueError("t must lie strictly between 0 and 1")
    base = arm_offsets_uniform(n, R=R, L=L)
    if gap_index is None:
        gap_index = (n - 2) // 2   # central gap, by default
    left, right = base[gap_index], base[gap_index + 1]
    new_sphere = left + t * (right - left)
    offsets = np.insert(base, gap_index + 1, new_sphere)

    gaps = np.diff(offsets)
    furrows = np.where(gaps <= 2 * R, np.sqrt(np.maximum(R**2 - (gaps / 2) ** 2, 0.0)), R)
    mu_eff = 1.0 - furrows.mean() / R
    return offsets, mu_eff


# ============================================================
# MULTI-SPHERE SHAPE
# ============================================================

class MultiSphereShape:
    """Union of spheres of common radius R describing a Rod/Cross/Star with
    a given mesoscopic roughness (uniform n-sphere arms, or one non-uniform
    arm insertion via `nonuniform_t`).
    """
    def __init__(self, kind, n, R=1.0, L=None, nonuniform_t=None):
        if kind not in ARM_AXES:
            raise ValueError("kind must be 'rod', 'cross', or 'star'")
        self.kind, self.n, self.R = kind, n, R
        self.L = L if L is not None else 6 * R

        if nonuniform_t is None:
            offsets = arm_offsets_uniform(n, R=R, L=self.L)
            self.mu_meso = mu_meso_uniform(n)
        else:
            offsets, self.mu_meso = arm_offsets_nonuniform(n, nonuniform_t, R=R, L=self.L)

        centers = []
        for axis in ARM_AXES[kind]:
            centers.append(np.outer(offsets, axis))
        centers = np.concatenate(centers, axis=0)
        # De-duplicate the shared centre sphere (present once per arm at offset 0).
        centers = np.unique(np.round(centers, 9), axis=0)
        self.centers = centers            # (K, 3)

    def distance_to_shape(self, points):
        """Signed distance of points (N,3) to the sphere union's surface."""
        d = np.linalg.norm(points[:, None, :] - self.centers[None, :, :], axis=2)
        return d.min(axis=1) - self.R

    def bounding_box(self, margin):
        extent = np.abs(self.centers).max() + self.R + margin
        return np.array([-extent] * 3), np.array([extent] * 3)


# ============================================================
# SOLID VOLUME (VOXEL GRID)
# ============================================================

def compute_particle_volume(shape, voxels_per_R=30, max_grid_res=200, verbose=False):
    R = shape.R
    mn, mx = shape.bounding_box(0.5 * R)
    total_extent = mx[0] - mn[0]
    voxel_size = R / voxels_per_R
    grid_res = int(np.clip(np.ceil(total_extent / voxel_size), 10, max_grid_res))

    xs = np.linspace(mn[0], mx[0], grid_res)
    ys = np.linspace(mn[1], mx[1], grid_res)
    zs = np.linspace(mn[2], mx[2], grid_res)
    voxel_vol = (xs[1] - xs[0]) * (ys[1] - ys[0]) * (zs[1] - zs[0])

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    inside = shape.distance_to_shape(pts) <= 0
    volume = inside.sum() * voxel_vol

    if verbose:
        print(f"  [{shape.kind}, n={shape.n}] grid_res={grid_res}, V={volume:.6f}")
    return volume


# ============================================================
# MONTE CARLO EXCLUDED VOLUME (paper Eq. 18, sphere-union overlap test)
# Shared engine: excluded_volume_core.py (same as the reproduction script)
# ============================================================

def excluded_volume_monte_carlo(shape, N_total, L=None, batch_size=50_000,
                                 seed=12345, checkpoints=30, verbose=True):
    """Returns (V_ex, history, L); history = [(N, V_ex(N), standard_error(N)), ...]
    (absolute units, binomial standard error as in the reproduction script)."""
    R = shape.R
    if L is None:
        extent = np.abs(shape.centers).max() + R
        L = 8 * (2 * extent)
    Ns = np.unique(np.logspace(np.log10(max(batch_size, 100)), np.log10(N_total),
                               checkpoints).astype(np.int64))
    Ns = np.unique(np.r_[Ns, N_total])
    Ns, v, se, _ = excluded_volume_mc(sphere_union_overlap(shape.centers, R),
                                      L, Ns, seed, batch_size)
    history = list(zip(Ns.tolist(), v.tolist(), se.tolist()))
    if verbose:
        for n_done, est, err in history:
            print(f"  [{shape.kind}, n={shape.n}] N={n_done:>9d}  V_ex~{est:.4f} +/- {err:.4f}")
    return float(v[-1]), history, L


# ============================================================
# MAIN / DEMO
# ============================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default='.')
    ap.add_argument('--n-total', type=int, default=100_000_000)
    args = ap.parse_args()

    R = 1.0
    N_TOTAL = args.n_total
    ns = [12]   # mesoscopic roughness levels (highest to have smooth version taken into account)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Mesoscopic-roughness Rod / Cross / Star: multi-sphere excluded volume")
    print("=" * 78)

    results = []
    for kind in ['rod', 'cross', 'star']:
        if kind == 'rod':
            continue
        for n in ns:
            print(f"Processing {kind} with n={n}")
            shape = MultiSphereShape(kind, n, R=R)
            V = compute_particle_volume(shape, voxels_per_R=30)
            seed = 1000 * ['rod', 'cross', 'star'].index(kind) + n
            V_ex, hist, L = excluded_volume_monte_carlo(shape, N_TOTAL, seed=seed, verbose=False)
            Ns_h, v_h, se_h = (np.array(c) for c in zip(*hist))
            results.append((kind, n, shape.mu_meso, V, Ns_h, v_h / V, se_h / V))
            print(f"{kind:6s} n={n:3d}  mu_meso={shape.mu_meso:.4f}  V={V:.4f}  "
                  f"V_ex={V_ex:.4f} +/- {se_h[-1]:.4f}  "
                  f"V_rex={V_ex/V:.4f} +/- {se_h[-1]/V:.4f}  L={L}")

    # Convergence of V_rex = V_ex/V with N, +/- 1 standard error (as in the paper reproduction).
    fig, axs = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5), squeeze=False)
    for ax, (kind, n, mu, V, Ns_h, est, se) in zip(axs[0], results):
        ax.plot(Ns_h, est, lw=2, label='MC estimate')
        ax.fill_between(Ns_h, est - se, est + se, alpha=.22, label='± 1 standard error')
        ax.axhline(est[-1], color='black', ls='--', lw=1.2,
                   label=f'final = {est[-1]:.4f} ± {se[-1]:.4f}')
        ax.set_xscale('log'); ax.set_xlabel('Number of samples, N')
        ax.set_ylabel(r'$V_{ex}/V$'); ax.set_title(f'{kind}, n = {n}')
        ax.grid(True, which='both', alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('Monte Carlo excluded-volume convergence (Eq. 18)')
    fig.tight_layout()
    fig.savefig(out / 'excluded_volume_convergence_roughness.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
