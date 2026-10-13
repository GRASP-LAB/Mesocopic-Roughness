"""Shared Monte-Carlo excluded-volume engine (Eq. 18: V_ex ~= (N_C/N) L^3).

One fixed reference body sits at the origin with identity orientation.  A
second copy gets a uniform random translation in the cube [-L/2, L/2]^3 and a
uniform random rotation.  A shape-specific predicate decides overlap; the
estimator, rotation sampling, running checkpoints and binomial standard error
are identical for every shape.
"""
from __future__ import annotations
import numpy as np


def random_rotation_matrices(n, rng):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Rm = np.empty((n, 3, 3))
    Rm[:, 0, 0] = 1 - 2*(y*y + z*z); Rm[:, 0, 1] = 2*(x*y - z*w);     Rm[:, 0, 2] = 2*(x*z + y*w)
    Rm[:, 1, 0] = 2*(x*y + z*w);     Rm[:, 1, 1] = 1 - 2*(x*x + z*z); Rm[:, 1, 2] = 2*(y*z - x*w)
    Rm[:, 2, 0] = 2*(x*z - y*w);     Rm[:, 2, 1] = 2*(y*z + x*w);     Rm[:, 2, 2] = 1 - 2*(x*x + y*y)
    return Rm


def sphere_union_overlap(centers, R):
    """Overlap predicate for a union of equal spheres (radius R, given centres).

    Two copies overlap iff any centre of the moved copy is within 2R of any
    centre of the reference copy.
    """
    A = np.asarray(centers, dtype=float)

    def overlap(rot, trans):
        B = np.einsum('nij,kj->nki', rot, A) + trans[:, None, :]
        hit = np.zeros(len(rot), dtype=bool)
        for a in A:
            diff = B - a[None, None, :]
            hit |= np.any(np.einsum('nkj,nkj->nk', diff, diff) <= (2 * R) ** 2, axis=1)
        return hit

    return overlap


def excluded_volume_mc(overlap_fn, L, Ns, seed, batch_size=10_000):
    """Estimate V_ex at every sample count in Ns (one running pass).

    overlap_fn(rot (m,3,3), trans (m,3)) -> bool array (m,).
    Returns (Ns, V_ex, standard_error, total_hits); V_ex and the binomial
    standard error are in absolute units (not normalised by particle volume).
    """
    Ns = np.unique(np.asarray(Ns, dtype=np.int64))
    Nmax = int(Ns[-1])
    rng = np.random.default_rng(seed)
    hits_at = np.zeros(Ns.size, dtype=np.int64)
    hits_total = done = j = 0
    while done < Nmax:
        m = min(batch_size, Nmax - done)
        trans = rng.uniform(-L / 2, L / 2, size=(m, 3))
        rot = random_rotation_matrices(m, rng)
        cs = np.cumsum(overlap_fn(rot, trans), dtype=np.int64)
        done += m
        while j < Ns.size and Ns[j] <= done:
            k = int(Ns[j]) - (done - m)
            hits_at[j] = hits_total + cs[k - 1]
            j += 1
        hits_total += int(cs[-1])
    p = hits_at / Ns
    return Ns, p * L**3, np.sqrt(np.maximum(p * (1 - p), 0) / Ns) * L**3, hits_total
