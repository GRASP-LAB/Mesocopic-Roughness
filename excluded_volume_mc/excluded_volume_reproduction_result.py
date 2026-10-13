#!/usr/bin/env python3
"""Monte-Carlo excluded-volume estimator and convergence study.

Implements Eq. (18), V_ex ~= (N_C/N)L^3, for a fixed reference body and
random translations/orientations of a second copy.  All reported estimates
and uncertainties are normalized by the particle volume V, i.e. V_ex/V.
The tetrahedron test is an exact convex-polyhedron separating-axis test
(not a mesh approximation).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from excluded_volume_core import excluded_volume_mc, sphere_union_overlap


def tetrahedron(edge=1.0):
    a = float(edge)
    # centered regular tetrahedron
    v = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], dtype=float) * a/(2*np.sqrt(2))
    faces = np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]])
    # outward normals, oriented away from the centroid
    normals=[]
    for f in faces:
        n=np.cross(v[f[1]]-v[f[0]], v[f[2]]-v[f[0]])
        if np.dot(n, v[f].mean(axis=0)) < 0: n=-n
        normals.append(n/np.linalg.norm(n))
    normals=np.array(normals)
    edges=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    edgevec=v[edges[:,1]]-v[edges[:,0]]
    return v, normals, edges, edgevec


def tetra_exact(edge=1.0):
    a=float(edge)
    V=a**3/(6*np.sqrt(2))
    S=np.sqrt(3)*a**2
    # Mean radius R=M/(4 pi), M=sum(edge length * exterior dihedral angle).
    exterior=np.arccos(-1/3)
    R=3*a*exterior/(4*np.pi)
    return V, S, R, 2*V+2*S*R


def estimate_sphere_running(Ns, radius=1.0, seed=20260930):
    L = 4 * radius
    V_particle = (4.0 / 3.0) * np.pi * radius**3
    overlap = sphere_union_overlap(np.zeros((1, 3)), radius)
    _, v, se, hits = excluded_volume_mc(overlap, L, Ns, seed)
    return v / V_particle, se / V_particle, hits, L, V_particle


def tetra_overlap(edge=1.0):
    """Exact separating-axis overlap test (face normals + edge cross products)."""
    ref, fn, edges, ev = tetrahedron(edge)

    def overlap(rot, t):
        m = len(rot)
        rv = np.einsum('mij,vj->mvi', rot, ref) + t[:, None, :]
        rfn = np.einsum('mij,vj->mvi', rot, fn)
        rev = np.einsum('mij,kj->mki', rot, ev)
        cr = np.cross(ev[None, :, None, :], rev[:, None, :, :]).reshape(m, -1, 3)
        norm = np.linalg.norm(cr, axis=2)
        cr = cr / np.where(norm[..., None] > 1e-12, norm[..., None], 1)
        axes = np.concatenate([np.broadcast_to(fn, (m, 4, 3)), rfn, cr], axis=1)
        pr = np.einsum('mak,vk->mav', axes, ref)
        ps = np.einsum('mak,mvk->mav', axes, rv)
        return np.all((pr.max(2) >= ps.min(2)) & (ps.max(2) >= pr.min(2)), axis=1)

    return overlap


def estimate_tetra_running(Ns, edge=1.0, seed=20260931, batch=10000):
    L = 4 * edge * np.sqrt(6) / 4
    V_particle = edge**3 / (6 * np.sqrt(2))
    _, v, se, hits = excluded_volume_mc(tetra_overlap(edge), L, Ns, seed, batch)
    return v / V_particle, se / V_particle, hits, L, V_particle



def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir', default='.')
    ap.add_argument('--seed', type=int, default=20260930)
    ap.add_argument('--max-n', type=int, default=10_000_000)
    args=ap.parse_args()
    # More points at high N while retaining the requested logarithmic range.
    Ns=np.unique(np.r_[np.geomspace(100,args.max_n,24).astype(int), [100,1000,10000,100000,args.max_n]])
    Ns.sort()
    s_est,s_se,s_hits,Ls,s_particle_v=estimate_sphere_running(Ns,seed=args.seed)
    t_est,t_se,t_hits,Lt,t_particle_v=estimate_tetra_running(Ns,seed=args.seed+1)
    # Exact dimensionless benchmark ratios: (V_ex/V).
    s_exact=8.0
    V,S,R,t_excluded=tetra_exact(1.0)
    t_exact=t_excluded/V
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),sharey=False)
    for ax,est,se,exact,title,color in [(axs[0],s_est,s_se,s_exact,'Sphere, r = 1','tab:blue'),(axs[1],t_est,t_se,t_exact,'Regular tetrahedron, edge = 1','tab:orange')]:
        ax.plot(Ns,est,color=color,lw=2,label='MC estimate')
        ax.fill_between(Ns,est-se,est+se,color=color,alpha=.22,label='± 1 standard error')
        ax.axhline(exact,color='black',ls='--',lw=1.4,label=f'exact = {exact:.5f}')
        ax.set_xscale('log'); ax.set_xlabel('Number of samples, N'); ax.set_ylabel(r'$V_{ex}/V$')
        ax.set_title(title); ax.grid(True,which='both',alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('Monte Carlo excluded-volume convergence (Eq. 18)')
    fig.tight_layout()
    fig.savefig(out/'excluded_volume_convergence.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    sphere_rel=abs(s_est[-1]-s_exact)/s_exact
    tetra_rel=abs(t_est[-1]-t_exact)/t_exact
    print(f'SPHERE V_ex/V reference={s_exact:.10f} final={s_est[-1]:.10f} relative_error={sphere_rel:.6%} hits={s_hits} L={Ls}')
    print(f'TETRA V_ex/V reference={t_exact:.10f} final={t_est[-1]:.10f} relative_error={tetra_rel:.6%} hits={t_hits} L={Lt}')
    print(f'PARTICLE VOLUMES sphere={s_particle_v:.10f} tetrahedron={t_particle_v:.10f}')
    print(f'TETRA geometry V={V:.10f} S={S:.10f} R={R:.10f}')
    print('N values:',Ns.tolist())

if __name__=='__main__': main()
