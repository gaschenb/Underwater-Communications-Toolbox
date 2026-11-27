"""Vectorised Monte Carlo photon transport between a beam transmitter and a disk receiver.

Variance reduction:
  * ballistic light is computed exactly: unscattered rays are traced to the receiver plane
    (geometry) and weighted by e^{-cL};
  * forced first collision: scattered photons are made to interact before the receiver plane,
    with weight (1 - e^{-c s_max});
  * semi-analytic local estimate: at every scattering event the probability of scattering
    straight into the aperture (within the field of view) and arriving unattenuated is added:
        Δw = w · ω0 · β̃(θ_s) · A cos θ_r / r² · e^{-c r}
    with ω0 = b/c the single-scattering albedo. This gives smooth impulse responses with
    10^4–10^5 photons even for small apertures.
Photons are terminated by Russian roulette (w < 1e-4) or after `max_orders` scatterings.
Delays are excess delays relative to the line-of-sight time L/v.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import V_WATER, trapz


@dataclass
class MCResult:
    L: float
    n_photons: int
    ballistic: float                 # fraction of transmitted power received unscattered
    scattered: float                 # fraction received after >= 1 scattering
    by_order: np.ndarray             # scattered fraction per order 1..max_orders
    t_edges: np.ndarray              # excess-delay bin edges (s)
    h_scat: np.ndarray               # scattered power per bin (fraction of Tx power)
    geom: float                      # geometric capture of unscattered rays
    paths: list = field(default_factory=list)   # sample trajectories (k × 3 arrays)
    angles: np.ndarray = None        # arrival angles (rad) of scattered contributions
    angle_w: np.ndarray = None

    @property
    def total(self):
        return self.ballistic + self.scattered

    def delay_stats(self):
        """Mean and rms excess delay of the received power (ballistic at 0)."""
        tc = 0.5 * (self.t_edges[1:] + self.t_edges[:-1])
        w = np.concatenate([[self.ballistic], self.h_scat])
        t = np.concatenate([[0.0], tc])
        tot = w.sum()
        if tot <= 0:
            return 0.0, 0.0
        m = (w * t).sum() / tot
        return float(m), float(np.sqrt(max((w * (t - m) ** 2).sum() / tot, 0.0)))


def _rotate(ux, uy, uz, theta, phi):
    """Rotate direction (ux,uy,uz) by polar angle theta and azimuth phi (standard MC update)."""
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    out = np.empty((3, ux.size))
    near = np.abs(uz) > 0.99999
    den = np.sqrt(np.maximum(1 - uz * uz, 1e-30))
    nx = st * (ux * uz * cp - uy * sp) / den + ux * ct
    ny = st * (uy * uz * cp + ux * sp) / den + uy * ct
    nz = -st * cp * den + uz * ct
    out[0], out[1], out[2] = nx, ny, nz
    if near.any():
        s = np.sign(uz[near])
        out[0, near] = st[near] * cp[near]
        out[1, near] = st[near] * sp[near]
        out[2, near] = s * ct[near]
    n = np.sqrt((out ** 2).sum(0))
    return out[0] / n, out[1] / n, out[2] / n


def launch(rng, n, divergence_rad, waist_m):
    """Gaussian beam: 1/e² radius waist_m at z = 0, 1/e² half-angle divergence."""
    r = waist_m / 2 * np.sqrt(-2 * np.log(1 - rng.random(n)))           # 1/e² radius w ⇒ σ = w/2
    a = 2 * np.pi * rng.random(n)
    x, y = r * np.cos(a), r * np.sin(a)
    th = divergence_rad / 2 * np.sqrt(-2 * np.log(1 - rng.random(n)))
    ph = 2 * np.pi * rng.random(n)
    ux, uy, uz = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
    return x, y, np.zeros(n), ux, uy, uz


_CAP = {}


def cap_pdf(phase, mu, alpha):
    """Phase function (1/sr) averaged over a cone of half-angle alpha whose axis is at angle
    arccos(mu) from the forward direction (small-angle, flat-disk geometry for the table; the point
    value beyond 1.5 rad where the phase function is smooth on the scale of alpha)."""
    key = id(phase)
    if key not in _CAP:
        from scipy.interpolate import RegularGridInterpolator
        al = np.geomspace(1e-5, 0.6, 48)
        th = np.concatenate([[0.0], np.geomspace(1e-6, 1.6, 110)])
        tab = np.empty((al.size, th.size))
        for i, a in enumerate(al):
            for j, t in enumerate(th):
                rho = np.linspace(max(t - a, 0.0), t + a, 241)
                if t < 1e-12:
                    frac = (rho <= a).astype(float)
                else:
                    cosv = (rho * rho + t * t - a * a) / (2 * np.maximum(rho, 1e-300) * t)
                    frac = np.arccos(np.clip(cosv, -1, 1)) / np.pi
                    frac[rho <= a - t] = 1.0
                f = phase.pdf_sr(np.cos(rho)) * 2 * np.pi * rho * frac
                tab[i, j] = trapz(f, rho) / (np.pi * a * a)
        _CAP[key] = (phase, RegularGridInterpolator((np.log(al), th), np.log(np.maximum(tab, 1e-300)),
                                                    bounds_error=False, fill_value=None))
    _, itp = _CAP[key]
    th = np.arccos(np.clip(mu, -1, 1))
    out = phase.pdf_sr(mu)
    m = th < 1.5
    if np.any(m):
        a = np.clip(np.broadcast_to(alpha, th.shape)[m], 1e-5, 0.6)
        out = np.array(out, float, copy=True)
        out[m] = np.exp(itp(np.column_stack([np.log(a), th[m]])))
    return out


def simulate(water_c, water_b, phase, L, divergence_rad, waist_m, aperture_radius, fov_rad,
             offset=(0.0, 0.0), n_photons=20000, max_orders=40, n_bins=1000, t_max=None,
             seed=0, n_paths=60, progress=None, omega_near=0.5):
    """Monte Carlo channel between a beam at the origin (pointing +z) and a receiver disk at
    z = L, centred at `offset` (x, y), with its normal facing the transmitter."""
    rng = np.random.default_rng(seed)
    c, b = float(water_c), float(water_b)
    w0 = b / c if c > 0 else 0.0
    A = np.pi * aperture_radius ** 2
    rx, ry = offset
    cos_fov = np.cos(fov_rad)
    if t_max is None:                                  # excess-delay window
        t_max = max(20 * L / V_WATER * min(1.0, 0.02 + 0.05 * b * L), 2e-9)
    edges = np.linspace(0.0, t_max, n_bins + 1)
    h = np.zeros(n_bins)
    by_order = np.zeros(max_orders)
    ang_list, angw_list = [], []

    # ---------------- ballistic: straight rays
    x, y, z, ux, uy, uz = launch(rng, 200_000, divergence_rad, waist_m)
    ok = uz > 0
    t = L / np.where(ok, uz, 1)
    xe, ye = x + ux * t, y + uy * t
    hit = ok & ((xe - rx) ** 2 + (ye - ry) ** 2 <= aperture_radius ** 2) & (uz >= cos_fov)
    geom = float(hit.mean())
    ballistic = geom * np.exp(-c * L)

    # ---------------- scattered photons
    x, y, z, ux, uy, uz = launch(rng, n_photons, divergence_rad, waist_m)
    w = np.ones(n_photons)
    s_tot = np.zeros(n_photons)
    # forced first collision before the receiver plane
    smax = np.where(uz > 0, (L - z) / np.maximum(uz, 1e-12), 1e3 * L)
    pcol = 1 - np.exp(-c * smax)
    s = -np.log(1 - rng.random(n_photons) * pcol) / c
    w *= pcol
    paths = []
    keep_idx = np.arange(min(n_paths, n_photons))
    trail = [[np.array([x[i], y[i], z[i]])] for i in keep_idx]
    ids = np.arange(n_photons)
    for order in range(1, max_orders + 1):
        x, y, z = x + ux * s, y + uy * s, z + uz * s
        s_tot = s_tot + s
        for j, i in enumerate(keep_idx):
            k = np.searchsorted(ids, i)
            if k < ids.size and ids[k] == i:
                trail[j].append(np.array([x[k], y[k], z[k]]))
        # scattering event: weight times albedo, then local estimate toward the aperture
        w = w * w0
        dx, dy, dz = rx - x, ry - y, L - z
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        vx, vy, vz = dx / r, dy / r, dz / r
        mu_s = ux * vx + uy * vy + uz * vz
        cos_r = vz                                           # arrival angle w.r.t. Rx normal
        inside = (dz > 0) & (cos_r >= cos_fov)
        dOmega = A * np.maximum(cos_r, 0) / (r * r + A / np.pi)     # regularised near the disk
        # probability that the next direction points into the aperture: the phase function averaged
        # over the aperture's angular disk (a point value of the sharp forward peak is very noisy)
        p_dir = cap_pdf(phase, mu_s, np.sqrt(dOmega / np.pi)) * dOmega
        near = dOmega > omega_near          # aperture too large seen from here: count crossings
        contrib = np.where(inside & ~near, w * p_dir * np.exp(-c * r), 0.0)
        tt = (s_tot + r - L) / V_WATER
        np.add.at(h, np.clip(np.searchsorted(edges, tt) - 1, 0, n_bins - 1), contrib)
        by_order[order - 1] += contrib.sum()
        sel = inside & ~near
        if sel.any():
            ang_list.append(np.arccos(np.clip(cos_r[sel], -1, 1)))
            angw_list.append(contrib[sel])
        # new direction
        th = phase.sample(rng, ux.size)
        ph = 2 * np.pi * rng.random(ux.size)
        ux, uy, uz = _rotate(ux, uy, uz, th, ph)
        s = -np.log(1 - rng.random(ux.size)) / c
        # photons that would cross the receiver plane leave the volume (their direct arrival
        # is already accounted for by the local estimates)
        cross = (uz > 0) & (z + uz * s >= L)
        # after a 'near' event the flight itself is scored when it enters the aperture
        nc = cross & near
        if nc.any():
            sp = (L - z[nc]) / uz[nc]
            hx, hy = x[nc] + ux[nc] * sp - rx, y[nc] + uy[nc] * sp - ry
            acc = (hx * hx + hy * hy <= aperture_radius ** 2) & (uz[nc] >= cos_fov)
            cw = np.where(acc, w[nc], 0.0)
            ct = (s_tot[nc] + sp - L) / V_WATER
            np.add.at(h, np.clip(np.searchsorted(edges, ct) - 1, 0, n_bins - 1), cw)
            by_order[order - 1] += cw.sum()
            ang_list.append(np.arccos(np.clip(uz[nc][acc], -1, 1)))
            angw_list.append(cw[acc])
        # Russian roulette
        small = w < 1e-4
        survive = rng.random(ux.size) < 0.1
        w = np.where(small & survive, w * 10, w)
        alive = ~cross & ~(small & ~survive) & (z > -L) & ((s_tot - L) / V_WATER < 1.5 * t_max)
        if progress is not None:
            progress(order, max_orders)
        if not alive.any():
            break
        x, y, z, ux, uy, uz, w, s, s_tot, ids = (v[alive] for v in (x, y, z, ux, uy, uz, w, s, s_tot, ids))
    for tr in trail:
        paths.append(np.array(tr))
    scattered = h.sum() / n_photons
    ang = np.concatenate(ang_list) if ang_list else np.zeros(0)
    angw = np.concatenate(angw_list) / n_photons if angw_list else np.zeros(0)
    return MCResult(L=L, n_photons=n_photons, ballistic=float(ballistic), scattered=float(scattered),
                    by_order=by_order / n_photons, t_edges=edges, h_scat=h / n_photons, geom=geom,
                    paths=paths, angles=ang, angle_w=angw)
