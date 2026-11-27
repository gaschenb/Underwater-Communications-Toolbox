"""Camera (optical camera communication, OCC) links: image formation of the transmitter through
water and decoding from global- or rolling-shutter frame sequences.

Image formation, per frame f (and row r for a rolling shutter):
  exposure window  [t_f, t_f + T_e]               global shutter, t_f = (f + φ) / fps
                   [t_f + r t_row, … + T_e]         rolling shutter, t_row = 1 / (fps (H + vblank))
  electrons(pixel) = Σ_LED [ N_b T_b(pixel) + N_s T_s(pixel) ] ∫_window x_LED(t) dt / T_e
                     + (L_bg A Ω_pix Δλ η / hν + i_dark) T_e
  N_b, N_s: ballistic and scattered photoelectrons per exposure at full drive (channel model with
  the camera aperture f/N and field of view), T_b a disk of the source's image diameter f D / L
  blurred by the lens PSF, T_s a scattering halo with the small-angle apparent width
  θ_h = θ_rms / √3. Shot noise is Poisson, then read noise, full-well clipping and ADC.
Decoding: the source is located in the mean image; per-frame (global) or per-row (rolling) sums
over its region of interest, minus the border background, give a sample stream with known
exposure windows. Chip-based schemes keep samples whose window lies inside one chip; chips never
seen (inter-frame gaps, rows outside the source image) are erasures. The first complete symbols
observed form a known pilot header that calibrates the levels.
Camera-specific schemes
  UFSOOK      undersampled frequency-shift OOK: '1' blinks at (n + ½) fps (alternate frames differ),
              '0' at n fps (frames agree); one bit per two frames, flicker-free (Roberts 2013)
  UPSOOK      undersampled phase-shift OOK: blink at n fps with phase 0 / π; one bit per frame
  RS-FSK      rolling-shutter FSK: the tone (stripe spacing) in each frame carries log2 M bits
  Spatial OOK an LED array imaged on the sensor, one bit per LED per frame (MIMO)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import schemes as S
from .channel import forward_lobe
from .constants import photon_energy

CHIP_SCHEMES = ["OOK-NRZ", "OOK-RZ", "Manchester", "2-PPM", "4-PPM", "8-PPM", "MPPM (8,2)", "VPPM (50 %)",
                "VPPM (25 %)", "4-PWM", "4-PAM", "8-PAM", "4-CSK", "16-CSK"]
SPECIAL = ["UFSOOK", "UPSOOK", "RS-FSK", "Spatial OOK"]
CAMERA_SCHEMES = CHIP_SCHEMES + SPECIAL


def applicable(name, cam):
    if name == "RS-FSK":
        return cam.shutter == "rolling"
    if name.endswith("CSK"):
        return cam.color
    return True


# =========================================================================== image formation
def _disk(H, W, cy, cx, r, ss=3):
    """Area fraction of each pixel covered by a disk (supersampled)."""
    y0, y1 = int(max(np.floor(cy - r - 2), 0)), int(min(np.ceil(cy + r + 2), H))
    x0, x1 = int(max(np.floor(cx - r - 2), 0)), int(min(np.ceil(cx + r + 2), W))
    out = np.zeros((H, W))
    if y1 <= y0 or x1 <= x0:
        return out
    yy = (np.arange(y0 * ss, y1 * ss) + 0.5) / ss
    xx = (np.arange(x0 * ss, x1 * ss) + 0.5) / ss
    inside = ((yy[:, None] - cy) ** 2 + (xx[None, :] - cx) ** 2) <= max(r, 0.35) ** 2
    out[y0:y1, x0:x1] = inside.reshape(y1 - y0, ss, x1 - x0, ss).mean((1, 3))
    return out


def _gauss(H, W, cy, cx, sig):
    y = np.arange(H) + 0.5 - cy
    x = np.arange(W) + 0.5 - cx
    g = np.exp(-y[:, None] ** 2 / (2 * sig ** 2)) * np.exp(-x[None, :] ** 2 / (2 * sig ** 2))
    return g / (2 * np.pi * sig ** 2)


@dataclass
class Optics:
    templates: np.ndarray          # (n_led, H, W) expected electrons per exposure at full drive
    centres: list
    r_img: float                   # image radius (px)
    halo_px: float
    bg_e: float                    # background + dark electrons per pixel per exposure
    Nb: float                      # ballistic electrons per LED per exposure (on sensor)
    Ns: float
    gain: float
    gain_b: float


def image_optics(sc, n_led, lam=None):
    cam, tx = sc.camera, sc.tx
    lam = tx.wavelength_nm if lam is None else lam
    H, W = cam.height, cam.width
    ch = sc.channel(wavelength=lam, camera=True)
    eta = cam.qe * sc.optics.optics_T / photon_energy(lam)
    P_led = tx.power_w / max(n_led, 1)
    T = cam.exposure_s
    Nb = P_led * ch.gain_ballistic * eta * T
    Ns = P_led * ch.gain_scattered * eta * T
    L = sc.range_m
    r_img = 0.5 * tx.source_mm / 1e3 / L / cam.ifov
    P, th1 = forward_lobe(sc.water.phase_function(lam), 10.0)
    th_rms = th1 * np.sqrt(max(P * float(sc.water.b(lam)) * L, 1e-9))
    halo = max(th_rms / np.sqrt(3) / np.sqrt(2) / cam.ifov, 1.0)
    from scipy.ndimage import gaussian_filter
    k = int(round(np.sqrt(n_led)))
    pitch = tx.pitch_mm / 1e3 / L / cam.ifov
    temps, cents = [], []
    for i in range(n_led):
        a, b = divmod(i, k)
        cy = H / 2 + (a - (k - 1) / 2) * pitch
        cx = W / 2 + (b - (k - 1) / 2) * pitch
        d = gaussian_filter(_disk(H, W, cy, cx, r_img), 0.7, mode="constant")
        area = np.pi * max(r_img, 0.35) ** 2
        tpl = Nb * d / area + Ns * _gauss(H, W, cy, cx, halo)
        temps.append(tpl); cents.append((cy, cx))
    Lr = sc.ambient.radiance(sc.water, lam)
    omega = cam.ifov ** 2
    bg = Lr * np.pi * (cam.aperture_m / 2) ** 2 * omega * cam.band_nm * sc.optics.optics_T * eta * T \
        + cam.dark_e_s * T
    return Optics(np.array(temps), cents, r_img, halo, float(bg), float(Nb), float(Ns), ch.gain, ch.gain_ballistic)


def sensor(e_mean, cam, rng):
    e = rng.poisson(np.maximum(e_mean, 0)).astype(float) + cam.read_noise * rng.standard_normal(e_mean.shape)
    e = np.clip(e, 0, cam.full_well)
    return np.round(e * cam.dn_per_e)


# =========================================================================== transmit waveforms
def _square(t, f, shift=0.0):
    return (np.mod(f * (t - shift), 1.0) < 0.5).astype(float)


@dataclass
class CamRun:
    scheme: str
    frames: np.ndarray             # (F, H, W) or (F, H, W, 3) digital numbers
    t: np.ndarray                  # fine time grid of the drive (s)
    drive: np.ndarray              # (n_t, n_led, n_ch)
    frame_t: np.ndarray            # frame exposure start (row 0)
    exposure: float
    row_time: float
    rolling: bool
    samples_t: np.ndarray          # centre time of every sample window
    samples: np.ndarray            # normalised sample values (n, n_led or channels)
    valid: np.ndarray
    tx_bits: np.ndarray
    rx_bits: np.ndarray
    errors: float
    n_bits: int
    erased_bits: int
    duration: float
    chip_rate: float = 0.0
    chips_tx: np.ndarray = None
    chips_rx: np.ndarray = None
    chip_seen: np.ndarray = None
    optics: Optics = None
    row_profile: np.ndarray = None # (F, H) background-subtracted ROI row sums (rolling)
    roi: tuple = None
    info: dict = field(default_factory=dict)

    @property
    def ber(self):
        return self.errors / max(self.n_bits, 1)

    @property
    def goodput(self):
        """Correctly decoded bits per second (erasures and errors removed)."""
        return max(self.n_bits - self.errors, 0) / max(self.duration, 1e-12)


def _hue(lam):
    return 2 if lam < 490 else (1 if lam < 585 else 0)        # camera channel index B, G, R


def simulate(sc, scheme=None, n_frames=None, seed=0, n_pilot=None):
    scheme = sc.camera_scheme if scheme is None else scheme
    cam, tx = sc.camera, sc.tx
    rng = np.random.default_rng(seed)
    F = sc.n_frames if n_frames is None else int(n_frames)
    Tf, T = 1.0 / cam.fps, cam.exposure_s
    rolling = cam.shutter == "rolling"
    H, W = cam.height, cam.width
    trow = cam.row_time if rolling else 0.0
    frame_t = (np.arange(F) + sc.frame_phase) * Tf
    t_end = frame_t[-1] + T + H * trow + Tf
    is_csk = scheme.endswith("CSK")
    if is_csk and not cam.color:
        raise ValueError("CSK needs a colour camera")
    if scheme == "RS-FSK" and not rolling:
        raise ValueError("RS-FSK needs a rolling-shutter camera")
    n_led = tx.array ** 2 if (scheme == "Spatial OOK" or tx.array > 1) else 1
    nch = 3 if is_csk else 1
    # ------------------------------------------------------------- drive waveform
    chips = chip_seq = None
    Rc = 0.0
    sch = None
    if scheme in SPECIAL:
        f0 = sc.camera_carrier_hz
        n = max(1, int(round(f0 / cam.fps)))
        if scheme in ("UFSOOK", "UPSOOK"):
            dt = min(1 / (n * cam.fps * 24), T / 6)
        elif scheme == "RS-FSK":
            dt = min(1 / (f0 * 3 * 24), T / 6, trow / 2)
        else:
            dt = min(Tf / 50, T / 6)
    else:
        sch = S.make(scheme)
        Rc = sc.camera_chip_rate
        dt = min(1 / (Rc * 12), T / 6, (trow / 2 if rolling else np.inf))
    dt = max(dt, t_end / 3e6)
    t = np.arange(0, t_end, dt)
    D = np.zeros((t.size, n_led, nch))
    info = {}
    if sch is not None:
        nchip = int(np.ceil(t_end * Rc)) + sch.n
        nsym = int(np.ceil(nchip / sch.n))
        bits = rng.integers(0, 2, nsym * sch.k).astype(np.int8)
        chips, syms = sch.encode(bits)
        idx = np.minimum((t * Rc).astype(int), len(chips) - 1)
        if is_csk:
            D[:, :, :] = chips[idx][:, None, :]
        else:
            D[:, :, 0] = chips[idx][:, None]
    elif scheme == "UFSOOK":
        n = max(1, int(round(sc.camera_carrier_hz / cam.fps)))
        nb = int(np.ceil(t_end / (2 * Tf))) + 1
        bits = rng.integers(0, 2, nb).astype(np.int8)
        bits[:2] = [0, 1]                                    # pilot: one '0' then one '1'
        b = bits[np.minimum((t / (2 * Tf)).astype(int), nb - 1)]
        f = np.where(b == 1, (n + 0.5) * cam.fps, n * cam.fps)
        # exposure of frame 0 centred in an 'on' half-cycle
        shift = sc.frame_phase * Tf + T / 2 - 1 / (4 * f) + (H // 2) * trow
        D[:, :, 0] = _square(t, f, shift)[:, None]
        info["f_blink"] = (n * cam.fps, (n + 0.5) * cam.fps)
    elif scheme == "UPSOOK":
        n = max(1, int(round(sc.camera_carrier_hz / cam.fps)))
        fb = n * cam.fps
        nb = int(np.ceil(t_end / Tf)) + 1
        bits = rng.integers(0, 2, nb).astype(np.int8)
        bits[:4] = [1, 0, 1, 0]
        b = bits[np.minimum((t / Tf).astype(int), nb - 1)]
        shift = sc.frame_phase * Tf + T / 2 - 1 / (4 * fb) + (H // 2) * trow + np.where(b == 1, 0.0, 0.5 / fb)
        D[:, :, 0] = _square(t, fb, shift)[:, None]
        info["f_blink"] = (fb,)
    elif scheme == "RS-FSK":
        M = 4
        tones = sc.camera_carrier_hz * (1 + 0.5 * np.arange(M))
        nb = int(np.ceil(t_end / Tf)) + 1
        syms = rng.integers(0, M, nb)
        syms[:M] = np.arange(M)                               # pilot: every tone once
        bits = S.int_to_bits(syms, 2)
        s_t = syms[np.minimum((t / Tf).astype(int), nb - 1)]
        D[:, :, 0] = _square(t, tones[s_t])[:, None]
        info["tones"] = tones
    elif scheme == "Spatial OOK":
        nb = int(np.ceil(t_end / Tf)) + 1
        B = rng.integers(0, 2, (nb, n_led)).astype(np.int8)
        B[0], B[1] = 1, 0                                     # pilot frames: all on, all off
        bits = B.ravel()
        D[:, :, 0] = B[np.minimum((t / Tf).astype(int), nb - 1)]
    # ------------------------------------------------------------- exposures
    cum = np.concatenate([np.zeros((1, n_led, nch)), np.cumsum(D, 0) * dt], 0)
    tg = np.concatenate([t, [t[-1] + dt]])
    rows = np.arange(H) if rolling else np.zeros(1)
    starts = frame_t[:, None] + rows[None, :] * trow                 # (F, R)
    def integ(a, b):
        out = np.empty(a.shape + (n_led, nch))
        for l in range(n_led):
            for c in range(nch):
                out[..., l, c] = (np.interp(b, tg, cum[:, l, c]) - np.interp(a, tg, cum[:, l, c])) / T
        return out
    E = integ(starts, starts + T)                                     # (F, R, n_led, nch) mean drive
    # ------------------------------------------------------------- images
    lams = list(tx.rgb_nm) if is_csk else [tx.wavelength_nm]
    opts = [image_optics(sc, n_led, lam) for lam in lams]
    if cam.color:
        Mx = cam.color_matrix()
        src = np.zeros((nch, 3))
        for j, lam in enumerate(lams):
            src[j, _hue(lam)] = 1.0
        resp = src @ Mx.T                                             # (nch, cam channels)
        ncam = 3
    else:
        resp = np.ones((1, 1)); ncam = 1
    mean = np.zeros((F, H, W, ncam))
    for j in range(nch):
        for l in range(n_led):
            tpl = opts[j].templates[l]                                # (H, W)
            e_fl = E[:, :, l, j]                                      # (F, R)
            img = tpl[None] * (e_fl[:, :, None] if rolling else e_fl[:, :1, None])
            mean += img[..., None] * resp[j][None, None, None, :]
    mean += opts[0].bg_e
    frames = sensor(mean, cam, rng)
    if ncam == 1:
        frames = frames[..., 0]
    # ------------------------------------------------------------- decoding
    o = opts[0]
    fr = frames.astype(float) if frames.ndim == 4 else frames.astype(float)[..., None]
    avg = fr.mean(0).sum(-1)
    border = np.concatenate([avg[:4].ravel(), avg[-4:].ravel(), avg[:, :4].ravel(), avg[:, -4:].ravel()])
    bgl = np.median(border)
    # locate the source: centroid of pixels well above background
    exc = np.maximum(avg - bgl, 0)
    thr = 0.2 * exc.max() if exc.max() > 0 else np.inf
    msk = exc > thr
    if msk.sum() >= 1:
        yy, xx = np.nonzero(msk)
        wts = exc[msk]
        cy, cx = float((yy * wts).sum() / wts.sum()), float((xx * wts).sum() / wts.sum())
    else:
        cy, cx = H / 2, W / 2
    k = int(round(np.sqrt(n_led)))
    pitch = tx.pitch_mm / 1e3 / sc.range_m / cam.ifov
    rr = max(o.r_img * 1.3 + 1.5, 2.0)
    if n_led > 1 and scheme != "Spatial OOK":
        rr = max(rr, (k - 1) * pitch / 2 * 1.42 + o.r_img + 2)
    yy, xx = np.mgrid[0:H, 0:W]
    rois = []
    if scheme == "Spatial OOK":
        rr_led = max(min(o.r_img * 1.3 + 1.0, pitch * 0.48), 1.0)
        for i in range(n_led):
            a, b = divmod(i, k)
            py, px = cy + (a - (k - 1) / 2) * pitch, cx + (b - (k - 1) / 2) * pitch
            rois.append(((yy + 0.5 - py) ** 2 + (xx + 0.5 - px) ** 2) <= rr_led ** 2)
    else:
        rois.append(((yy + 0.5 - cy) ** 2 + (xx + 0.5 - cx) ** 2) <= rr ** 2)
    bg_f = np.stack([np.median(np.concatenate([f[:4].reshape(-1, fr.shape[-1]), f[-4:].reshape(-1, fr.shape[-1])]),
                               0) for f in fr])                       # (F, ch) per-frame background
    sub = fr - bg_f[:, None, None, :]
    out = dict(info=info, optics=o, roi=(cy, cx, rr))
    duration = F * Tf
    if rolling:
        m = rois[0]
        prof = np.einsum("fhwc,hw->fhc", sub, m.astype(float))      # (F, H, ch) row sums
        # per-row 'on' level: the largest value over frames, smoothed over neighbouring rows
        from scipy.ndimage import uniform_filter1d, maximum_filter1d
        top = np.abs(prof.sum(-1)).max(0) if F > 1 else np.abs(prof.sum(-1))[0]
        wrow = uniform_filter1d(maximum_filter1d(top, 3), 7)
        valid_r = wrow > 0.25 * wrow.max() if wrow.max() > 0 else np.zeros(H, bool)
        st = starts + T / 2                                            # (F, H)
        vals = prof / np.where(valid_r, wrow, 1)[None, :, None]
        s_t = st.ravel(); s_v = vals.reshape(-1, vals.shape[-1]); valid = np.tile(valid_r, F)
        out["row_profile"] = prof.sum(-1)
    else:
        vals = np.stack([np.einsum("fhwc,hw->fc", sub, m.astype(float)) for m in rois], 1)  # (F, n_roi, ch)
        s_t = frame_t + T / 2
        s_v = vals.reshape(F, -1)
        valid = np.ones(F, bool)
    if rolling and scheme in ("UFSOOK", "UPSOOK"):
        # undersampled schemes take one sample per frame: the valid row closest to the source centre
        rv = np.nonzero(valid_r)[0]
        r0 = int(rv[np.argmin(np.abs(rv + 0.5 - cy))]) if rv.size else int(cy)
        s_v = vals[:, r0]; s_t = st[:, r0]; valid = np.ones(F, bool)
    # ---- scheme decisions
    if sch is not None:
        res = _decode_chips(sch, s_t, s_v, valid, T, Rc, chips, syms, bits, n_pilot, is_csk)
    elif scheme == "UFSOOK":
        v = s_v[:, 0]
        nb_obs = F // 2
        a, b = v[0:2 * nb_obs:2], v[1:2 * nb_obs:2]
        d = np.abs(a - b)
        on, off = max(v.max(), 1e-9), v.min()
        thr = 0.5 * (on - off)
        # pilot: bit 0 ('0'), bit 1 ('1') calibrate the difference threshold
        if nb_obs >= 2:
            thr = 0.5 * (d[0] + d[1])
        rb = (d > thr).astype(np.int8)
        tb = bits[:nb_obs]
        e = int(np.sum(rb[2:] != tb[2:])); nbits = max(nb_obs - 2, 0)
        res = dict(rx_bits=rb, errors=e, n_bits=nbits, erased=0, tx_bits=tb)
    elif scheme == "UPSOOK":
        v = s_v[:, 0]
        m1, m0 = v[[0, 2]].mean(), v[[1, 3]].mean()
        rb = (v > 0.5 * (m0 + m1)).astype(np.int8)
        tb = bits[:F]
        e = int(np.sum(rb[4:] != tb[4:]))
        res = dict(rx_bits=rb, errors=e, n_bits=max(F - 4, 0), erased=0, tx_bits=tb)
    elif scheme == "RS-FSK":
        tones = info["tones"]
        prof = out["row_profile"]
        st = (starts + T / 2)
        rs = []
        for f_ in range(F):
            vr = valid_r
            x = prof[f_, vr] - prof[f_, vr].mean()
            tt = st[f_, vr]
            pw = [np.abs(np.sum(x * np.exp(-2j * np.pi * ft * tt))) for ft in tones]
            rs.append(int(np.argmax(pw)))
        rs = np.array(rs)
        tx_s = S.bits_to_int(bits[: 2 * F], 2)
        rb = S.int_to_bits(rs, 2)
        e = int(np.sum(rb[8:] != bits[8: 2 * F]))
        res = dict(rx_bits=rb, errors=e, n_bits=max(2 * F - 8, 0), erased=0, tx_bits=bits[: 2 * F],
                   rx_syms=rs, tx_syms=tx_s)
        out["info"]["n_rows"] = int(valid_r.sum())
    else:  # Spatial OOK
        v = s_v                                                       # (F, n_led)
        on, off = v[0], v[1]
        rb = (v > 0.5 * (on + off)[None, :]).astype(np.int8).ravel()
        tb = bits[: F * n_led]
        e = int(np.sum(rb[2 * n_led:] != tb[2 * n_led:]))
        res = dict(rx_bits=rb, errors=e, n_bits=max((F - 2) * n_led, 0), erased=0, tx_bits=tb)
    run = CamRun(scheme=scheme, frames=frames, t=t, drive=D, frame_t=frame_t, exposure=T, row_time=trow,
                 rolling=rolling, samples_t=s_t, samples=s_v, valid=valid, tx_bits=res["tx_bits"],
                 rx_bits=res["rx_bits"], errors=res["errors"], n_bits=res["n_bits"],
                 erased_bits=res.get("erased", 0), duration=duration, chip_rate=Rc,
                 chips_tx=res.get("chips_tx"), chips_rx=res.get("chips_rx"), chip_seen=res.get("chip_seen"),
                 optics=o, row_profile=out.get("row_profile"), roi=out["roi"], info=out["info"])
    run.info.update(r_img=o.r_img, halo_px=o.halo_px, Nb=o.Nb, Ns=o.Ns, bg_e=o.bg_e, gain=o.gain,
                    n_led=n_led, peak_dn=float(frames.max()), symbols=res.get("symbols"),
                    observed_fraction=res.get("observed_fraction", 1.0), pilot=res.get("pilot", 0))
    return run


def _decode_chips(sch, s_t, s_v, valid, T, Rc, chips, syms, bits, n_pilot, is_csk):
    Tc = 1.0 / Rc
    eps = 1e-9 * Tc
    j0 = np.floor((s_t - T / 2) / Tc + eps).astype(int)
    j1 = np.floor((s_t + T / 2) / Tc - eps).astype(int)
    ok = valid & (j0 == j1) & (j0 >= 0) & (j0 < len(chips))
    nchip = len(chips)
    nc = s_v.shape[1]
    acc = np.zeros((nchip, nc)); cnt = np.zeros(nchip)
    np.add.at(acc, j0[ok], s_v[ok])
    np.add.at(cnt, j0[ok], 1)
    seen = cnt > 0
    y = acc / np.maximum(cnt, 1)[:, None]
    n = sch.n
    nsym = nchip // n
    comp = seen[: nsym * n].reshape(nsym, n).all(1)
    idx = np.nonzero(comp)[0]
    if n_pilot is None:
        n_pilot = max(4, 1 << sch.k) if sch.k <= 4 else 24
    # pilot header: the first complete symbols, extended until every symbol value (levels for
    # PAM / CSK points) has been seen, at most half of what was observed
    need = (1 << sch.k) if sch.family in ("level", "colour") else 0
    cnt_s = {}
    npil = 0
    for i_, s_ in enumerate(syms[idx]):
        cnt_s[s_] = cnt_s.get(s_, 0) + 1
        npil = i_ + 1
        if npil >= n_pilot and (need == 0 or len(cnt_s) == need):
            break
        if npil >= idx.size // 2:
            break
    pil, dat = idx[:npil], idx[npil:]
    chip_rx = np.full(nchip, np.nan)
    chip_rx[seen] = y[seen, 0] if not is_csk else y[seen].sum(1)
    base = dict(chips_tx=chips if not is_csk else chips.sum(1), chips_rx=chip_rx, chip_seen=seen,
                observed_fraction=float(seen.mean()), pilot=int(pil.size))
    if pil.size < 2 or dat.size == 0:
        return dict(base, rx_bits=np.zeros(0, np.int8), errors=0, n_bits=0, erased=int(nsym * sch.k),
                    tx_bits=bits[: nsym * sch.k], symbols=0)
    sel = lambda ii: (ii[:, None] * n + np.arange(n)[None, :]).ravel()
    if is_csk:
        yp, lab = y[sel(pil)], syms[pil]
        yd = y[sel(dat)]
    else:
        yp, lab = y[sel(pil), 0], chips[sel(pil)]
        yd = y[sel(dat), 0]
    if is_csk and any(not np.any(lab == m) for m in range(sch.M)):     # pilot missed a point
        return dict(base, rx_bits=np.zeros(0, np.int8), errors=0, n_bits=0, erased=int(nsym * sch.k),
                    tx_bits=bits[: nsym * sch.k], symbols=0)
    cal = sch.calibrate(yp, lab)
    rs, rb, _ = sch.decode(yd, cal, n_sym=dat.size)
    tb = S.int_to_bits(syms[dat], sch.k)
    e = int(np.sum(rb[: tb.size] != tb[: rb.size]))
    return dict(base, rx_bits=rb, errors=e, n_bits=int(tb.size), erased=int((nsym - idx.size) * sch.k),
                tx_bits=tb, symbols=int(dat.size), cal=cal)


# =========================================================================== analytic camera link
def sample_stats(sc, scheme=None):
    """Per-sample ROI statistics (electrons) at drive 0 and 1, samples per chip and the observed
    fraction, for closed-form camera BER curves."""
    scheme = sc.camera_scheme if scheme is None else scheme
    cam = sc.camera
    n_led = sc.tx.array ** 2 if (scheme == "Spatial OOK" or sc.tx.array > 1) else 1
    lam = sc.tx.rgb_nm[1] if scheme.endswith("CSK") else sc.tx.wavelength_nm
    o = image_optics(sc, n_led, lam)
    rr = max(o.r_img * 1.3 + 1.5, 2.0)
    H, W = cam.height, cam.width
    cy, cx = o.centres[0] if scheme == "Spatial OOK" else (H / 2, W / 2)
    yy, xx = np.mgrid[0:H, 0:W]
    roi = ((yy + 0.5 - cy) ** 2 + (xx + 0.5 - cx) ** 2) <= rr ** 2
    tpl = o.templates[0] if scheme == "Spatial OOK" else o.templates.sum(0)
    rolling = cam.shutter == "rolling"
    rn2 = cam.read_noise ** 2 + 1 / (12 * cam.dn_per_e ** 2)          # read + ADC quantisation noise
    if rolling:
        rows = np.nonzero(roi.any(1))[0]
        sig_rows = (tpl * roi).sum(1)[rows]
        npx = roi.sum(1)[rows]
        keep = sig_rows > 0.25 * sig_rows.max()
        s1 = float(np.mean(sig_rows[keep])) if keep.any() else 0.0
        npix = float(np.mean(npx[keep])) if keep.any() else 1.0
        n_rows = int(keep.sum())
        Tc = 1 / sc.camera_chip_rate
        per_chip = max((Tc - cam.exposure_s) / cam.row_time, 0.0)
        frame_obs = n_rows * cam.row_time
        obs = min(1.0, max(frame_obs - Tc, 0) * cam.fps) if per_chip >= 1 else 0.0
    else:
        s1 = float((tpl * roi).sum()); npix = float(roi.sum()); n_rows = 0
        per_chip = 1.0; obs = 1.0
    b = o.bg_e * npix
    var0 = b + rn2 * npix
    var1 = b + s1 + rn2 * npix
    return dict(m0=0.0, m1=s1, s0=float(np.sqrt(var0)), s1=float(np.sqrt(var1)), per_chip=per_chip,
                observed=obs, npix=npix, n_rows=n_rows, signal_e=s1, bg_e=b, optics=o)


def ber_theory(sc, scheme=None):
    scheme = sc.camera_scheme if scheme is None else scheme
    st = sample_stats(sc, scheme)
    m0, m1, s0, s1 = st["m0"], st["m1"], st["s0"], st["s1"]
    if m1 <= 0:
        return 0.5
    if scheme in CHIP_SCHEMES:
        k = max(st["per_chip"], 1.0)
        v = S.make(scheme).theory(m0, s0 / np.sqrt(k), m1, s1 / np.sqrt(k))
        return float(v)
    if scheme in ("UPSOOK", "Spatial OOK"):
        return float(S.OOK().theory(m0, s0, m1, s1))
    if scheme == "UFSOOK":
        # '1' if two frames differ: |a - b| vs threshold at half the on-off swing
        return float(S.OOK().theory(0.0, np.sqrt(2) * s0, m1, np.sqrt(s0 ** 2 + s1 ** 2)))
    if scheme == "RS-FSK":
        N = max(st["n_rows"], 1)
        A = 2 / np.pi * m1                                            # square-wave fundamental
        sig2 = 0.5 * (s0 ** 2 + s1 ** 2)
        es = A ** 2 * N / 4 / sig2
        M = 4
        from math import comb
        Ps = sum((-1) ** (k + 1) * comb(M - 1, k) / (k + 1) * np.exp(-k / (k + 1) * es) for k in range(1, M))
        return float(min(0.5, max(Ps, 0) * (M / 2) / (M - 1)))
    return np.nan


def throughput(sc, scheme=None):
    """Raw bit rate (bit/s) delivered by the camera link before errors (erasures removed)."""
    scheme = sc.camera_scheme if scheme is None else scheme
    cam = sc.camera
    if scheme in CHIP_SCHEMES:
        sch = S.make(scheme)
        st = sample_stats(sc, scheme)
        return sc.camera_chip_rate * sch.bits_per_chip * st["observed"]
    if scheme == "UFSOOK":
        return cam.fps / 2
    if scheme == "UPSOOK":
        return cam.fps
    if scheme == "RS-FSK":
        return 2 * cam.fps
    if scheme == "Spatial OOK":
        return cam.fps * sc.tx.array ** 2
    return 0.0
