"""Intensity-modulation / direct-detection (IM/DD) schemes.

Every scheme maps bits to a sequence of *chips*: normalised drive levels x ∈ [0, 1] held for one
chip period (shape (n,) or (n, 3) for colour-shift keying). The receiver integrates each chip
(integrate-and-dump) and the scheme decides from those chip statistics, after calibrating levels
on a known training sequence. Closed-form BER uses the Gaussian chip statistics of the two drive
levels (μ0, σ0 at x = 0; μ1, σ1 at x = 1) and treats intermediate levels by interpolation.

Families
  pulse     OOK-NRZ, OOK-RZ, Manchester, L-PPM, MPPM, DPPM, DPIM, DH-PIM, VPPM, PWM
  level     M-PAM
  subcarrier SIM-BPSK, SIM-QPSK
  ofdm      DCO-OFDM, ACO-OFDM, Flip-OFDM
  colour    CSK (4 / 16 / 64 points, IEEE 802.15.7-style triangle in RGB drive space)
"""
from __future__ import annotations

from itertools import combinations
from math import log2

import numpy as np

from .constants import trapz
from scipy.special import erfc


def Qf(x):
    return 0.5 * erfc(np.asarray(x, float) / np.sqrt(2))


def bits_to_int(b, k):
    b = np.asarray(b, int).reshape(-1, k)
    return b @ (1 << np.arange(k - 1, -1, -1))


def int_to_bits(v, k):
    v = np.asarray(v, int)
    return ((v[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.int8).ravel()


def gray(v):
    v = np.asarray(v, int)
    return v ^ (v >> 1)


def igray(g):
    g = np.asarray(g, int).copy()
    s = g >> 1
    while np.any(s):
        g ^= s
        s >>= 1
    return g


def level_stats(y, labels):
    """Mean and std of chip statistics per distinct label (labels hashable per row)."""
    out = {}
    for lv in np.unique(labels):
        v = y[labels == lv]
        out[float(lv)] = (float(np.mean(v)), float(np.std(v)) if v.size > 1 else 0.0)
    return out


def _pois(n, lam):
    from scipy.stats import poisson
    return poisson.pmf(n, lam)


def poisson_ook(n0, n1):
    """Photon-counting OOK with the best integer threshold: min_T ½[P(N0 ≥ T) + P(N1 < T)]."""
    from scipy.stats import poisson
    T = np.arange(0, int(n1 + 10 * np.sqrt(n1 + 1) + 20))
    pe = 0.5 * (poisson.sf(T - 1, n0) + poisson.cdf(T - 1, n1))
    return float(np.min(pe))


def poisson_compare(n0, n1, L=2):
    """P(error) when the pulse chip (mean n1) must out-count L-1 empty chips (mean n0); ties broken
    at random (L-PPM symbol error; L = 2 gives Manchester / 2-PPM)."""
    from scipy.stats import poisson
    from scipy.special import comb as C
    nmax = int(n1 + n0 + 12 * np.sqrt(n1 + n0 + 1) + 30)
    n = np.arange(nmax + 1)
    p1, p0 = poisson.pmf(n, n1), poisson.pmf(n, n0)
    F0 = poisson.cdf(n - 1, n0)                        # P(N0 < n)
    win = np.zeros_like(p1)
    for k in range(L):
        win += C(L - 1, k) * p0 ** k * F0 ** (L - 1 - k) / (k + 1)
    return float(max(0.0, 1 - np.sum(p1 * win)))


def ml_threshold(m0, s0, m1, s1):
    return (m0 * s1 + m1 * s0) / (s0 + s1) if s0 + s1 > 0 else 0.5 * (m0 + m1)


def align_errors(tx, rx, k):
    """Bit errors between two symbol sequences allowing insertions / deletions (edit distance at
    symbol level: a substituted symbol costs its differing bits, an inserted or deleted symbol
    k/2). Variable-length schemes (DPPM, DPIM, DH-PIM) lose symbol sync on a chip error, so a plain
    bit-by-bit comparison would overstate their BER. Returns (bit_errors, n_bits)."""
    tx, rx = np.asarray(tx, int), np.asarray(rx, int)
    n, m = len(tx), len(rx)
    if n == 0:
        return 0, 0
    if m == 0:
        return n * k / 2, n * k
    pop = np.array([bin(i).count("1") for i in range(1 << k)])
    h = k / 2
    ramp = h * np.arange(m + 1)
    prev = ramp.copy()                                 # aligning tx[:0] with rx[:j]
    for i in range(1, n + 1):
        cur = np.empty(m + 1)
        cur[0] = i * h
        sub = prev[:-1] + pop[tx[i - 1] ^ rx]
        dele = prev[1:] + h
        cur[1:] = np.minimum(sub, dele)
        # insertions: cur[j] = min_l (cur[l] + (j - l) h) -> running minimum of cur - ramp
        cur = np.minimum.accumulate(cur - ramp) + ramp
        prev = cur
    return float(min(prev[m], n * k)), n * k


# =========================================================================== base
class Scheme:
    key = "base"
    family = "pulse"
    fixed = True               # fixed number of chips per symbol (camera-compatible)
    channels = 1
    k = 1                       # bits per symbol
    n = 1                       # chips per symbol (mean for variable-length schemes)

    def __init__(self, name=None):
        self.name = name or self.key

    # rates -----------------------------------------------------------------------------------
    @property
    def bits_per_chip(self):
        return self.k / self.n

    def chip_rate(self, bit_rate):
        return bit_rate / self.bits_per_chip

    @property
    def duty(self):
        """Mean drive level (average-to-peak optical power ratio)."""
        if "_duty" not in self.__dict__:
            rng = np.random.default_rng(7)
            c, _ = self.encode(rng.integers(0, 2, self.k * 400))
            self._duty = float(np.mean(c))
        return self._duty

    def n_bits(self, n):
        return max(self.k, (int(n) // self.k) * self.k)

    # interface -------------------------------------------------------------------------------
    def encode(self, bits):
        raise NotImplementedError

    def training(self):
        """Known chips for level calibration -> (chips, labels)."""
        rng = np.random.default_rng(99)
        x = np.tile([0.0, 1.0], 24)
        x = x[rng.permutation(x.size)]
        return x, x

    def calibrate(self, y, labels):
        st = level_stats(y, labels)
        m0, s0 = st.get(0.0, (0.0, 1e-9))
        m1, s1 = st.get(1.0, (1.0, 1e-9))
        return dict(m0=m0, s0=max(s0, 1e-15), m1=m1, s1=max(s1, 1e-15), thr=ml_threshold(m0, s0, m1, s1),
                    levels=st)

    def decode(self, y, cal, n_sym=None):
        """-> (symbols, bits, extra dict)"""
        raise NotImplementedError

    def ber(self, tx_syms, tx_bits, rx_syms, rx_bits):
        if self.fixed:
            n = min(len(tx_bits), len(rx_bits))
            e = int(np.sum(tx_bits[:n] != rx_bits[:n])) + (len(tx_bits) - n) // 2
            return e, len(tx_bits)
        return align_errors(tx_syms, rx_syms, self.k)

    def theory(self, m0, s0, m1, s1):
        """Closed-form (or approximate) BER from chip statistics; None if not available."""
        return None

    def theory_counts(self, n0, n1):
        """Exact photon-counting BER from mean counts per chip (None: use the Gaussian form)."""
        return None

    def describe(self):
        return f"{self.name}: {self.k} bit / {self.n:g} chips, duty {self.duty:.2f}"


def _thr_bits(y, cal):
    return (np.asarray(y) > cal["thr"]).astype(np.int8)


def _ook_theory(m0, s0, m1, s1):
    t = ml_threshold(m0, s0, m1, s1)
    return 0.5 * (Qf((t - m0) / s0) + Qf((m1 - t) / s1))


# =========================================================================== pulse schemes
class OOK(Scheme):
    key = "OOK-NRZ"

    def encode(self, bits):
        b = np.asarray(bits, int)
        return b.astype(float), b

    def decode(self, y, cal, n_sym=None):
        b = _thr_bits(y, cal)
        return b, b, {}

    def theory(self, m0, s0, m1, s1):
        return _ook_theory(m0, s0, m1, s1)

    def theory_counts(self, n0, n1):
        return poisson_ook(n0, n1)


class OOKRZ(Scheme):
    key = "OOK-RZ"
    n = 2

    def encode(self, bits):
        b = np.asarray(bits, int)
        c = np.zeros((b.size, 2)); c[:, 0] = b
        return c.ravel(), b

    def decode(self, y, cal, n_sym=None):
        y = np.asarray(y)[: (len(y) // 2) * 2].reshape(-1, 2)
        b = (y[:, 0] > cal["thr"]).astype(np.int8)
        return b, b, {}

    def theory(self, m0, s0, m1, s1):
        return _ook_theory(m0, s0, m1, s1)

    def theory_counts(self, n0, n1):
        return poisson_ook(n0, n1)


class Manchester(Scheme):
    key = "Manchester"
    n = 2

    def encode(self, bits):
        b = np.asarray(bits, int)
        c = np.stack([b, 1 - b], 1).astype(float)
        return c.ravel(), b

    def decode(self, y, cal, n_sym=None):
        y = np.asarray(y)[: (len(y) // 2) * 2].reshape(-1, 2)
        b = (y[:, 0] > y[:, 1]).astype(np.int8)
        return b, b, {"diff": y[:, 0] - y[:, 1]}

    def theory(self, m0, s0, m1, s1):
        return Qf((m1 - m0) / np.sqrt(s0 ** 2 + s1 ** 2))

    def theory_counts(self, n0, n1):
        return poisson_compare(n0, n1, 2)


class PPM(Scheme):
    key = "PPM"

    def __init__(self, L=4, name=None):
        self.L = int(L); self.k = int(log2(L)); self.n = self.L
        super().__init__(name or f"{self.L}-PPM")

    def encode(self, bits):
        s = bits_to_int(bits, self.k)
        c = np.zeros((s.size, self.L)); c[np.arange(s.size), s] = 1
        return c.ravel(), s

    def decode(self, y, cal, n_sym=None):
        y = np.asarray(y)[: (len(y) // self.L) * self.L].reshape(-1, self.L)
        s = np.argmax(y, 1)
        return s, int_to_bits(s, self.k), {}

    def theory(self, m0, s0, m1, s1):
        z = np.linspace(-8, 8, 801)
        phi = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
        q = Qf((m1 + s1 * z - m0) / s0)              # one empty chip beats the pulse chip
        Ps = trapz(phi * -np.expm1((self.L - 1) * np.log1p(-np.minimum(q, 1 - 1e-16))), z)
        return Ps * (self.L / 2) / (self.L - 1)

    def theory_counts(self, n0, n1):
        return poisson_compare(n0, n1, self.L) * (self.L / 2) / (self.L - 1)


class MPPM(Scheme):
    key = "MPPM"

    def __init__(self, n=8, w=2, name=None):
        self.n, self.w = int(n), int(w)
        self.combos = list(combinations(range(self.n), self.w))
        self.k = int(np.floor(log2(len(self.combos))))
        self.table = np.zeros((1 << self.k, self.n))
        for i in range(1 << self.k):
            self.table[i, list(self.combos[i])] = 1
        super().__init__(name or f"MPPM ({self.n},{self.w})")

    def encode(self, bits):
        s = bits_to_int(bits, self.k)
        return self.table[s].ravel(), s

    def decode(self, y, cal, n_sym=None):
        y = np.asarray(y)[: (len(y) // self.n) * self.n].reshape(-1, self.n)
        corr = y @ self.table.T                          # ML for equal-energy codewords
        s = np.argmax(corr, 1)
        return s, int_to_bits(s, self.k), {}

    def theory(self, m0, s0, m1, s1):
        # nearest codewords differ by moving one pulse: metric difference y_i - y_j
        Ps = min(1.0, self.w * (self.n - self.w) * Qf((m1 - m0) / np.sqrt(s0 ** 2 + s1 ** 2)))
        return min(0.5, Ps * (1 << (self.k - 1)) / ((1 << self.k) - 1))


class DPPM(Scheme):
    """Differential PPM: the PPM symbol truncated after its pulse (m empty chips then a pulse)."""
    key = "DPPM"
    fixed = False

    def __init__(self, L=8, name=None):
        self.L = int(L); self.k = int(log2(L)); self.n = (self.L + 1) / 2
        super().__init__(name or f"{self.L}-DPPM")

    def encode(self, bits):
        s = bits_to_int(bits, self.k)
        out = []
        for v in s:
            out.extend([0.0] * int(v)); out.append(1.0)
        return np.array(out), s

    def decode(self, y, cal, n_sym=None):
        b = np.asarray(y) > cal["thr"]
        syms, run = [], 0
        for on in b:
            if on:
                syms.append(min(run, self.L - 1)); run = 0
            else:
                run += 1
                if run > self.L - 1:                     # missing pulse: emit and resync
                    syms.append(self.L - 1); run = 0
        s = np.array(syms, int)
        return s, int_to_bits(s, self.k) if s.size else np.zeros(0, np.int8), {}

    def theory(self, m0, s0, m1, s1):
        t = ml_threshold(m0, s0, m1, s1)
        d = 1 / self.n
        pse = d * Qf((m1 - t) / s1) + (1 - d) * Qf((t - m0) / s0)
        return min(0.5, pse * self.n)


class DPIM(Scheme):
    """Digital pulse interval modulation: a pulse chip followed by g guard chips and m empty chips."""
    key = "DPIM"
    fixed = False

    def __init__(self, L=8, guard=1, name=None):
        self.L, self.g = int(L), int(guard); self.k = int(log2(L))
        self.n = 1 + self.g + (self.L - 1) / 2
        super().__init__(name or f"{self.L}-DPIM")

    def encode(self, bits):
        s = bits_to_int(bits, self.k)
        out = []
        for v in s:
            out.append(1.0); out.extend([0.0] * (self.g + int(v)))
        return np.array(out), s

    def decode(self, y, cal, n_sym=None):
        b = np.asarray(y) > cal["thr"]
        p = np.nonzero(b)[0]
        if p.size == 0:
            return np.zeros(0, int), np.zeros(0, np.int8), {}
        gaps = np.diff(np.concatenate([p, [len(b)]])) - 1 - self.g
        s = np.clip(gaps, 0, self.L - 1)
        if n_sym is not None and gaps[-1] > self.L - 1:       # trailing silence
            s = s[:-1] if s.size > n_sym else s
        return s, int_to_bits(s, self.k), {}

    def theory(self, m0, s0, m1, s1):
        t = ml_threshold(m0, s0, m1, s1)
        d = 1 / self.n
        pse = d * Qf((m1 - t) / s1) + (1 - d) * Qf((t - m0) / s0)
        return min(0.5, pse * self.n)


class DHPIM(Scheme):
    """Dual-header PIM (α = 2): H1 = [1,0,0] then m empty chips for m < L/2,
    H2 = [1,1,0] then L-1-m empty chips otherwise."""
    key = "DH-PIM"
    fixed = False

    def __init__(self, L=8, name=None):
        self.L = int(L); self.k = int(log2(L))
        self.n = 3 + (self.L / 2 - 1) / 2
        super().__init__(name or f"{self.L}-DH-PIM")

    def encode(self, bits):
        s = bits_to_int(bits, self.k)
        out = []
        h = self.L // 2
        for v in s:
            if v < h:
                out.extend([1.0, 0.0, 0.0]); out.extend([0.0] * int(v))
            else:
                out.extend([1.0, 1.0, 0.0]); out.extend([0.0] * int(self.L - 1 - v))
        return np.array(out), s

    def decode(self, y, cal, n_sym=None):
        b = (np.asarray(y) > cal["thr"]).astype(int)
        syms, i, N, h = [], 0, len(b), self.L // 2
        while i < N and not b[i]:
            i += 1
        while i < N:
            two = i + 1 < N and b[i + 1]
            j = i + (2 if two else 1)
            while j < N and not b[j]:
                j += 1
            gap = j - i - 3                               # empty chips after the header
            v = (self.L - 1 - max(gap, 0)) if two else max(gap, 0)
            syms.append(int(np.clip(v, h if two else 0, self.L - 1 if two else h - 1)))
            i = j
        s = np.array(syms, int)
        return s, int_to_bits(s, self.k) if s.size else np.zeros(0, np.int8), {}

    def theory(self, m0, s0, m1, s1):
        t = ml_threshold(m0, s0, m1, s1)
        d = self.duty
        pse = d * Qf((m1 - t) / s1) + (1 - d) * Qf((t - m0) / s0)
        return min(0.5, pse * self.n)


class VPPM(Scheme):
    """Variable PPM (IEEE 802.15.7 dimming): a pulse of width d·N at the start (0) or end (1) of
    each bit period of N chips."""
    key = "VPPM"

    def __init__(self, duty=0.5, N=8, name=None):
        self.N = int(N); self.d = float(np.clip(duty, 1 / N, 1 - 1 / N)); self.n = self.N
        self.w = max(1, int(round(self.d * self.N)))
        self.t0 = np.zeros(self.N); self.t0[: self.w] = 1
        self.t1 = np.zeros(self.N); self.t1[self.N - self.w:] = 1
        super().__init__(name or f"VPPM ({int(round(self.d * 100))} %)")

    def encode(self, bits):
        b = np.asarray(bits, int)
        c = np.where(b[:, None] == 1, self.t1[None, :], self.t0[None, :])
        return c.ravel(), b

    def decode(self, y, cal, n_sym=None):
        y = np.asarray(y)[: (len(y) // self.N) * self.N].reshape(-1, self.N)
        m = y @ (self.t1 - self.t0)
        b = (m > 0).astype(np.int8)
        return b, b, {"diff": m}

    def theory(self, m0, s0, m1, s1):
        nd = int(np.sum(self.t1 > self.t0))
        return Qf(np.sqrt(nd) * (m1 - m0) / np.sqrt(s0 ** 2 + s1 ** 2))

    def theory_counts(self, n0, n1):
        nd = int(np.sum(self.t1 > self.t0))
        return poisson_compare(nd * n0, nd * n1, 2)


class PWM(Scheme):
    """M-ary pulse-width modulation: symbol m is a pulse of m+1 chips in an M-chip frame."""
    key = "PWM"

    def __init__(self, M=4, name=None):
        self.M = int(M); self.k = int(log2(M)); self.n = self.M
        super().__init__(name or f"{self.M}-PWM")

    def encode(self, bits):
        s = gray(bits_to_int(bits, self.k))
        c = (np.arange(self.M)[None, :] <= s[:, None]).astype(float)
        return c.ravel(), bits_to_int(bits, self.k)

    def decode(self, y, cal, n_sym=None):
        y = np.asarray(y)[: (len(y) // self.M) * self.M].reshape(-1, self.M)
        z = (y - cal["thr"])
        metric = np.cumsum(z, 1)
        g = np.argmax(metric, 1)
        s = igray(g)
        return s, int_to_bits(s, self.k), {}

    def theory(self, m0, s0, m1, s1):
        Pse = 2 * (self.M - 1) / self.M * _ook_theory(m0, s0, m1, s1)
        return min(0.5, Pse / self.k)


# =========================================================================== multilevel
class PAM(Scheme):
    key = "PAM"
    family = "level"

    def __init__(self, M=4, name=None):
        self.M = int(M); self.k = int(log2(M)); self.n = 1
        super().__init__(name or f"{self.M}-PAM")

    def encode(self, bits):
        s = bits_to_int(bits, self.k)
        return gray(s) / (self.M - 1), s

    def training(self):
        rng = np.random.default_rng(98)
        x = np.repeat(np.arange(self.M) / (self.M - 1), 12)
        x = x[rng.permutation(x.size)]
        return x, x

    def calibrate(self, y, labels):
        cal = super().calibrate(y, labels)
        lv = sorted(cal["levels"].items())
        cal["means"] = np.array([m for _, (m, s) in lv])
        cal["stds"] = np.array([max(s, 1e-15) for _, (m, s) in lv])
        mu, sd = cal["means"], cal["stds"]
        cal["thresholds"] = np.array([ml_threshold(mu[i], sd[i], mu[i + 1], sd[i + 1]) for i in range(self.M - 1)])
        return cal

    def decode(self, y, cal, n_sym=None):
        g = np.searchsorted(cal["thresholds"], np.asarray(y))
        s = igray(g)
        return s, int_to_bits(s, self.k), {}

    def theory(self, m0, s0, m1, s1):
        lv = np.arange(self.M) / (self.M - 1)
        mu = m0 + lv * (m1 - m0)
        sd = np.sqrt(s0 ** 2 + lv * (s1 ** 2 - s0 ** 2))
        pe = 0.0
        for i in range(self.M - 1):
            t = ml_threshold(mu[i], sd[i], mu[i + 1], sd[i + 1])
            pe += Qf((t - mu[i]) / sd[i]) + Qf((mu[i + 1] - t) / sd[i + 1])
        return pe / self.M / self.k


# =========================================================================== subcarrier
class SIM(Scheme):
    """Subcarrier intensity modulation: x = 1/2 + m/2 · Re{s e^{j2π f n}}, PSK symbol s,
    `cycles` subcarrier cycles per symbol of `n` chips; coherent demodulation with a pilot phase."""
    key = "SIM"
    family = "subcarrier"

    def __init__(self, order=2, cycles=2, chips=8, depth=0.9, name=None):
        self.order, self.cyc, self.n, self.m = int(order), int(cycles), int(chips), float(depth)
        self.k = int(log2(order))
        self.ph = np.exp(2j * np.pi * self.cyc * (np.arange(self.n) + 0.5) / self.n)
        super().__init__(name or ("SIM-BPSK" if order == 2 else "SIM-QPSK"))

    def _points(self, s):
        if self.order == 2:
            return np.where(s == 1, -1.0 + 0j, 1.0 + 0j)
        g = gray(s)
        return np.exp(1j * (np.pi / 4 + np.pi / 2 * g))

    def encode(self, bits):
        s = bits_to_int(bits, self.k)
        a = self._points(s)
        c = 0.5 + 0.5 * self.m * np.real(a[:, None] * self.ph[None, :])
        return c.ravel(), s

    def training(self):
        s = np.zeros(16, int)
        c, _ = self.encode(int_to_bits(s, self.k))
        return c, np.repeat(np.arange(16), self.n)

    def calibrate(self, y, labels):
        y = np.asarray(y).reshape(-1, self.n)
        dc = float(np.mean(y))
        z = (y - dc) @ np.conj(self.ph) * 2 / self.n
        ref = np.mean(z) / self._points(np.zeros(1, int))[0]   # known pilot symbol
        noise = np.std(z - ref)
        return dict(dc=dc, ref=ref, sigma=noise, thr=dc)

    def decode(self, y, cal, n_sym=None):
        y = np.asarray(y)[: (len(y) // self.n) * self.n].reshape(-1, self.n)
        z = (y - cal["dc"]) @ np.conj(self.ph) * 2 / self.n / cal["ref"]
        if self.order == 2:
            s = (z.real < 0).astype(int)
        else:
            q = np.mod(np.round((np.angle(z) - np.pi / 4) / (np.pi / 2)), 4).astype(int)
            s = igray(q)
        return s, int_to_bits(s, self.k), {"points": z}

    def theory(self, m0, s0, m1, s1):
        a = 0.5 * self.m * (m1 - m0)
        sig = np.sqrt(0.5 * (s0 ** 2 + s1 ** 2))
        snr = a * np.sqrt(self.n / 2) / sig
        return Qf(snr) if self.order == 2 else Qf(snr / np.sqrt(2))


# =========================================================================== OFDM
def qam_points(M):
    if M == 2:
        return np.array([-1.0 + 0j, 1.0 + 0j])
    m = int(round(np.sqrt(M)))
    k2 = int(log2(m))
    lv = 2 * np.arange(m) - (m - 1)
    pts = np.zeros(M, complex)
    for i in range(M):
        hi, lo = i >> k2, i & (m - 1)
        pts[i] = lv[igray(np.array([hi]))[0]] + 1j * lv[igray(np.array([lo]))[0]]
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def qam_theory(M, snr):
    """Gray-coded square QAM BER at per-symbol SNR (Es/N0)."""
    snr = np.asarray(snr, float)
    if M == 2:
        return Qf(np.sqrt(2 * snr))
    k = log2(M)
    return np.minimum(0.5, 4 / k * (1 - 1 / np.sqrt(M)) * Qf(np.sqrt(3 * snr / (M - 1))))


class OFDM(Scheme):
    """Optical OFDM. variant 'DCO' (DC bias of `bias` σ, clipped to [0, 1]), 'ACO' (odd subcarriers,
    negative half clipped), 'Flip' (positive and flipped negative halves in two blocks)."""
    key = "OFDM"
    family = "ofdm"

    def __init__(self, variant="DCO", N=64, M=16, cp=8, bias=2.0, clip=3.0, name=None):
        self.variant, self.N, self.M, self.cp = variant, int(N), int(M), int(cp)
        self.bias, self.clip = float(bias), float(clip)
        if variant == "ACO":
            self.data = np.arange(1, self.N // 2, 2)
        else:
            self.data = np.arange(1, self.N // 2)
        self.kq = int(log2(M))
        self.k = self.kq * self.data.size
        self.blocks = 2 if variant == "Flip" else 1
        self.n = (self.N + self.cp) * self.blocks
        self.pts = qam_points(M)
        # time-domain std of a unit-power-per-carrier Hermitian frame (orthonormal IFFT)
        self.sigma = np.sqrt(2 * self.data.size / self.N)
        super().__init__(name or f"{variant}-OFDM")
        # clipping distortion: noiseless loop-back error vector magnitude per carrier
        rng = np.random.default_rng(5)
        X = self.pts[rng.integers(0, M, (40, self.data.size))]
        tr, _ = self.training()
        y = self._shape(self._frame(X)).ravel()
        Ht = np.mean(self._fft(tr) / self._pilot, 0)
        self.clip_evm2 = float(np.mean(np.abs(self._fft(y) / Ht - X) ** 2))

    def _frame(self, X):
        F = np.zeros((X.shape[0], self.N), complex)
        F[:, self.data] = X
        F[:, self.N - self.data] = np.conj(X)
        return np.fft.ifft(F, axis=1).real * np.sqrt(self.N)

    def _shape(self, x):
        s = self.sigma
        if self.variant == "DCO":
            y = (x + self.bias * s) / (self.bias * s + self.clip * s)
            y = np.clip(y, 0, 1)
            blocks = [y]
        elif self.variant == "ACO":
            blocks = [np.clip(x / (self.clip * s), 0, 1)]
        else:
            blocks = [np.clip(x / (self.clip * s), 0, 1), np.clip(-x / (self.clip * s), 0, 1)]
        out = []
        for b in blocks:
            out.append(np.concatenate([b[:, self.N - self.cp:], b], 1))
        return np.concatenate(out, 1) if len(out) > 1 else out[0]

    def encode(self, bits):
        b = np.asarray(bits, int)
        s = bits_to_int(b, self.kq).reshape(-1, self.data.size)
        x = self._frame(self.pts[s])
        return self._shape(x).ravel(), s.ravel()

    def n_bits(self, n):
        return max(self.k, (int(n) // self.k) * self.k)

    def training(self):
        rng = np.random.default_rng(97)
        s = rng.integers(0, 4, (4, self.data.size))
        self._pilot = qam_points(4)[s]
        x = self._frame(self._pilot)
        return self._shape(x).ravel(), np.repeat(np.arange(4), self.n)

    def _fft(self, y):
        y = np.asarray(y)[: (len(y) // self.n) * self.n].reshape(-1, self.blocks, self.N + self.cp)[:, :, self.cp:]
        if self.variant == "Flip":
            t = y[:, 0] - y[:, 1]
        else:
            t = y[:, 0]
        return np.fft.fft(t - (np.mean(t) if self.variant != "Flip" else 0), axis=1)[:, self.data] / np.sqrt(self.N)

    def calibrate(self, y, labels):
        Y = self._fft(y)
        H = np.mean(Y / self._pilot, 0)
        noise = np.mean(np.abs(Y / H - self._pilot) ** 2)
        return dict(H=H, noise=float(noise), thr=0.0)

    def decode(self, y, cal, n_sym=None):
        Y = self._fft(y) / cal["H"]
        d = np.abs(Y[..., None] - self.pts[None, None, :])
        s = np.argmin(d, 2).ravel()
        return s, int_to_bits(s, self.kq), {"points": Y.ravel(), "per_carrier": Y}

    def snr_from_chips(self, m0, s0, m1, s1):
        """Per-subcarrier Es/N0: carrier amplitude in chip units over the chip noise (orthonormal FFT)."""
        amp = (m1 - m0) / self.sigma                   # unit-power carriers have time std `sigma`
        d = self.duty
        sig2 = d * s1 ** 2 + (1 - d) * s0 ** 2
        if self.variant == "DCO":
            a = amp / (self.bias + self.clip)
        elif self.variant == "ACO":
            a = amp / self.clip / 2                    # clipping halves the odd carriers
        else:
            a = amp / self.clip
            sig2 = 2 * sig2                            # difference of two blocks doubles the noise
        return a ** 2 / sig2

    def theory(self, m0, s0, m1, s1):
        snr = self.snr_from_chips(m0, s0, m1, s1)
        snr = 1 / (1 / max(snr, 1e-30) + self.clip_evm2)          # plus clipping distortion
        return float(qam_theory(self.M, snr))


# =========================================================================== colour-shift keying
def csk_points(M):
    """Barycentric (R, G, B) drive vectors: centroids of the n² sub-triangles of the colour
    triangle (n = 2, 4, 8 for 4, 16, 64 points); each sums to 1 (constant total power)."""
    n = int(round(np.sqrt(M)))
    pts = []
    for i in range(n):
        for j in range(n - i):
            a = np.array([i + 1 / 3, j + 1 / 3, n - i - j - 2 / 3]); pts.append(a / n)      # upward
            if i + j < n - 1:
                b = np.array([i + 2 / 3, j + 2 / 3, n - i - j - 4 / 3]); pts.append(b / n)   # downward
    P = np.array(pts)
    return P[:M]


class CSK(Scheme):
    key = "CSK"
    family = "colour"
    channels = 3

    def __init__(self, M=4, name=None):
        self.M = int(M); self.k = int(log2(M)); self.n = 1
        self.P = csk_points(M)
        super().__init__(name or f"{self.M}-CSK")

    @property
    def duty(self):
        return 1.0 / 3.0

    def encode(self, bits):
        s = bits_to_int(bits, self.k)
        return self.P[s], s

    def training(self):
        s = np.repeat(np.arange(self.M), 8)
        return self.P[s], s

    def calibrate(self, y, labels):
        y = np.asarray(y).reshape(-1, 3)
        means = np.array([y[labels == m].mean(0) for m in range(self.M)])
        resid = np.concatenate([y[labels == m] - means[m] for m in range(self.M)])
        cov = np.cov(resid.T) + 1e-30 * np.eye(3)
        bg = np.zeros(3)
        return dict(means=means, cov=cov, icov=np.linalg.pinv(cov), thr=0.0, bg=bg)

    def decode(self, y, cal, n_sym=None):
        y = np.asarray(y).reshape(-1, 3)
        d = y[:, None, :] - cal["means"][None, :, :]
        md = np.einsum("nmi,ij,nmj->nm", d, cal["icov"], d)
        s = np.argmin(md, 1)
        return s, int_to_bits(s, self.k), {"rgb": y}

    def theory_colour(self, cols):
        """Union bound with per-colour chip statistics cols = [(m0, s0, m1, s1)] for R, G, B:
        each colour has its own water attenuation, so the received constellation is skewed."""
        g = np.array([c[2] - c[0] for c in cols])
        sg = np.array([np.sqrt(0.5 * (c[1] ** 2 + c[3] ** 2)) for c in cols])
        Y = self.P * (g / sg)[None, :]
        tot = 0.0
        for i in range(self.M):
            for j in range(self.M):
                if i != j:
                    h = bin(i ^ j).count("1")
                    tot += h * Qf(np.linalg.norm(Y[i] - Y[j]) / 2)
        return float(min(0.5, tot / (self.M * self.k)))

    def theory(self, m0, s0, m1, s1):
        P = self.P
        dmin = min(np.linalg.norm(P[i] - P[j]) for i in range(self.M) for j in range(i + 1, self.M))
        sig = np.sqrt(0.5 * (s0 ** 2 + s1 ** 2))
        Ps = min(1.0, 3 * Qf(dmin * (m1 - m0) / (2 * sig)))
        return min(0.5, Ps / self.k)


def chromaticity(rgb):
    """Map RGB drive / received vectors to 2-D triangle coordinates for constellation plots."""
    rgb = np.asarray(rgb, float)
    s = rgb.sum(-1, keepdims=True)
    w = rgb / np.where(np.abs(s) < 1e-30, 1, s)
    V = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2], [0.0, 0.0]])        # R, G, B vertices
    return w @ V


# =========================================================================== registry
SCHEMES = {
    "OOK-NRZ": lambda: OOK(),
    "OOK-RZ": lambda: OOKRZ(),
    "Manchester": lambda: Manchester(),
    "2-PPM": lambda: PPM(2),
    "4-PPM": lambda: PPM(4),
    "8-PPM": lambda: PPM(8),
    "16-PPM": lambda: PPM(16),
    "MPPM (8,2)": lambda: MPPM(8, 2),
    "MPPM (12,3)": lambda: MPPM(12, 3),
    "4-DPPM": lambda: DPPM(4),
    "8-DPPM": lambda: DPPM(8),
    "4-DPIM": lambda: DPIM(4),
    "8-DPIM": lambda: DPIM(8),
    "16-DPIM": lambda: DPIM(16),
    "8-DH-PIM": lambda: DHPIM(8),
    "VPPM (50 %)": lambda: VPPM(0.5),
    "VPPM (25 %)": lambda: VPPM(0.25),
    "4-PWM": lambda: PWM(4),
    "4-PAM": lambda: PAM(4),
    "8-PAM": lambda: PAM(8),
    "SIM-BPSK": lambda: SIM(2),
    "SIM-QPSK": lambda: SIM(4),
    "DCO-OFDM": lambda: OFDM("DCO", 64, 16),
    "ACO-OFDM": lambda: OFDM("ACO", 64, 16),
    "Flip-OFDM": lambda: OFDM("Flip", 64, 16),
    "4-CSK": lambda: CSK(4),
    "16-CSK": lambda: CSK(16),
}

FAMILIES = {
    "On-off": ["OOK-NRZ", "OOK-RZ", "Manchester"],
    "Pulse position": ["2-PPM", "4-PPM", "8-PPM", "16-PPM", "MPPM (8,2)", "MPPM (12,3)", "4-DPPM", "8-DPPM"],
    "Pulse interval / width": ["4-DPIM", "8-DPIM", "16-DPIM", "8-DH-PIM", "VPPM (50 %)", "VPPM (25 %)", "4-PWM"],
    "Multilevel / subcarrier": ["4-PAM", "8-PAM", "SIM-BPSK", "SIM-QPSK"],
    "OFDM": ["DCO-OFDM", "ACO-OFDM", "Flip-OFDM"],
    "Colour": ["4-CSK", "16-CSK"],
}


def make(name, **kw):
    """Scheme by registry name, or a family with parameters: make('PPM', L=32),
    make('OFDM', variant='ACO', N=128, M=64), make('CSK', M=64)."""
    fam = {"OOK": OOK, "OOK-NRZ": OOK, "OOK-RZ": OOKRZ, "Manchester": Manchester, "PPM": PPM, "MPPM": MPPM,
           "DPPM": DPPM, "DPIM": DPIM, "DH-PIM": DHPIM, "VPPM": VPPM, "PWM": PWM, "PAM": PAM, "SIM": SIM,
           "OFDM": OFDM, "CSK": CSK}
    if kw or name in fam and name not in SCHEMES:
        return fam[name](**kw)
    if name in SCHEMES:
        return SCHEMES[name]()
    raise KeyError(f"unknown scheme {name!r}")
