"""Command line interface.

    uwoc list                                       built-in scenarios
    uwoc schemes                                    modulation schemes (photodiode and camera)
    uwoc water "Jerlov 3C" --lam 520                inherent optical properties
    uwoc channel "Coastal, 10 m" --model mc         channel gain, delay spread, bandwidth
    uwoc link "Coastal, 10 m" --scheme 8-PPM        one photodiode frame end to end
    uwoc camera "Diver to diver" --frames 30        one camera capture end to end
    uwoc budget "Clear ocean"                       link budget and margin
    uwoc ber "Clear ocean" --schemes OOK-NRZ 4-PPM --ranges 20:120:6 --mc
    uwoc gallery docs/figures                       regenerate the README figures
Scenario names may be abbreviated to a unique prefix.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .core import camera as C, link, schemes as S
from .core.scenarios import SCENARIOS
from .core.water import WATER_TYPES, custom_water, water_from_name


def _scenario(name):
    if name in SCENARIOS:
        return SCENARIOS[name].copy()
    hits = [k for k in SCENARIOS if k.lower().startswith(name.lower())]
    if len(hits) == 1:
        return SCENARIOS[hits[0]].copy()
    raise SystemExit(f"unknown scenario {name!r}; try: uwoc list")


def _apply(sc, a):
    if getattr(a, "range", None) is not None:
        sc.range_m = a.range
    if getattr(a, "water", None):
        sc.water = water_from_name(a.water)
    if getattr(a, "a", None) is not None and getattr(a, "b", None) is not None:
        sc.water = custom_water(a.a, a.b, sc.wavelength)
    if getattr(a, "rate", None) is not None:
        sc.bit_rate = a.rate
    return sc


def _print(d, as_json):
    if as_json:
        print(json.dumps({k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else str(v))
                          for k, v in d.items()}, indent=2))
    else:
        for k, v in d.items():
            print(f"{k:28s} {v:.6g}" if isinstance(v, (float, np.floating)) else f"{k:28s} {v}")


def cmd_list(a):
    for name, sc in SCENARIOS.items():
        what = sc.scheme if sc.receiver == "photodiode" else f"camera: {sc.camera_scheme}"
        print(f"{name:42s} {sc.water.name:16s} {what:22s} {sc.description}")


def cmd_schemes(a):
    for fam, names in S.FAMILIES.items():
        print(f"[{fam}]")
        for n in names:
            print("  " + S.make(n).describe())
    print("[camera]")
    for n in C.CAMERA_SCHEMES:
        print("  " + n)


def cmd_water(a):
    w = water_from_name(a.name) if a.a is None else custom_water(a.a, a.b, a.lam)
    d = dict(water=w.name, wavelength_nm=a.lam, **w.iops(a.lam), best_wavelength_nm=w.best_wavelength(),
             phase_g=w.phase_function(a.lam).g, phase_B=w.phase_function(a.lam).B)
    _print(d, a.json)


def cmd_channel(a):
    sc = _apply(_scenario(a.scenario), a)
    ch = sc.channel(model=a.model)
    db = lambda x: 10 * np.log10(max(x, 1e-300))
    _print(dict(scenario=sc.name, model=a.model, range_m=sc.range_m, gain_dB=db(ch.gain),
                ballistic_dB=db(ch.gain_ballistic), scattered_dB=db(ch.gain_scattered),
                mean_delay_ns=ch.mean_delay * 1e9, rms_delay_ns=ch.rms_delay * 1e9,
                bandwidth_3dB_MHz=ch.bandwidth_3db() / 1e6), a.json)


def cmd_link(a):
    sc = _apply(_scenario(a.scenario), a)
    sch = S.make(a.scheme or sc.scheme)
    r = link.simulate(sc, sch, n_bits=a.bits, sps=a.sps, seed=a.seed)
    b = link.link_budget(sc, sch)
    _print(dict(scenario=sc.name, scheme=sch.name, range_m=sc.range_m, bit_rate=r.info["bit_rate"],
                chip_rate=r.chip_rate, Pr_dBm=b["Pr_dBm"], Pavg_dBm=b["Pavg_dBm"], Pb_dBm=b["Pb_dBm"], Q=b["Q"],
                ber_mc=r.ber, errors=int(r.errors), bits=r.n_bits, ber_theory=b["ber"]), a.json)
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        from .examples.gallery import fig_photodiode
        fig_photodiode(sc, sch.name, a.plot)
        print(f"saved {a.plot}")


def cmd_camera(a):
    sc = _apply(_scenario(a.scenario), a)
    name = a.scheme or sc.camera_scheme
    r = C.simulate(sc, name, n_frames=a.frames, seed=a.seed)
    _print(dict(scenario=sc.name, scheme=name, shutter=sc.camera.shutter, frames=len(r.frame_t),
                ber=r.ber, errors=int(r.errors), bits=r.n_bits, ber_theory=C.ber_theory(sc, name),
                goodput_bps=r.goodput, image_radius_px=r.info["r_img"], peak_dn=r.info["peak_dn"],
                chips_observed=r.info.get("observed_fraction", 1.0)), a.json)
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        from .examples.gallery import fig_camera
        fig_camera(sc, name, a.plot)
        print(f"saved {a.plot}")


def cmd_budget(a):
    sc = _apply(_scenario(a.scenario), a)
    sch = S.make(a.scheme or sc.scheme)
    b = link.link_budget(sc, sch)
    for k, v in b["terms"]:
        print(f"{k:28s} {v:10.2f} dB" if k != "Tx peak power" else f"{k:28s} {v:10.2f} dBm")
    req = link.required_power_dbm(sc, sch, a.target)
    print(f"{'received peak power':28s} {b['Pr_dBm']:10.2f} dBm")
    print(f"{'background power':28s} {b['Pb_dBm']:10.2f} dBm")
    print(f"{'required for BER %.0e' % a.target:28s} {req:10.2f} dBm")
    print(f"{'margin':28s} {b['Pr_dBm'] - req:10.2f} dB")
    print(f"{'BER':28s} {b['ber']:10.3g}")
    print(f"{'max range at target':28s} {link.max_range(sc, sch, a.target):10.2f} m")


def cmd_ber(a):
    sc = _apply(_scenario(a.scenario), a)
    lo, hi, n = (float(x) for x in a.ranges.split(":"))
    Rs = np.geomspace(lo, hi, int(n))
    names = a.schemes or [sc.scheme]
    cam = sc.receiver == "camera" if a.receiver is None else a.receiver == "camera"
    print("range_m  " + "  ".join(f"{n:>14s}" for n in names))
    cols = []
    for nme in names:
        if cam:
            th = np.array([C.ber_theory(sc.copy(range_m=float(L)), nme) for L in Rs])
            mc = (np.array([C.simulate(sc.copy(range_m=float(L)), nme, n_frames=a.frames, seed=i).ber
                            for i, L in enumerate(Rs)]) if a.mc else None)
        else:
            th = link.ber_vs_range(sc, S.make(nme), Rs)
            mc = link.monte_carlo_ber(sc, nme, ranges=Rs, n_bits=a.bits) if a.mc else None
        cols.append((th, mc))
    for i, L in enumerate(Rs):
        cells = []
        for th, mc in cols:
            cells.append(f"{th[i]:.2e}" + (f"|{mc[i]:.2e}" if mc is not None else ""))
        print(f"{L:7.2f}  " + "  ".join(f"{c:>14s}" for c in cells))
    if a.mc:
        print("(closed form|Monte Carlo)")


def cmd_gallery(a):
    from .examples.gallery import main as gallery
    gallery([a.out])


def main(argv=None):
    p = argparse.ArgumentParser(prog="uwoc", description="Underwater optical communications toolbox")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list built-in scenarios").set_defaults(fn=cmd_list)
    sub.add_parser("schemes", help="list modulation schemes").set_defaults(fn=cmd_schemes)
    w = sub.add_parser("water", help="inherent optical properties of a water type")
    w.add_argument("name", nargs="?", default="Jerlov II", help=", ".join(WATER_TYPES))
    w.add_argument("--lam", type=float, default=470.0); w.add_argument("-a", type=float); w.add_argument("-b", type=float)
    w.add_argument("--json", action="store_true"); w.set_defaults(fn=cmd_water)

    def common(s, scheme=True):
        s.add_argument("scenario")
        s.add_argument("--range", type=float, help="link range (m)")
        s.add_argument("--water", help="water type, e.g. 'Jerlov 3C'")
        s.add_argument("-a", type=float, help="custom absorption (1/m), with -b")
        s.add_argument("-b", type=float, help="custom scattering (1/m), with -a")
        s.add_argument("--rate", type=float, help="bit rate (b/s)")
        if scheme:
            s.add_argument("--scheme")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--json", action="store_true")

    c = sub.add_parser("channel", help="channel gain, impulse response statistics")
    common(c, False); c.add_argument("--model", choices=["beer", "effective", "mc"], default="effective")
    c.set_defaults(fn=cmd_channel)
    l = sub.add_parser("link", help="simulate one photodiode frame")
    common(l); l.add_argument("--bits", type=int, default=4000); l.add_argument("--sps", type=int, default=8)
    l.add_argument("--plot", help="save an intensity-vs-time figure"); l.set_defaults(fn=cmd_link)
    k = sub.add_parser("camera", help="simulate one camera capture")
    common(k); k.add_argument("--frames", type=int, default=16)
    k.add_argument("--plot", help="save a frame-sequence figure"); k.set_defaults(fn=cmd_camera)
    b = sub.add_parser("budget", help="link budget and margin")
    common(b); b.add_argument("--target", type=float, default=1e-6); b.set_defaults(fn=cmd_budget)
    e = sub.add_parser("ber", help="BER versus range (closed form, optional Monte Carlo)")
    common(e, False); e.add_argument("--schemes", nargs="+"); e.add_argument("--ranges", default="5:60:8",
                                                                            help="from:to:points (m)")
    e.add_argument("--receiver", choices=["photodiode", "camera"])
    e.add_argument("--mc", action="store_true"); e.add_argument("--bits", type=int, default=4000)
    e.add_argument("--frames", type=int, default=16)
    e.set_defaults(fn=cmd_ber)
    g = sub.add_parser("gallery", help="regenerate README figures")
    g.add_argument("out", nargs="?", default="docs/figures"); g.set_defaults(fn=cmd_gallery)
    a = p.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
