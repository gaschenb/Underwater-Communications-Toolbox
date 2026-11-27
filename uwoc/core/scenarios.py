"""Link scenarios: water + geometry + transmitter + receiver (photodiode or camera) + impairments."""
from __future__ import annotations

from dataclasses import dataclass, field, replace


from .channel import Ambient, Geometry, Pointing, Turbulence, channel
from .devices import CAMERAS, DETECTORS, TRANSMITTERS, Camera, Detector, ReceiverOptics, Transmitter
from .water import Water, water_from_name


@dataclass
class Scenario:
    name: str = "custom"
    description: str = ""
    water: Water = field(default_factory=lambda: water_from_name("Jerlov II"))
    range_m: float = 20.0
    tx: Transmitter = field(default_factory=lambda: TRANSMITTERS["Blue LED, 1 W"].copy())
    optics: ReceiverOptics = field(default_factory=ReceiverOptics)
    detector: Detector = field(default_factory=lambda: DETECTORS["PIN photodiode"].copy())
    camera: Camera = field(default_factory=lambda: CAMERAS["Rolling-shutter CMOS (mono)"].copy())
    receiver: str = "photodiode"           # 'photodiode' or 'camera'
    turbulence: Turbulence = field(default_factory=Turbulence)
    pointing: Pointing = field(default_factory=Pointing)
    ambient: Ambient = field(default_factory=Ambient)
    scheme: str = "OOK-NRZ"
    camera_scheme: str = "Manchester"
    bit_rate: float = 1e6
    camera_chip_rate: float = 2000.0       # chips/s for chip-based schemes on a camera
    camera_carrier_hz: float = 240.0       # UFSOOK / UPSOOK blink frequency, RS-FSK lowest tone
    frame_phase: float = 0.0               # camera frame start relative to the Tx clock (fraction of a frame)
    n_frames: int = 8
    channel_model: str = "effective"
    mc_photons: int = 20000

    def copy(self, **kw):
        out = replace(self, water=replace(self.water), tx=replace(self.tx), optics=replace(self.optics),
                      detector=replace(self.detector), camera=replace(self.camera),
                      turbulence=replace(self.turbulence), pointing=replace(self.pointing),
                      ambient=replace(self.ambient))
        for k, v in kw.items():
            setattr(out, k, v)
        return out

    @property
    def wavelength(self):
        return self.tx.wavelength_nm

    def geometry(self, range_m=None, wavelength=None, camera=False):
        if camera:
            ap, fov = self.camera.aperture_m * 1e3, self.camera.fov_deg / 2
        else:
            ap, fov = self.optics.aperture_mm, self.optics.fov_deg
        return Geometry(range_m=self.range_m if range_m is None else range_m,
                        wavelength_nm=self.wavelength if wavelength is None else wavelength,
                        divergence_deg=self.tx.divergence_deg, waist_mm=self.tx.waist_mm,
                        aperture_mm=ap, fov_deg=fov)

    def channel(self, range_m=None, wavelength=None, model=None, camera=False, offset=None, seed=0,
                progress=None):
        model = model or self.channel_model
        off = self.pointing.offset_m if offset is None else offset
        return channel(self.water, self.geometry(range_m, wavelength, camera), model, off,
                       mc_photons=self.mc_photons, seed=seed, progress=progress)


def _sc(name, desc, water, R, tx, det="PIN photodiode", scheme="OOK-NRZ", rate=1e6, receiver="photodiode",
        cam="Rolling-shutter CMOS (mono)", cam_scheme="Manchester", cam_rate=2000.0, carrier=240.0, ap=50,
        fov=10, filt=30, cam_kw=None,
        turb=None, point=None, amb=None, frames=8, **tx_kw):
    t = TRANSMITTERS[tx].copy(**tx_kw)
    return Scenario(name=name, description=desc, water=water_from_name(water), range_m=R, tx=t,
                    optics=ReceiverOptics(aperture_mm=ap, fov_deg=fov, filter_nm=filt),
                    detector=DETECTORS[det].copy(), camera=CAMERAS[cam].copy(**(cam_kw or {})),
                    receiver=receiver, scheme=scheme, camera_scheme=cam_scheme, bit_rate=rate,
                    camera_chip_rate=cam_rate, camera_carrier_hz=carrier, n_frames=frames,
                    turbulence=turb or Turbulence(), pointing=point or Pointing(), ambient=amb or Ambient())


SCENARIOS = {s.name: s for s in [
    _sc("Clear ocean, 50 m, blue laser + APD", "Jerlov IB, 450 nm laser diode, APD, 10 Mbit/s OOK",
        "Jerlov IB", 50, "Blue laser diode, 0.5 W", "APD (M = 50)", "OOK-NRZ", 10e6, ap=50, fov=5,
        divergence_deg=2.0, amb=Ambient("overcast", 100, "horizontal")),
    _sc("Coastal, 10 m, LED + PIN", "Jerlov 3C, 525 nm LED, PIN, 1 Mbit/s 4-PPM",
        "Jerlov 3C", 10, "Green LED, 1 W", "PIN photodiode", "4-PPM", 1e6, ap=50, fov=20,
        amb=Ambient("clear sky", 8, "horizontal")),
    _sc("Harbour, 4 m, LED + SiPM", "turbid harbour at night, LED array and photon-counting SiPM, Manchester",
        "Turbid harbour", 4, "Blue LED array, 10 W", "SiPM (photon counting)", "Manchester", 2e6, ap=25,
        fov=15, filt=10, amb=Ambient("night", 5, "horizontal"), wavelength_nm=520),
    _sc("Deep sea, 120 m, PMT photon counting", "Jerlov I at 1000 m (dark), laser + PMT, 16-PPM",
        "Jerlov I", 120, "Blue laser diode, 0.5 W", "PMT (photon counting)", "16-PPM", 1e6, ap=100, fov=5,
        divergence_deg=3.0, amb=Ambient("night", 1000, "horizontal")),
    _sc("High-rate OFDM, 15 m", "Jerlov II, blue laser diode, APD, 50 Mbit/s DCO-OFDM 16-QAM",
        "Jerlov II", 15, "Blue laser diode, 0.5 W", "APD (M = 50)", "DCO-OFDM", 50e6, ap=50, fov=10,
        divergence_deg=4.0),
    _sc("Turbulent thermocline, 30 m", "Jerlov IB with gamma-gamma scintillation, SIM-BPSK",
        "Jerlov IB", 30, "Blue laser diode, 0.5 W", "APD (M = 50)", "SIM-BPSK", 2e6, ap=30, fov=5,
        divergence_deg=1.5, turb=Turbulence("gamma-gamma", "si", 0.6, coherence_s=2e-3)),
    _sc("Diver to diver, 4 m, rolling shutter", "Jerlov 1C, diffused LED lamp read by a rolling-shutter "
        "camera, Manchester chips at 2.4 kchip/s", "Jerlov 1C", 4, "Green LED, 1 W", receiver="camera",
        cam="Rolling-shutter CMOS (mono)", cam_scheme="Manchester", cam_rate=2400.0,
        amb=Ambient("clear sky", 6, "horizontal"), source_mm=150, divergence_deg=40),
    _sc("ROV docking, 3 m, rolling-shutter FSK", "Jerlov 5C, rolling-shutter camera decoding stripe "
        "frequency (4-FSK, one symbol per frame)", "Jerlov 5C", 3, "Blue LED array, 10 W", receiver="camera",
        cam="Rolling-shutter CMOS (mono)", cam_scheme="RS-FSK", carrier=1000.0,
        amb=Ambient("overcast", 10, "horizontal"), source_mm=200, divergence_deg=40),
    _sc("Beacon, 20 m, UFSOOK", "Jerlov II, global-shutter camera, undersampled frequency-shift OOK",
        "Jerlov II", 20, "Blue LED, 1 W", receiver="camera", cam="Global-shutter machine vision",
        cam_scheme="UFSOOK", carrier=480.0, amb=Ambient("overcast", 30, "horizontal"), cam_kw=dict(fps=120.0, f_number=8.0),
        source_mm=60, frames=24),
    _sc("LED panel MIMO, 5 m", "Jerlov 3C, 4×4 LED array imaged by a high-speed camera (spatial OOK, "
        "16 bit per frame)", "Jerlov 3C", 5, "Blue LED array, 10 W", receiver="camera",
        cam="Global-shutter high-speed", cam_scheme="Spatial OOK", array=4, pitch_mm=40.0, source_mm=25.0,
        divergence_deg=30),
    _sc("Colour CSK camera, 6 m", "Jerlov 1C, RGB LED with 4-CSK read by a colour global-shutter camera",
        "Jerlov 1C", 6, "RGB LED, 3 W", receiver="camera", cam="Global-shutter colour", cam_scheme="4-CSK",
        cam_rate=30.0, source_mm=40.0, cam_kw=dict(fps=30.0, exposure_s=1e-4, f_number=4.0), frames=16),
]}
