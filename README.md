# Underwater Optical Comms Toolbox (`uwoc`)

This toolbox aims to simulate practical scenarios regarding underwater optical links: how far a
given LED or laser reaches in a given water, which modulation scheme to use, and what the
receiver actually sees. It covers both photodiode receivers and cameras. The water is set by
Jerlov type [1, 2] or directly by its absorption and scattering coefficients. The light is
carried through a Monte Carlo channel [3, 4] or a fast analytic model fitted to it. Every scheme
is simulated at the waveform level and checked against a closed-form BER. 

![GUI, photodiode link](docs/figures/gui_photodiode.png)

## Install and run

```bash
git clone https://github.com/gaschenb/Underwater-Communications-Toolbox.git
cd Underwater-Communications-Toolbox
pip install -e ".[gui]"
uwoc-gui                         # or: python -m uwoc.gui
uwoc list                        # built-in scenarios
```

## Features

1. Water optics. Jerlov oceanic I to III and coastal 1C to 9C [1, 2], the three Petzold waters
   [5], or your own absorption a and scattering b at a reference wavelength. A bio-optical mix
   (chlorophyll, CDOM, minerals) is also available.
2. Channel. Beer–Lambert, an analytic "effective" model, and a Monte Carlo photon tracer, with
   beam divergence, receiver aperture and field of view, pointing offset and jitter [6],
   turbulence [7, 8] and solar background.
3. Transmitters and detectors. LEDs, LED arrays, lasers, micro-LEDs and RGB LEDs; PIN, APD [9],
   SiPM and PMT detectors, the last two as photon counters [10].
4. 27 modulation schemes for photodiodes: OOK-NRZ, OOK-RZ, Manchester, 2/4/8/16-PPM, MPPM [11],
   DPPM [12], DPIM and DH-PIM [13], VPPM and CSK [14], PWM, PAM, subcarrier BPSK and QPSK,
   and DCO, ACO and Flip OFDM [15, 16, 17].
5. Camera links with global or rolling shutters [18, 19]. These run the fixed-length pulse
   schemes sampled per frame or per row, plus UFSOOK [20], UPSOOK [21], rolling-shutter FSK and
   spatial OOK from an LED array.
6. Analytics: intensity vs time, eye diagrams, constellations, spectra, noise breakdowns, fading
   series, camera frame sequences, exposure timing, row profiles, pixel statistics, link-budget
   waterfalls, BER vs range or power, BER maps over range and bit rate, maximum rate vs range,
   a power vs bandwidth efficiency map and fade outage.
7. A GUI with live plots, a command line, and a figure gallery.

## Methods

1. Water. Pure-water absorption follows Pope and Fry [22] and pure seawater scattering Morel
   [23]. Chlorophyll absorption uses the Bricaud et al. spectral shape [24], and CDOM and
   detritus use exponential slopes. The Jerlov types are set by their chlorophyll and CDOM
   loads so that K_d matches the published tables [1, 2]. K_d is estimated as (a + b_b)/0.85,
   after Gordon [25].
2. Phase functions. Fournier–Forand [26] with a chosen backscatter fraction (1.83 % gives the
   Petzold average [5]), Henyey–Greenstein [27] and two-term HG [28]. Each is tabulated with its
   CDF for sampling.
3. Monte Carlo channel. Photons leave a Gaussian beam and their first collision is forced
   before the receiver plane. At every scattering event a semi-analytic local estimate [3, 29]
   scores the probability of reaching the aperture. Here I average the phase function over the
   aperture's angular size, because a point value of the sharp forward peak is very noisy. When
   the aperture subtends more than 0.5 sr, the next flight is scored instead by whether it
   crosses the aperture. Russian roulette ends low-weight photons. The outputs are gain,
   ballistic and scattered power, power per scattering order, the impulse response, arrival
   angles and sample paths.
4. Effective channel. Unscattered light uses Gaussian beam capture [6]. Light scattered into a
   15° forward lobe stays near the beam and spreads as θ₁(PbL)^0.75. That growth is slower than
   diffusion because the phase function is so peaked. The captured part has a gamma-shaped
   delay. I fitted the three constants to the Monte Carlo over Jerlov IB, 3C and harbour water,
   3 to 15 attenuation lengths, 1 to 20° divergence and 5 to 40° field of view. It is within a
   factor of 1.4 rms (about 2 at worst) in power and 1.3 rms in mean delay.
5. Turbulence and pointing. The scintillation index comes from the Rytov variance with aperture
   averaging [7]. Fading is log-normal, gamma-gamma [8] or exponential, and runs as a correlated
   Gaussian-copula process in time. BER averages use stratified quantiles. Pointing jitter is a
   correlated Rayleigh offset on the Gaussian footprint [6].
6. Photodiode link. The waveform runs at several samples per chip: drive, then the LED
   response, then the channel impulse response, then fading, then detector noise (shot, thermal,
   APD excess noise [9], counting statistics [10]), then the receiver filter, then
   integrate-and-dump. A preamble m-sequence sets timing, and training chips calibrate the
   levels. Errors are counted directly, with an edit-distance alignment for variable-length
   schemes like DPIM.
7. Closed-form BER. Each scheme has its own expression [30, 31]. Photon counters use exact
   Poisson statistics. ISI is set by the composite LED, detector and channel bandwidth. OFDM
   includes clipping noise and zero-forcing noise enhancement [15, 17]. CSK uses a union bound
   with per-colour attenuation, since red, green and blue fade differently in water. The
   result is averaged over the fading distribution.
8. Camera link. Each frame is built from the source disk, lens blur, a forward-scatter halo and
   ambient light. Every row integrates the drive over its own exposure window, which makes the
   rolling-shutter stripes [18, 19]. Then come Poisson and read noise and the ADC. The receiver
   finds the source, sums its region of interest per frame (global shutter) or per row (rolling
   shutter), calibrates from pilot symbols and decodes. Chips that fall in the inter-frame gap
   are erasures. The theoretical BER comes from the sample statistics.

## Results and examples

Regenerate everything with `uwoc gallery docs/figures`. GUI screenshots are added when PyQt5 is
installed.

1. Water optics: Jerlov K_d spectra, attenuation of the classic waters, and the phase functions.
   ![Water](docs/figures/water.png)
2. Channel: received power vs range in four waters (Monte Carlo dots over the effective model),
   scattered impulse responses, and Monte Carlo photon paths.
   ![Channel](docs/figures/channel.png)
3. Drive waveforms for a sample of the schemes.
   ![Schemes](docs/figures/schemes.png)
4. Photodiode intensity vs time: 4-PPM over 12.3 m of Jerlov 3C near its BER cliff, and DCO-OFDM.
   ![4-PPM](docs/figures/photodiode_ppm.png)
   ![OFDM](docs/figures/photodiode_ofdm.png)
5. Camera frame sequences: rolling-shutter stripes (Manchester at 2.4 kchip/s, 30 fps), an
   undersampled global-shutter beacon, and two frames from every camera scenario.
   ![Rolling shutter](docs/figures/camera_rolling.png)
   ![UFSOOK](docs/figures/camera_ufsook.png)
   ![Camera scenarios](docs/figures/cameras.png)
6. BER vs range (closed form against waveform Monte Carlo), the largest bit rate at BER 1e-6
   vs range, and power vs bandwidth efficiency for all schemes.
   ![BER](docs/figures/ber.png)
7. Validation. Closed-form BER agrees with the waveform Monte Carlo
   within a factor of a few at BER 2e-3 for every scheme family, including photon counting and
   CSK. Monte Carlo ballistic power matches Beer–Lambert with geometric capture, and Jerlov K_d
   values fall in the published ranges [1].

The built-in scenarios:

1. Clear ocean, 50 m: Jerlov IB, 0.5 W blue laser, APD, 10 Mbit/s OOK.
2. Coastal, 10 m: Jerlov 3C, 1 W green LED, PIN, 1 Mbit/s 4-PPM.
3. Harbour, 4 m: turbid harbour water at night, LED array, SiPM photon counting, Manchester.
4. Deep sea, 120 m: Jerlov I with no daylight, laser, PMT, 16-PPM.
5. High-rate OFDM, 15 m: Jerlov II, laser, APD, 50 Mbit/s DCO-OFDM.
6. Turbulent thermocline, 30 m: Jerlov IB with gamma-gamma scintillation, SIM-BPSK.
7. Diver to diver, 4 m: Jerlov 1C, diffused LED lamp, rolling-shutter camera, Manchester.
8. ROV docking, 3 m: Jerlov 5C, rolling-shutter camera, 4-tone RS-FSK.
9. Beacon, 20 m: Jerlov II, global-shutter camera, UFSOOK.
10. LED panel MIMO, 5 m: Jerlov 3C, 4×4 LED array, high-speed camera, spatial OOK.
11. Colour CSK camera, 6 m: Jerlov 1C, RGB LED, colour global-shutter camera, 4-CSK.

## GUI

The GUI has five workspaces that share one scenario. A change to the water, range, transmitter
or receiver on one page carries over to the others, and plots recompute live in a worker thread.
You can't zoom or pan out past the extent of the data in any plot.

1. Water & channel: IOP spectra, attenuation vs range (Beer, effective, Monte Carlo), impulse
   response, photon paths, phase function.
2. Photodiode link: intensity vs time, eye diagram, levels and constellation, spectrum, fading,
   noise breakdown.
3. Camera link: frames, exposure timing, samples vs time, row profile, constellation and levels,
   pixel statistics.
4. Link budget: budget waterfall, power vs range, BER map over range and bit rate, detector
   comparison.
5. BER & throughput: BER curves (Ctrl+R adds Monte Carlo), throughput vs range, efficiency map,
   fade outage.

Shortcuts:

1. Ctrl+R runs, including the Monte Carlo on the BER page.
2. Esc cancels.
3. Ctrl+L toggles live updates.
4. Ctrl+E exports the plot (PNG), and Ctrl+Shift+E exports the table (CSV).
5. Ctrl+T switches between the light and dark themes.
6. F1 opens help.

![Channel](docs/figures/gui_channel.png)
![Eye](docs/figures/gui_eye.png)
![OFDM](docs/figures/gui_ofdm.png)
![Camera frames](docs/figures/gui_camera_frames.png)
![Camera timing](docs/figures/gui_camera_timing.png)
![CSK camera](docs/figures/gui_camera_csk.png)
![MIMO camera](docs/figures/gui_camera_mimo.png)
![Budget map](docs/figures/gui_budget_map.png)
![BER](docs/figures/gui_ber.png)
![Fading](docs/figures/gui_fading.png)
![Dark theme](docs/figures/gui_dark.png)

## CLI

```bash
uwoc water "Jerlov 3C" --lam 520                    # IOPs; -a 0.2 -b 0.5 for your own water
uwoc channel Coastal --model mc                     # gain, scattered/ballistic, delay spread, bandwidth
uwoc link Coastal --scheme 8-DPIM --plot pd.png     # one photodiode frame and its intensity-vs-time figure
uwoc camera Diver --frames 16 --plot cam.png        # one camera capture and its frame sequence
uwoc budget "Clear ocean" --water "Jerlov III"      # link budget, margin, max range
uwoc ber Coastal --schemes OOK-NRZ 16-PPM DCO-OFDM --ranges 8:16:6 --mc
uwoc gallery docs/figures                           # regenerate the README figures
```

A scenario name can be shortened to its first word or two. The common options are `--range`,
`--water`, `-a/-b`, `--rate`, `--scheme`, `--seed` and `--json`.

## Library use

```python
import numpy as np
from uwoc.core.scenarios import SCENARIOS
from uwoc.core.water import WATER_TYPES, custom_water
from uwoc.core import link, camera, schemes

sc = SCENARIOS["Coastal, 10 m, LED + PIN"].copy(range_m=12.0)
sc.water = custom_water(a=0.25, b=1.1, lam_ref=520)      # or WATER_TYPES["Jerlov 3C"]

ch = sc.channel(model="mc")                              # Monte Carlo channel
print(ch.gain, ch.rms_delay, ch.bandwidth_3db())

run = link.simulate(sc, schemes.make("8-DPIM"), n_bits=20000)
print(run.ber, link.ber_theory(sc, schemes.make("8-DPIM")))
t, y = run.t, run.i_f                                    # detector current vs time

cam = SCENARIOS["Diver to diver, 4 m, rolling shutter"]
r = camera.simulate(cam, "Manchester", n_frames=16)
print(r.frames.shape, r.ber, r.goodput)                  # (frames, rows, cols)

ber = link.ber_vs_range(sc, schemes.make("16-PPM"), np.linspace(5, 20, 31))
```

## Layout

```
uwoc/
  core/        constants, water, montecarlo, channel, devices, schemes, link, camera, scenarios, analysis
  gui/         app, base, widgets, theme, one module per workspace
  examples/    gallery.py (README figures)
  cli.py
docs/figures/
```

## Citations

1. N. G. Jerlov, *Marine Optics*, Elsevier, 1976.
2. C. D. Mobley, *Light and Water: Radiative Transfer in Natural Waters*, Academic Press, 1994.
3. C. Gabriel, M.-A. Khalighi, S. Bourennane, P. Léon and V. Rigaud, "Monte-Carlo-based channel
   characterization for underwater optical communication systems," *J. Opt. Commun. Netw.*
   5(1), 1–12, 2013.
4. W. C. Cox and J. F. Muth, "Simulating channel losses in an underwater optical communication
   system," *J. Opt. Soc. Am. A* 31(5), 920–934, 2014.
5. T. J. Petzold, "Volume scattering functions for selected ocean waters," Scripps Institution
   of Oceanography, SIO Ref. 72-78, 1972.
6. A. A. Farid and S. Hranilovic, "Outage capacity optimization for free-space optical links
   with pointing errors," *J. Lightwave Technol.* 25(7), 1702–1710, 2007.
7. L. C. Andrews and R. L. Phillips, *Laser Beam Propagation through Random Media*, 2nd ed.,
   SPIE Press, 2005.
8. M. A. Al-Habash, L. C. Andrews and R. L. Phillips, "Mathematical model for the irradiance
   probability density function of a laser beam propagating through turbulent media,"
   *Opt. Eng.* 40(8), 1554–1562, 2001.
9. R. J. McIntyre, "Multiplication noise in uniform avalanche diodes," *IEEE Trans. Electron
   Devices* 13(1), 164–168, 1966.
10. R. M. Gagliardi and S. Karp, *Optical Communications*, 2nd ed., Wiley, 1995.
11. H. Sugiyama and K. Nosu, "MPPM: a method for improving the band-utilization efficiency in
    optical PPM," *J. Lightwave Technol.* 7(3), 465–472, 1989.
12. D. Shiu and J. M. Kahn, "Differential pulse-position modulation for power-efficient optical
    communication," *IEEE Trans. Commun.* 47(8), 1201–1210, 1999.
13. Z. Ghassemlooy, A. R. Hayes, N. L. Seed and E. D. Kaluarachchi, "Digital pulse interval
    modulation for optical communications," *IEEE Commun. Mag.* 36(12), 95–99, 1998.
14. IEEE Std 802.15.7-2011, *Short-Range Wireless Optical Communication Using Visible Light*, 2011.
15. J. Armstrong and A. J. Lowery, "Power efficient optical OFDM," *Electron. Lett.* 42(6),
    370–372, 2006.
16. N. Fernando, Y. Hong and E. Viterbo, "Flip-OFDM for unipolar communication systems,"
    *IEEE Trans. Commun.* 60(12), 3726–3733, 2012.
17. S. D. Dissanayake and J. Armstrong, "Comparison of ACO-OFDM, DCO-OFDM and ADO-OFDM in IM/DD
    systems," *J. Lightwave Technol.* 31(7), 1063–1072, 2013.
18. C. Danakis, M. Afgani, G. Povey, I. Underwood and H. Haas, "Using a CMOS camera sensor for
    visible light communication," *IEEE Globecom Workshops*, 1244–1248, 2012.
19. N. T. Le, M. A. Hossain and Y. M. Jang, "A survey of design and implementation for optical
    camera communication," *Signal Process. Image Commun.* 53, 95–109, 2017.
20. R. D. Roberts, "Undersampled frequency shift ON-OFF keying (UFSOOK) for camera
    communications (CamCom)," *Wireless and Optical Communication Conference (WOCC)*, 2013.
21. P. Luo, Z. Ghassemlooy, H. Le Minh, X. Tang and H.-M. Tsai, "Undersampled phase shift ON-OFF
    keying for camera communication," *Int. Conf. Wireless Commun. Signal Process. (WCSP)*, 2014.
22. R. M. Pope and E. S. Fry, "Absorption spectrum (380–700 nm) of pure water. II. Integrating
    cavity measurements," *Appl. Opt.* 36(33), 8710–8723, 1997.
23. A. Morel, "Optical properties of pure water and pure sea water," in *Optical Aspects of
    Oceanography*, N. G. Jerlov and E. Steemann Nielsen, eds., Academic Press, 1–24, 1974.
24. A. Bricaud, M. Babin, A. Morel and H. Claustre, "Variability in the chlorophyll-specific
    absorption coefficients of natural phytoplankton," *J. Geophys. Res.* 100(C7),
    13321–13332, 1995.
25. H. R. Gordon, "Can the Lambert-Beer law be applied to the diffuse attenuation coefficient
    of ocean water?" *Limnol. Oceanogr.* 34(8), 1389–1409, 1989.
26. G. R. Fournier and J. L. Forand, "Analytic phase function for ocean water," *Proc. SPIE*
    2258, Ocean Optics XII, 194–201, 1994.
27. L. G. Henyey and J. L. Greenstein, "Diffuse radiation in the galaxy," *Astrophys. J.* 93,
    70–83, 1941.
28. V. I. Haltrin, "One-parameter two-term Henyey–Greenstein phase function for light
    scattering in seawater," *Appl. Opt.* 41(6), 1022–1028, 2002.
29. C. D. Mobley, B. Gentili, H. R. Gordon, Z. Jin, G. W. Kattawar, A. Morel, P. Reinersman,
    K. Stamnes and R. H. Stavn, "Comparison of numerical models for computing underwater light
    fields," *Appl. Opt.* 32(36), 7484–7504, 1993.
30. Z. Ghassemlooy, W. Popoola and S. Rajbhandari, *Optical Wireless Communications: System and
    Channel Modelling with MATLAB*, 2nd ed., CRC Press, 2019.
31. Z. Zeng, S. Fu, H. Zhang, Y. Dong and J. Cheng, "A survey of underwater optical wireless
    communications," *IEEE Commun. Surveys Tuts.* 19(1), 204–238, 2017.
