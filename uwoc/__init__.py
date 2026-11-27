"""Underwater optical communications toolbox: seawater optics, scattering channels, turbulence,
photodiode and camera receivers, and a library of intensity-modulation schemes."""
__version__ = "1.0.0"

from .core.water import Water, water_from_name, WATER_TYPES  # noqa: F401
