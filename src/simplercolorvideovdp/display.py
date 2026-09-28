"""Display photometry and geometry configuration for SimplerColorVideoVDP."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union
import torch


@dataclass
class DisplayPhotometry:
    """Photometric properties of the display.

    Attributes:
        peak_luminance: Maximum display luminance in cd/m² (nit), e.g. 200 for SDR, 1500 for HDR.
        contrast: Display contrast ratio (e.g. 1000 for 1000:1).
        eotf: Electro-optical transfer function ('sRGB', 'PQ', 'HLG', 'linear', or gamma value e.g. '2.2').
        e_ambient: Ambient illuminance in lux (e.g. 250 for bright office, 0 for dark room).
        k_refl: Panel surface reflectivity fraction (default 0.005, i.e. 0.5%).
        exposure: Content exposure multiplier (default 1.0).
        source_colorspace: Source color gamut ('sRGB', 'BT.709', 'BT.2020', etc.).
    """
    peak_luminance: float = 200.0
    contrast: float = 1000.0
    eotf: str = "sRGB"
    e_ambient: float = 250.0
    k_refl: float = 0.005
    exposure: float = 1.0
    source_colorspace: str = "sRGB"

    def get_black_level(self) -> Tuple[float, float]:
        """Returns (Y_black, Y_refl) in cd/m²."""
        y_black = self.peak_luminance / self.contrast
        y_refl = self.e_ambient / math.pi * self.k_refl
        return y_black, y_refl

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Converts display-encoded pixel values in [0, 1] to absolute linear luminance (cd/m²)."""
        if self.eotf != "linear":
            v = v.clamp(0.0, 1.0)

        y_black, y_refl = self.get_black_level()

        if self.eotf == "sRGB" or self.eotf == "BT.709":
            lin = torch.where(v > 0.04045, ((v + 0.055) / 1.055) ** 2.4, v / 12.92)
            if self.exposure == 1.0:
                l = (self.peak_luminance - y_black) * lin + y_black + y_refl
            else:
                l = (self.peak_luminance - y_black) * (lin * self.exposure).clamp(0.0, 1.0) + y_black + y_refl
        elif self.eotf == "linear":
            l = (v * self.exposure).clamp(min=max(0.005, y_black), max=self.peak_luminance) + y_refl
        elif self.eotf == "PQ":
            # PQ EOTF inversion to linear
            l_max = 10000.0
            n = 0.1593017578125
            m = 78.84375
            c1 = 0.8359375
            c2 = 18.8515625
            c3 = 18.6875
            im_t = torch.pow(v, 1.0 / m)
            lin = l_max * torch.pow((im_t - c1).clamp(min=0.0) / (c2 - c3 * im_t), 1.0 / n)
            l = (lin * self.exposure).clamp(min=0.005, max=self.peak_luminance) + y_black + y_refl
        elif self.eotf == "HLG":
            gamma = 1.2
            if self.peak_luminance > 1000.0:
                gamma = 1.2 + 0.42 * math.log10(self.peak_luminance / 1000.0) - 0.07623 * math.log10(max(self.e_ambient, 1e-6) / 5.0)
            a = 0.17883277
            b = 1.0 - 4.0 * a
            c = 0.5 - a * math.log(4.0 * a)
            rgb_s = torch.where(v <= 0.5, torch.pow(v, 2.0) / 3.0, (torch.exp((v - c) / a) + b) / 12.0)
            y_s = 0.2627 * rgb_s[..., 0:1, :, :] + 0.6780 * rgb_s[..., 1:2, :, :] + 0.0593 * rgb_s[..., 2:3, :, :]
            rgb_d = (y_s ** (gamma - 1.0)) * rgb_s
            if self.exposure == 1.0:
                l = (self.peak_luminance - y_black) * rgb_d + y_black + y_refl
            else:
                l = (self.peak_luminance - y_black) * (rgb_d * self.exposure).clamp(0.0, 1.0) + y_black + y_refl
        elif self.eotf[0].isdigit():
            gamma = float(self.eotf)
            l = (self.peak_luminance - y_black) * (torch.pow(v, gamma) * self.exposure).clamp(0.0, 1.0) + y_black + y_refl
        else:
            raise ValueError(f"Unsupported EOTF: '{self.eotf}'")

        return l


@dataclass
class DisplayGeometry:
    """Geometric setup of the display viewing condition.

    Attributes:
        resolution: (width, height) pixel resolution.
        distance_m: Viewing distance in meters.
        diagonal_size_inches: Display diagonal size in inches.
        ppd: Explicit pixels per degree (overrides physical calculation if provided).
    """
    resolution: Tuple[int, int] = (3840, 2160)
    distance_m: Optional[float] = 0.7472
    diagonal_size_inches: Optional[float] = 30.0
    ppd: Optional[float] = None

    def get_ppd(self) -> float:
        """Computes the central pixels-per-degree (ppd)."""
        if self.ppd is not None:
            return float(self.ppd)

        if self.distance_m is None or self.diagonal_size_inches is None:
            raise ValueError("Must specify either `ppd` or both `distance_m` and `diagonal_size_inches`.")

        w, h = self.resolution
        ar = w / h
        height_mm = math.sqrt((self.diagonal_size_inches * 25.4) ** 2 / (1.0 + ar ** 2))
        width_m = (ar * height_mm) / 1000.0

        pix_deg = 2.0 * math.degrees(math.atan(0.5 * width_m / w / self.distance_m))
        return 1.0 / pix_deg


# Standard display presets
DISPLAY_PRESETS = {
    "standard_4k": {
        "photometry": DisplayPhotometry(peak_luminance=200.0, contrast=1000.0, eotf="sRGB", e_ambient=250.0, source_colorspace="sRGB"),
        "geometry": DisplayGeometry(resolution=(3840, 2160), distance_m=0.7472, diagonal_size_inches=30.0),
    },
    "standard_fhd": {
        "photometry": DisplayPhotometry(peak_luminance=200.0, contrast=1000.0, eotf="sRGB", e_ambient=250.0, source_colorspace="sRGB"),
        "geometry": DisplayGeometry(resolution=(1920, 1080), distance_m=0.6, diagonal_size_inches=24.0),
    },
    "standard_phone": {
        "photometry": DisplayPhotometry(peak_luminance=500.0, contrast=10000.0, eotf="sRGB", e_ambient=250.0, source_colorspace="sRGB"),
        "geometry": DisplayGeometry(resolution=(2400, 1080), distance_m=0.4, diagonal_size_inches=6.0),
    },
    "standard_hdr_pq": {
        "photometry": DisplayPhotometry(peak_luminance=1500.0, contrast=1000000.0, eotf="PQ", e_ambient=10.0, source_colorspace="BT.2020-PQ"),
        "geometry": DisplayGeometry(resolution=(3840, 2160), distance_m=0.7472, diagonal_size_inches=30.0),
    },
    "standard_hdr_hlg": {
        "photometry": DisplayPhotometry(peak_luminance=1500.0, contrast=1000000.0, eotf="HLG", e_ambient=10.0, source_colorspace="BT.2020-HLG"),
        "geometry": DisplayGeometry(resolution=(3840, 2160), distance_m=0.7472, diagonal_size_inches=30.0),
    },
    "standard_hdr_linear": {
        "photometry": DisplayPhotometry(peak_luminance=1500.0, contrast=1000000.0, eotf="linear", e_ambient=10.0, source_colorspace="BT.709-linear"),
        "geometry": DisplayGeometry(resolution=(3840, 2160), distance_m=0.7472, diagonal_size_inches=30.0),
    },
    "standard_hdr_linear_dark": {
        "photometry": DisplayPhotometry(peak_luminance=1500.0, contrast=1000000.0, eotf="linear", e_ambient=0.0, source_colorspace="BT.709-linear"),
        "geometry": DisplayGeometry(resolution=(3840, 2160), distance_m=0.7472, diagonal_size_inches=30.0),
    },
    "sdr_4k_30": {
        "photometry": DisplayPhotometry(peak_luminance=200.0, contrast=1000.0, eotf="sRGB", e_ambient=250.0, source_colorspace="sRGB"),
        "geometry": DisplayGeometry(resolution=(3840, 2160), distance_m=0.7472, diagonal_size_inches=30.0),
    },
}


def load_display_model(
    display_name: str = "standard_4k",
    photometry: Optional[DisplayPhotometry] = None,
    geometry: Optional[DisplayGeometry] = None,
    ppd: Optional[float] = None,
) -> Tuple[DisplayPhotometry, DisplayGeometry]:
    """Resolves display photometry and geometry from name or explicit arguments."""
    if display_name in DISPLAY_PRESETS:
        preset = DISPLAY_PRESETS[display_name]
        default_photo = preset["photometry"]
        default_geom = preset["geometry"]
    else:
        default_photo = DisplayPhotometry()
        default_geom = DisplayGeometry()

    resolved_photo = photometry if photometry is not None else default_photo
    resolved_geom = geometry if geometry is not None else default_geom

    if ppd is not None:
        resolved_geom = DisplayGeometry(ppd=ppd)

    return resolved_photo, resolved_geom
