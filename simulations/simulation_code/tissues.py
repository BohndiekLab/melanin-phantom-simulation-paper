from pathlib import Path

import numpy as np
import pandas as pd

from .iad_sim import iad_sim_layers
from .spectrum import Spectrum
from .scattering import musp_ray_mie
from .utils import fig2img
from .colour_sim import get_colour


class Tissue:
    def __init__(self, absorption, scattering, scattering_anisotropy, refractive_index, name=""):
        self.absorption = absorption
        self.scattering = scattering
        self.scattering_anisotropy = scattering_anisotropy
        self.refractive_index = refractive_index
        self.name = name

    def __call__(self, wavelengths, unit_scale=1):
        mua = self.absorption(wavelengths) * unit_scale
        mus = self.scattering(wavelengths) * unit_scale

        return (mua, mus, self.scattering_anisotropy(wavelengths)[()],
                self.refractive_index)

    def get_semi_infinite_reflectance_transmission(self, wls):
        result = iad_sim_layers([self], [100000], wls)
        return result

    def _repr_png_(self):
        import matplotlib.pyplot as plt
        import matplotlib

        wls = np.arange(300, 700, 20)
        result = self.get_semi_infinite_reflectance_transmission(wls)
        patch = get_colour(wls, result[2])[0]
        fig, ax = plt.subplots()
        ax.plot(wls, result[2])
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        # Create a Rectangle patch

        mid_wavelengths = (np.max(wls) - np.min(wls)) / 2
        rgb = patch.RGB
        rgb[rgb < 0] = 0
        rgb[rgb > 1] = 1
        rect = matplotlib.patches.Rectangle((np.min(wls), 0.5), mid_wavelengths, 0.5,
                                            facecolor=rgb)

        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.set_title(f"Reflectance of {self.name}")
        img = fig2img(fig)
        plt.close(fig)
        return img._repr_png_()


def melanin_absorber(mvf=0.01):
    # Jacques
    wavelengths = np.arange(250, 1401, 1)
    mua = 519 * ((wavelengths / 500) ** (-3.5)) * mvf
    absorber = Spectrum(wavelengths, mua, "Melanosome", {"Melanosome Volume Fraction": mvf},
                        "/cm", "absorption")
    return absorber


def heavy_water_absorber(vf=1., outside_range=0):
    file = Path(__file__).parent / "spectra/heavywater2.csv"
    df = pd.read_csv(file)
    wavelengths_data = df["Wavelength"]
    mua_data = vf * df["mua"] / 1000
    wavelengths = np.arange(250, 1401, 1)
    if outside_range != "leftright":
        mua = np.interp(wavelengths, wavelengths_data, mua_data, left=outside_range, right=outside_range)
    else:
        mua = np.interp(wavelengths, wavelengths_data, mua_data, left=mua_data.iloc[0], right=mua_data.iloc[-1])
    return Spectrum(wavelengths, mua, "Heavy Water", {"Heavy Water Volume Fraction": vf},
                    "/cm", "absorption")


def collagen_absorber(cvf=1.):
    file = Path(__file__).parent / "spectra/collagen.csv"
    df = pd.read_csv(file)
    wavelengths = df["Wavelength (nm)"]
    mua = df["Absorption (/cm)"] * cvf
    return Spectrum(wavelengths, mua, "Collagen", {"Collagen Volume Fraction": cvf},
                    "/cm", "absorption")


def lipid_absorber(lvf=1.):
    file = Path(__file__).parent / "spectra/lipid.txt"
    df = pd.read_table(file)
    wavelengths = df["nm"]
    mua = df["mu_a(/m)"] * lvf / 100  # Convert to per cm
    return Spectrum(wavelengths, mua, "Lipid", {"Lipid Volume Fraction": lvf},
                    "/cm", "absorption")


def blood_absorber(oxy=0.8,
                   bvf=None,
                   blood_moles=None):
    if blood_moles is None:
        blood_moles = 150 / 64500
    if bvf is not None:
        blood_moles *= bvf

    file = Path(__file__).parent / "spectra/prahl_hb_molarpercm.txt"
    df = pd.read_table(file)

    wavelengths = df["lambda"]
    absorption = blood_moles * np.log(10) * (oxy * df["HbO2"] + (1 - oxy) * df["Hb"])

    return Spectrum(wavelengths, absorption, f"Blood (sO2={oxy:.2f})", {"Blood Oxygenation": oxy,
                                                                        "Blood Molar Concentration": blood_moles},
                    "/cm", "absorption")


def water_absorber(wvf):
    file = Path(__file__).parent / "spectra/water_full.csv"
    df = pd.read_csv(file)
    wavelengths = df["Wavelength (nm)"]
    mua = df["Absorption (/cm)"] * wvf
    return Spectrum(wavelengths, mua, "Water", {"Water Volume Fraction": wvf},
                    "/cm", "absorption")


def mie_rayleigh_scatterer(ap, fray, bmie, name="Generic", g=0.9):
    wavelengths = np.arange(250, 1401, 1)
    mus = musp_ray_mie(wavelengths, ap, fray, bmie) / (1 - g)
    scatterer = Spectrum(wavelengths, mus,
                         f"{name} Scatterer (Mie/Rayleigh)",
                         {"ap": ap, "fray": fray, "bmie": bmie,
                          "g": g},
                         "/cm", "scattering")
    if np.isscalar(g):
        g = np.ones(wavelengths.shape) * g
    g = Spectrum(wavelengths, g, f"{name} anisotropy", {"Anisotropy Type": "Constant"},
                 "", "anisotropy")
    return scatterer, g


def epidermis_scatterer():
    return mie_rayleigh_scatterer(66.7, 0.29, 0.687, "epidermis")


def blood_scatterer():
    return mie_rayleigh_scatterer(22.0, 0., 0.660, "blood")


def dermis_scatterer():
    return mie_rayleigh_scatterer(43.6, 0.41, 0.562, "dermis")
