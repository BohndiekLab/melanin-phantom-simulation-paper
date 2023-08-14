import copy

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from .utils import fig2img


class Spectrum:
    def __init__(self, wavelengths, values, spectrum_name="", attrs=None, units=None, meaning=None):
        self.values = values
        self.wavelengths = wavelengths
        self.interpolator = interp1d(wavelengths, values)
        self.name = spectrum_name
        self.attrs = attrs if attrs is not None else {}
        self.attrs["Composition"] = self.name
        self.units = units
        self.meaning = meaning

    def __call__(self, wavelength):
        return self.interpolator(wavelength)

    def _repr_png_(self):
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.semilogy(self.wavelengths, self.values)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(f"Value ({self.units})")
        ax.set_title(self.name)
        img = fig2img(fig)
        plt.close(fig)
        return img._repr_png_()

    def __add__(self, other: "Spectrum", outside_range=None):
        import os
        if outside_range is None:
            outside_range = float(os.environ.get("PA_SIM_EXTRAPOLATE", None))

        if other.meaning != self.meaning and other.meaning is not None and self.meaning is not None:
            raise AttributeError("Trying to add incompatible spectra.")
        if other.units != self.units:
            raise AttributeError("Units are inconsistent pls fix.")
        new = copy.deepcopy(self)
        if outside_range is None:
            new_wavelengths = np.arange(int(max(np.min(self.wavelengths), np.min(other.wavelengths))),
                                        int(min(np.max(self.wavelengths), np.max(other.wavelengths))), 1)
        else:
            new_wavelengths = np.arange(int(min(np.min(self.wavelengths), np.min(other.wavelengths))),
                                        int(max(np.max(self.wavelengths), np.max(other.wavelengths))), 1)
        new_values = np.interp(new_wavelengths, self.wavelengths, self.values, outside_range, outside_range) + \
                     np.interp(new_wavelengths, other.wavelengths, other.values, outside_range, outside_range)

        new.wavelengths = new_wavelengths
        new.values = new_values
        new.name = "Mixture"
        new.interpolator = interp1d(new_wavelengths, new_values)
        for a in self.attrs:
            if a in other.attrs:
                new.attrs[a] = (self.attrs[a], other.attrs[a])
        for a in other.attrs:
            new.attrs[a] = (None, other.attrs[a])
        return new

    def __mul__(self, other):
        if not np.isscalar(other):
            raise AttributeError("Can only multiply by scalar.")
        new = copy.deepcopy(self)
        new.values *= other
        return new

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def __truediv__(self, other):
        return self.__mul__(1 / other)
