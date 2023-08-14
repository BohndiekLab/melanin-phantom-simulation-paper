import numpy as np
import matplotlib.pyplot as plt

from .colour_sim import get_colour
from .iad_sim import iad_sim_layers

try:
    import pmcx
except ImportError:
    pymcx = None
from .utils import fig2img

from matplotlib.colors import ListedColormap

import io
from contextlib import redirect_stdout, redirect_stderr


def stack(x):
    if type(x[0]) == np.ndarray:
        return np.stack(x).astype(np.float32)
    else:
        return x


def stack_sum(x):
    if type(x[0]) == np.ndarray:
        return sum(x).astype(np.float32)
    else:
        return x[0]


def acuity_source(tissue_definition):
    y = (43.2 + 2.8) * np.tan(np.deg2rad(22.4))
    position = np.array([0, y, 43.2])  # mm
    direction = np.array([0, -y, -43.2 - 2.8])
    direction /= np.linalg.norm(direction)
    return ("msot_acuity_echo", list(tissue_definition.position_to_pixels(*position)),
            list(direction) + [0], [30 / tissue_definition.dx_mm, 0, 0, 0])


def invision_source(tissue_definition, sourceid=0):
    xz_angles = [0, 2 / 5, -2 / 5, 4 / 5, -4 / 5]
    xz_angles = [x * np.pi for x in xz_angles]
    xz_angle = xz_angles[sourceid // 2]

    distance_from_centre = 74.05 / 2
    y_distance = 24.74 / 2

    y_sign = (sourceid % 2 - 0.5) * 2

    source_position = np.array([distance_from_centre * np.sin(xz_angle),
                                -y_distance * y_sign,
                                distance_from_centre * np.cos(xz_angle)])

    illumination_angle = 0.41608649

    direction = np.array([-np.sin(xz_angle),
                          -y_sign * np.sin(illumination_angle),
                          -np.cos(xz_angle)])  # Should this be a minus sign?

    direction /= np.linalg.norm(direction)
    return ("invision", list(tissue_definition.position_to_pixels(*source_position)),
            list(direction) + [0], [tissue_definition.dx_mm, sourceid, 0, 0])


class TissueModel:
    def get_layer_reflectance_colour(self, wavelengths = None):
        if wavelengths is None:
            wavelengths = np.linspace(300, 700, 5)
        if self._reflectance_visible is None:
            iad = self.get_iad(wavelengths)
            self._reflectance_visible = wavelengths, iad[0], iad[2]
            patch, lab = get_colour(wavelengths, iad[2])
            self.lab = lab
            return patch.RGB
        else:
            return get_colour(self._reflectance_visible[0], self._reflectance_visibe[2])[0].RGB

    def get_iad(self, wavelengths):
        return iad_sim_layers(*self.get_iad_layer_input(), wavelengths)

    def get_iad_layer_input(self):
        layers = [l for l in self.layers if l[0].name != "Background"]
        # All layers must be in the same direction for this to work
        if not all([x[3] == layers[0][3] for x in layers]):
            raise ValueError("All layers must be along the same axis to do AD simulation.")
        layers = sorted(layers, key = lambda x: x[1])
        tissue = [l[0] for l in layers]
        thickness = [l[2]/10 for l in layers]
        return tissue, thickness

    def __init__(self, nx, ny, nz, dx_mm, cx=0, cy=0, cz=0, symmetry_axis="z",
                 background_properties=None):
        self._reflectance_visible = None
        self.lab = None
        self.layers = []
        self.definition = np.zeros((nx, ny, nz), dtype=np.uint8)
        self.sym_axis = "xyz".index(symmetry_axis)
        self.r_0 = (cx - nx * dx_mm / 2, cy - ny * dx_mm / 2, cz - nz * dx_mm / 2)
        self.dx_mm = dx_mm
        self.n = (nx, ny, nz)
        if background_properties is None:
            self.tissues = [lambda x, *args: [0, 0, 1, 1.33]]
        else:
            self.tissues = [background_properties]

    def mua(self, wavelength):
        m = np.zeros(self.definition.shape)
        for i, t in enumerate(self.tissues):
            mua, _, _, _ = t(wavelength)
            m[self.definition == i] = mua
        return m

    def imshow(self, axes=None, smooth=False):
        if axes is None:
            axes = (0, self.sym_axis)
        mean_axis = [x for x in range(3) if x not in axes][0]
        image = np.mean(self.definition, axis=mean_axis)
        
        if axes[1] > axes[0]:
            image = image.T
            axes = axes[::-1]

        fig, ax = plt.subplots()

        extent = (self.r_0[axes[1]], self.r_0[axes[1]] + self.n[axes[1]] * self.dx_mm,
                  self.r_0[axes[0]], self.r_0[axes[0]] + self.n[axes[0]] * self.dx_mm)
        from matplotlib.cm import Accent
        cmap = ListedColormap([Accent(i) for i in range(len(self.tissues))])
        im = ax.imshow(image, extent=extent, origin="lower", cmap=cmap, interpolation_stage="rgba", clim=(np.min(image)-0.5, np.max(image)+0.5))
        axis_labels = "xyz"
        ax.set_ylabel(axis_labels[axes[0]])
        ax.set_xlabel(axis_labels[axes[1]])
        cbar = plt.colorbar(im, ax=ax)
        return fig, ax, cbar

    def _repr_png_(self):
        fig, ax, _ = self.imshow()
        img = fig2img(fig)
        plt.close(fig)
        return img._repr_png_()

    def position_to_pixels(self, x, y, z):
        return (x - self.r_0[0]) // self.dx_mm, (y - self.r_0[1]) // self.dx_mm, (z - self.r_0[2]) // self.dx_mm

    def add_layer(self, z, thickness, tissue, axis=2):
        if tissue in self.tissues:
            i = self.tissues.index(tissue)
        else:
            i = len(self.tissues)
            self.tissues.append(tissue)
        nz = (z - self.r_0[axis]) / self.dx_mm
        delta_nz = thickness / self.dx_mm

        start = int(np.round(nz))
        end = start + int(np.round(delta_nz))
        start = max(start, 0)
        end = min(end, self.n[axis])
        self.definition[(slice(None, None),) * axis + (slice(start, end),)] = i
        self.layers.append((tissue, z, thickness, axis))

    def add_cylinder(self, cx, cy, radius, tissue, axis=2):
        if tissue in self.tissues:
            i = self.tissues.index(tissue)
        else:
            i = len(self.tissues)
            self.tissues.append(tissue)
        x = np.linspace(self.r_0[axis - 2] + self.dx_mm / 2,
                        self.r_0[axis - 2] + self.dx_mm / 2 + (self.n[axis - 2] - 1) * self.dx_mm, self.n[axis - 2])
        y = np.linspace(self.r_0[axis - 1] + self.dx_mm / 2,
                        self.r_0[axis - 1] + self.dx_mm / 2 + (self.n[axis - 1] - 1) * self.dx_mm, self.n[axis - 1])

        r = ((x[:, None] - cx) ** 2 + (y[None, :] - cy) ** 2) < radius ** 2
        if axis == 0:
            slicer = np.repeat(r[None], self.n[0], axis=0)
        elif axis == 1:
            slicer = np.repeat(r.T[:, None], self.n[1], axis=1)
        else:
            slicer = np.repeat(r[:, :, None], self.n[2], axis=2)
        self.definition[slicer] = i

    def _run_mcx(self, wavelength, source_def, nphotons=1e7, tstep=5e-9, include_reflectance=False, gpu_device=1):
        if include_reflectance:
            shape = list(self.definition.shape)
            shape[self.sym_axis] += 2
            volume = np.zeros(shape, dtype=self.definition)
            slicer = (slice(None, None),) * self.sym_axis + (slice(1, -2),)
            volume[slicer] = self.definition
        else:
            volume = self.definition

        sourcetype, source_position, source_direction, srcparam1 = source_def
        props = [list(t(wavelength, 0.1)) for t in self.tissues]
        cfg = {"nphoton": int(nphotons),
               "vol": volume,
               "isreflect": 0,
               "tstart": 0,
               "tend": tstep,
               "tstep": tstep,
               "srcpos": source_position,
               "srctype": sourcetype,  # invision msot_acuity_echo
               "srcdir": source_direction,
               "srcparam1": srcparam1,
               "srcparam2": [0, 0, 0, 0],
               "prop": props,
               "issaveref": 0,
               "unitinmm": self.dx_mm,
               "seed": 4711,
               "issrcfrom0": 1,
               "bc": "aaaaaa",
               "isspecular": False,
               "gpuid": gpu_device
               }
        f = io.StringIO()
        
        f2 = io.StringIO()
        with redirect_stdout(f):
            with redirect_stderr(f2):
                result = pmcx.run(cfg)
        result["flux"] = result["flux"][:, :, :, 0] * tstep
        result["mua"] = self.mua(wavelength)
        result["p0"] = result["flux"] * result["mua"]
        result["seg"] = volume
        result["wavelength"] = wavelength
        return result

    def run_mcx(self, wavelengths, source_def, nphotons=1e7, tstep=5e-9, middle_slice_only=False, gpu_device=1):
        if np.isscalar(wavelengths):
            wavelengths = [wavelengths]
        if type(source_def[0]) == str:
            # Single source only to sum over:
            source_def = [source_def]

        results = []
        for w in wavelengths:
            source_results = []
            for source in source_def:
                source_results.append(self._run_mcx(w, source, nphotons, tstep, gpu_device=gpu_device))
            results.append({a: stack_sum([x[a] for x in source_results]) for a in source_results[0].keys()})
            # Don't want to sum up seg
            results[-1]["seg"] = source_results[0]["seg"]

        dict_results = {a: stack([x[a] for x in results]) for a in results[0].keys()}

        if middle_slice_only:
            dict_results["flux"] = dict_results["flux"][:, :, dict_results["flux"].shape[2] // 2]
            dict_results["mua"] = dict_results["mua"][:, :, dict_results["mua"].shape[2] // 2]
            dict_results["p0"] = dict_results["p0"][:, :, dict_results["p0"].shape[2] // 2]
            dict_results["seg"] = dict_results["seg"][:, :, dict_results["seg"].shape[2] // 2]
        return dict_results
