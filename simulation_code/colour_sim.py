import colour
import numpy as np
from scipy.interpolate import interp1d


def get_colour(wl_original, r_original, name="Sample"):
    """
    Simulate the colour of a reflectance spectrum.

    :param wl_original:
    :param r_original:
    :param name:
    :return:
    """
    wl = np.arange(380, 800, 20)
    R = interp1d(wl_original, r_original, fill_value="extrapolate")(wl)

    cmfs = colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer']
    illuminant = colour.SDS_ILLUMINANTS["D65"]

    x = colour.msds_to_XYZ(R, cmfs=cmfs, illuminant=illuminant,
                           shape=colour.SpectralShape(wl[0],
                                                      wl[-1],
                                                      wl[1] - wl[0]
                                                      ),
                           method="integration")
    rgb = colour.XYZ_to_sRGB(x / 100)
    lab = colour.XYZ_to_Lab(x / 100)
    s = colour.plotting.ColourSwatch(rgb, name)
    return s, lab
