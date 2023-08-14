import numpy as np
import iadpython as iad


def iad_sim_layers(tissues, thicknesses, wavelengths):
    mua_s = []
    mus_s = []
    g_s = []
    n = None  # only one n allowed in this simulation.
    for t in tissues:
        mua, mus, g, new_n = t(wavelengths)
        mua /= 10  # convert to per mm
        mus /= 10  # convert to per mm
        mua_s.append(mua)
        mus_s.append(mus)
        g_s.append(g)
        if n is not None and new_n != n:
            print("n treated as constant for this simulation.")
        n = new_n

    d = np.array(thicknesses) * 10  # convert to mm
    mua = np.column_stack(mua_s)
    mus = np.column_stack(mus_s)
    g = np.column_stack(g_s)

    s = iad.Sample(quad_pts=16)
    s.update_quadrature()
    s.a = mus / (mua + mus)  # albedo
    s.b = d * (mus + mua)  # optical path-length
    s.d = d
    s.g = g
    s.n = n
    rt_result = s.rt()
    return rt_result
