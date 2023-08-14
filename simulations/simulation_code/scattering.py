def musp_ray_mie(wavelength, ap, fray, bmie):
    """
    Reduced scattering coefficient calculation for Rayleigh/Mie scattering.
    :param wavelength:
    :param ap:
    :param fray:
    :param bmie:
    :return:
    """
    return ap * (fray * (wavelength / 500) ** -4 + (1 - fray) * (wavelength / 500) ** -bmie)
