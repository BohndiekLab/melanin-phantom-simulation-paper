import numpy as np
import matplotlib
from patato import SpectralUnmixer, SO2Calculator, Reconstruction
from patato.unmixing.spectra import Haemoglobin, OxyHaemoglobin


def load_spectra_file(file_path: str, load_all_data:bool = False) -> tuple:
    data = np.load(file_path, allow_pickle=True)
    wavelengths = data["wavelengths"]
    oxygenations = data["oxygenations"]
    spectra = data["spectra"]
    distances = data["distances"]
    depths = data["depths"]
    melanin_concentration = data["melanin_concentration"]
    background_oxygenation = data["background_oxygenation"]
    if "timesteps" in data:
        timesteps = data["timesteps"]
    else:
        timesteps = None

    if not load_all_data:
        pre_length = len(spectra[0, :])
        # Signal intensity @ 800nm
        selector = spectra[21, :] / np.max(spectra[21, :]) > 0.10

        # Enforce that at least 10% of the data is used for training
        if np.sum(selector) / len(selector) < 0.1:
            print("Less than 10% of training data would be used. Using top 10% signals at 800nm.")
            selector = spectra[21, :] > np.percentile(spectra[21, :], 90)

        spectra = spectra[:, selector]
        oxygenations = oxygenations[selector]
        post_length = len(spectra[0, :])
        print(f"Using {(post_length/pre_length)*100:.1f}% of the data after filtering 10% of max")

        if str(distances) != "None":
            distances = distances[selector]
        if str(depths) != "None":
            depths = depths[selector]
        if str(melanin_concentration) != "None":
            melanin_concentration = melanin_concentration[selector]
        if str(background_oxygenation) != "None":
            background_oxygenation = background_oxygenation[selector]
        if "timesteps" in data:
            timesteps = data["timesteps"]


    if "tumour_mask" in data:
        tumour_mask = data["tumour_mask"]
    else:
        tumour_mask = None
    if "reference_mask" in data:
        reference_mask = data["reference_mask"]
    else:
        reference_mask = None
    if "mouse_body_mask" in data:
        mouse_body_mask = data["mouse_body_mask"]
    else:
        mouse_body_mask = None
    if "background_mask" in data:
        background_mask = data["background_mask"]
    else:
        background_mask = None
    if "lu" in data:
        lu = data["lu"]
    else:
        unmixer = SpectralUnmixer(chromophores=[Haemoglobin(), OxyHaemoglobin()], wavelengths=wavelengths)
        results, _, _ = unmixer.run(Reconstruction(raw_data=spectra, ax_1_labels=wavelengths), None)
        sO2, _, _ = SO2Calculator().run(results, None)
        lu = sO2.raw_data

    return (wavelengths, oxygenations, lu, spectra, melanin_concentration,
            background_oxygenation, distances, depths, timesteps,
            tumour_mask, reference_mask,
            mouse_body_mask, background_mask)


def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    old_LUT = np.asarray(cmap.colors)
    new_LUT = map(function, old_LUT)
    return matplotlib.colors.ListedColormap(list(new_LUT))


def normalise_sum_to_one(a):
    return a / np.linalg.norm(a)
