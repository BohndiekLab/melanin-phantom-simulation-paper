import numpy as np
import glob
from MSOTAnalysis.io import PAData
from MSOTAnalysis.processing.unmixer import SpectralUnmixer, SO2Calculator
from MSOTAnalysis.processing.spectra import Haemoglobin, OxyHaemoglobin
from MSOTAnalysis.io.msot_data import Reconstruction
import pandas as pd
from scipy.constants import mmHg, hecto
from tom.utils import normalise_sum_to_one
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt


def severinghaus(column):
    # Convert from hPa to mmHg
    po2 = column / mmHg * hecto
    # Apply the severinghaus equation to convert the po2 into so2
    return 100 / (1 + 23400 / (po2 ** 3 + 150 * po2))


def load_po2(file):
    file = file + "/pO2.txt"
    table = pd.read_table(file, skiprows=41, encoding_errors="ignore")
    data = table[[' dt (s) [A Ch.1 Main]',
                  'Oxygen (hPa) [A Ch.1 Main]',
                  ' dt (s) [A Ch.2 Main]',
                 'Oxygen (hPa) [A Ch.2 Main]']]
    data = data.rename(columns={' dt (s) [A Ch.1 Main]': 'dt_1 (s)',
                                ' dt (s) [A Ch.2 Main]': 'dt_2 (s)',
                                'Oxygen (hPa) [A Ch.1 Main]': 'pO2_1 (hPa)',
                                'Oxygen (hPa) [A Ch.2 Main]': 'pO2_2 (hPa)'})
    data["sO2_1"] = severinghaus(data["pO2_1 (hPa)"])
    data["sO2_2"] = severinghaus(data["pO2_2 (hPa)"])
    return (data['dt_1 (s)'] + data['dt_2 (s)']) / 2, (data["sO2_1"].values + data["sO2_2"].values) / 2


def read_tom_hdf5_data(path: str):
    files = glob.glob(path + "/*.hdf5")
    if len(files) != 1:
        raise FileExistsError("Found either none or too many hdf5 files in the folder...")
    pa = PAData.from_hdf5(files[0])
    pa.set_default_recon()  # Stops it loading from multiple reconstruction methods if you have them
    wavelengths = pa.get_wavelengths()
    recons = pa.get_scan_reconstructions()
    timestamps = pa.get_timestamps()

    wavelengths = wavelengths[2:13]
    timestamps = timestamps[:, 2:13]
    timestamps -= timestamps[0, 0]
    timestamps = timestamps
    recons = recons[:, 2:13, 130:200, 130:200, :].raw_data
    mask = recons[0, 0, :, :, :] < 600
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(recons[0, 0, :, :, :]))
    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(mask))
    plt.show()
    plt.close()
    for f_idx in range(len(recons)):
        for i in range(len(wavelengths)):
            recons[f_idx, i, :, :, :][mask] = None

    return timestamps, recons, wavelengths, mask


if __name__ == "__main__":
    SET_NAME = "medium_melanin"  # no_melanin medium_melanin high_melanin
    IN_PATH = f"D:/learned spectral unmixing/raw/flow_phantom_new/{SET_NAME}/"
    OUT_PATH = f"D:/learned spectral unmixing/processed/in_vitro/flow_phantom_new_{SET_NAME}.npz"

    po2_timestamps, po2_data = load_po2(IN_PATH)
    msot_timestamp, msot_data, wavelengths, mask = read_tom_hdf5_data(IN_PATH)

    nandata = np.not_equal(np.isnan(msot_data[:, 0:1, :, :, :]), True)

    distances = distance_transform_edt(nandata)
    distances = distances / np.max(distances)
    depths = np.ones_like(distances)

    po2_data = np.interp(np.linspace(min(po2_timestamps), max(po2_timestamps), len(msot_data)), po2_timestamps,
                         po2_data) / 100
    po2_data[po2_data < 0] = 0
    po2_timestamps = np.interp(np.linspace(min(po2_timestamps), max(po2_timestamps), len(msot_data)), po2_timestamps,
                               po2_timestamps)

    unmixer = SpectralUnmixer(chromophores=[Haemoglobin(), OxyHaemoglobin()], wavelengths=wavelengths)
    results, _, _ = unmixer.run(Reconstruction(raw_data=msot_data, ax_1_labels=wavelengths), None)
    sO2, _, _ = SO2Calculator().run(results, None)
    sO2 = sO2.raw_data[:, :, :, :, :]

    po2_data = np.tile(po2_data, (1, 70, 70, 1, 1))
    po2_timestamps = np.tile(po2_timestamps, (1, 70, 70, 1, 1))
    po2_data = np.moveaxis(po2_data, 4, 0)
    po2_timestamps = np.moveaxis(po2_timestamps, 4, 0)
    msot_data = np.swapaxes(msot_data, 0, 1)
    distances = np.swapaxes(distances, 0, 1)
    depths = np.swapaxes(depths, 0, 1)
    po2_data = np.swapaxes(po2_data, 0, 1)
    sO2 = np.swapaxes(sO2, 0, 1)
    po2_timestamps = np.swapaxes(po2_timestamps, 0, 1)
    distance_mask = distances > 0

    msot_data = msot_data[np.tile(distance_mask, (11, 1, 1, 1, 1))].reshape(len(wavelengths), -1)
    po2_data = po2_data[distance_mask].reshape((-1,))
    sO2 = sO2[distance_mask].reshape((-1,))
    po2_timestamps = po2_timestamps[distance_mask].reshape((-1,))
    distances = distances[distance_mask].reshape((-1,))
    depths = depths[distance_mask].reshape((-1,))
    pO2_timestep_mask = po2_timestamps > 3000
    msot_data = msot_data[:, pO2_timestep_mask]
    po2_data = po2_data[pO2_timestep_mask]
    sO2 = sO2[pO2_timestep_mask]
    distances = distances[pO2_timestep_mask]
    depths = depths[pO2_timestep_mask]
    po2_timestamps = po2_timestamps[pO2_timestep_mask]

    plt.figure()
    plt.plot(po2_timestamps, po2_data * 100, label="pO$_2$ reference", color="green")
    plt.plot(po2_timestamps, sO2 * 100, label="linear unmixing", color="red")
    plt.legend()
    plt.show()
    plt.close()

    np.savez(OUT_PATH,
             wavelengths=wavelengths,
             oxygenations=po2_data,
             spectra=msot_data,
             melanin_concentration=None,
             background_oxygenation=None,
             lu=sO2,
             depths=depths,
             distances=distances,
             timesteps=po2_timestamps
             )

    print("Normalising data...")
    msot_data = np.apply_along_axis(normalise_sum_to_one, 0, msot_data)
    print("Normalising data...[Done]")
