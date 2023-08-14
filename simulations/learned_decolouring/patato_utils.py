from scipy.constants import mmHg, hecto
import numpy as np
import glob
import pandas as pd
from scipy.io import loadmat
import h5py
from os.path import join, split, exists
from scipy.signal import savgol_filter
from patato import PAData
from patato.io.hdf.hdf5_interface import HDF5Reader

from .ml import gbr_estimate, random_forest_estimate

def normalise(x):
    return x / np.linalg.norm(x)

def severinghaus(column, units):
    # Convert from hPa to mmHg
    if units == "hPa":
        po2 = column / mmHg * hecto
    elif units == "mmHg":
        po2 = column
    else:
        raise ValueError("Units must be mmHg or hPa")
    # Apply the severinghaus equation to convert the po2 into so2
    return 100 / (1 + 23400 / (po2 ** 3 + 150 * po2))

def load_new_po2(file):
    file = glob.glob(join(file, "*.txt"))[0]
    table = pd.read_table(file, skiprows=41, encoding_errors="ignore")
    data = table[[' dt (s) [A Ch.1 Main]',
                  'Oxygen (hPa) [A Ch.1 Main]',
                  ' dt (s) [A Ch.2 Main]',
                 'Oxygen (hPa) [A Ch.2 Main]']]
    data = data.rename(columns={' dt (s) [A Ch.1 Main]': 'dt_1 (s)',
                                ' dt (s) [A Ch.2 Main]': 'dt_2 (s)',
                                'Oxygen (hPa) [A Ch.1 Main]': 'pO2_1 (hPa)',
                                'Oxygen (hPa) [A Ch.2 Main]': 'pO2_2 (hPa)'})
    data["sO2_1"] = severinghaus(data["pO2_1 (hPa)"], "hPa")
    data["sO2_2"] = severinghaus(data["pO2_2 (hPa)"], "hPa")
    return (data['dt_1 (s)'] + data['dt_2 (s)']) / 2, (data["sO2_1"].values + data["sO2_2"].values) / 2

def load_matlab_table(filename, key):
    simplified_data = loadmat(filename, simplify_cells=True)
    # For the data types:
    detailed_data = loadmat(filename, simplify_cells=False)
    column_names = simplified_data[key][0]
    data = simplified_data[key][1:]
    types = [d.dtype for d in detailed_data[key][1]]
    df = pd.DataFrame(data=data, columns=column_names)
    for data_type, column in zip(types, column_names):
        df[column] = df[column].astype(data_type)
    return df

def load_old_po2(folder):
    mat_files = glob.glob(join(folder, "pO2data*.mat"))
    if not mat_files:
        print(f"No po2 data in {folder}")
        return None
    dfs = []
    for mat_file in mat_files:
        df = load_matlab_table(mat_file, "pO2data")
        dfs.append(df)
    df = pd.concat(dfs)
    df["Time"] = pd.to_datetime(df["Time"], format='%H:%M:%S:%f')
    df = df.sort_values("Time", ignore_index=True)
    df["Time"] -= df["Time"][0]
    df["so2 (Pre)"] = severinghaus(df["mmHg (Pre)"], "mmHg")
    df["so2 (Post)"] = severinghaus(df["mmHg (Post)"], "mmHg")
    pO2_values = df["so2 (Pre)"].values
    if np.std(pO2_values) < 1e-10:
        pO2_values = df["so2 (Pre)"].values
    return df["Time"].values.astype(float) / 1e9, pO2_values


class FlowPhantomData(PAData):
    def __init__(self, scan_reader, po2_folder, po2_offset=0):
        super().__init__(scan_reader)
        # load po2_file and interpolate onto the same time as the so2 data.
        if len(glob.glob(join(po2_folder, "*.mat"))) > 0:
            self.po2 = load_old_po2(po2_folder)
            self.po2_format = "mmHg"
        else:
            self.po2 = load_new_po2(po2_folder)
            self.po2_format = "hPa"
        self.po2_offset = po2_offset
    
    # TODO: add a time delay option between the PA and the probes.
    def get_true_so2(self):
        # add smoothing
        t = self.get_timestamps()[:, 0]
        t -= t[0]
        print(self.po2_offset)
        s_interp = np.interp(t + self.po2_offset, *self.po2, left=np.nan, right=np.nan)
        return savgol_filter(s_interp, 10, 1, mode="nearest")

    @classmethod
    def from_hdf5(cls, filename, po2_file, po2_offset=0):
        file = h5py.File(filename, "r")
        return cls(HDF5Reader(file), po2_file)

from patato.core.image_structures.single_parameter_data import SingleParameterData
from patato.core.image_structures.unmixed_image import UnmixedData
from patato.processing.processing_algorithm import SpatialProcessingAlgorithm
    
valid_models = ["BASE" , "WATER_4cm", "ILLUM_POINT", "ACOUS", 
                "BG_0-100", "BG_H2O", "HET_0-100", "HET_60-80", 
                "HET_BG", "ILLUM_5mm", "INVIS", "INVIS_ACOUS", 
                "MSOT", "MSOT_ACOUS", "MSOT_ACOUS_SKIN", "MSOT_SKIN", 
                "RES_0.6", "SKIN", "SMALL", "WATER_2cm"]

# Implement a patato SpectralUnmixer equivalent for the learned method.
class LearnedSO2Calculator(SpatialProcessingAlgorithm):
    """The SO2 calculator. This takes in unmixed data and produces SO2 data.
    """
    def __init__(self, algorithm_id="", nan_invalid=False,
                 model="", model_path=""
                ):
        if model not in valid_models:
            raise ValueError(f"Model {model} is not valid. Please \
            choose from one of the following: {', '.join(valid_models)}.")
        super().__init__(algorithm_id)
        self.model = model
        self.nan_invalid = nan_invalid
        self.model_path = model_path

    def run(self, reconstruction, _, **kwargs):
        """
        Run the LSD SO2 calculator.

        Parameters
        ----------
        reconstruction
            The reconstructed data to process.
        _ : None
            Unused. This is here to make the interface consistent with the other algorithms.
        kwargs
            Unused.
        
        Returns
        -------
        tuple of (SingleImage, dict, None)
            The SO2 data, unused attributes, and unused by product. The first element is the only dataset that is used.
            The second two are there to make the interface consistent with the other algorithms.
        """
        spectra = reconstruction.raw_data
        spectra = spectra / np.linalg.norm(spectra, axis=1)[:, None]
        wavelengths = reconstruction.ax_1_labels
        
        so2 = np.zeros((spectra.shape[0], 1,) + spectra.shape[2:])
        
        spectra_shape = spectra.shape
        so2_shape = (spectra_shape[0], 1) + spectra_shape[2:]
        
        spectra = np.swapaxes(spectra, 0, 1)
        
        spectra = spectra.reshape((spectra.shape[0], -1))
        
        so2 = gbr_estimate(wavelengths, spectra, None, None, 
                           self.model_path, self.model)
        
        so2 = so2.reshape(so2_shape)
        
        # Convert to PATATO format.
        output_data = SingleParameterData(so2.astype(np.float32), ["so2"],
                                          algorithm_id=self.algorithm_id,
                                          attributes=reconstruction.attributes,
                                          field_of_view=reconstruction.fov_3d)
        # Add attributes:
        for a in reconstruction.attributes:
            output_data.attributes[a] = reconstruction.attributes[a]
        output_data.hdf5_sub_name = reconstruction.hdf5_sub_name
        return output_data, {}, None
