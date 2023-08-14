from tom.utils import load_spectra_file
from tom.ml import gbr_estimate
import matplotlib.pyplot as plt
import numpy as np

def normalise_sum_to_one(a):
    return a / np.linalg.norm(a)

# Phantom1_flow_phantom_medium_melanin
# Phantom2_flow_phantom_medium_melanin
DATASET = "Phantom1_flow_phantom_medium_melanin"
path_name = fr"I:\research\seblab\data\group_folders\Janek\learned_pa_oximetry\validation_data\in_vitro\{DATASET}\{DATASET}.npz"

(wavelengths, oxygenations, lu_values, spectra, melanin_concentration,
    background_oxygenation, distances, depths, timesteps, tumour_mask, reference_mask,
    mouse_body_mask, background_mask) = load_spectra_file(path_name, load_all_data=True)

# rf_spectra = zscore(spectra, axis=0, nan_policy='omit')
rf_spectra = np.apply_along_axis(normalise_sum_to_one, 0, spectra)

oxy_shape = np.shape(oxygenations)
rf_oxygenations = []
TRAINING_DATA = ["BASE", "WATER_4cm", "ILLUM_POINT"]
for dataset in TRAINING_DATA:
    oxygenations_a = gbr_estimate(wavelengths, np.reshape(rf_spectra, (len(wavelengths), -1)), None,
                                                          None, r"models_GBR_all/", dataset)
    rf_oxygenations.append(np.squeeze(np.reshape(oxygenations_a, oxy_shape)))

rf_oxygenations = [lu_values] + rf_oxygenations
TRAINING_DATA = ["Linear Unmixing"] + TRAINING_DATA

distinct_timesteps = np.unique(timesteps)
pO2_avg = np.asarray([np.mean(oxygenations[timesteps == step_value]) for step_value in distinct_timesteps])
pO2_std = np.asarray([np.std(oxygenations[timesteps == step_value]) for step_value in distinct_timesteps])

lus = []
lus_std = []
for oxy in rf_oxygenations:
    lus.append(np.asarray([np.mean(oxy[timesteps == step_value]) for step_value in distinct_timesteps]))
    lus_std.append(np.asarray([np.std(oxy[timesteps == step_value]) for step_value in distinct_timesteps]))

colours = ["blue", "orange", "red", "yellow", "pink"]

plt.figure(figsize=(6, 4))
plt.suptitle(DATASET[9:].replace("_", " "))
plt.plot(distinct_timesteps, pO2_avg * 100, label="pO$_2$ reference", color="green")
plt.fill_between(distinct_timesteps, (pO2_avg - pO2_std) * 100, (pO2_avg + pO2_std) * 100, color="green", alpha=0.3)
for data_idx, data_name in enumerate(TRAINING_DATA):
    plt.plot(distinct_timesteps, lus[data_idx] * 100, label=data_name, c=colours[data_idx])
    plt.fill_between(distinct_timesteps, (lus[data_idx] - lus_std[data_idx]) * 100, (lus[data_idx] + lus_std[data_idx]) * 100, color="blue", alpha=0.3)
plt.xlabel("Time [s]")
plt.ylabel("Blood oxygenation sO$_2$ [%]")
plt.legend()

plt.tight_layout()
plt.savefig(DATASET[9:]+".svg")
plt.show()
plt.close()
