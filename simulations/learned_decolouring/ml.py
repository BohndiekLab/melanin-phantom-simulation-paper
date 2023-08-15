import os
import pickle
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


def random_forest_estimate(test_wavelengths, test_spectra, train_spectra, train_oxygenations,
                           MODELS_PATH, train_ds_name):
    wavelengths_str = "[" + "_".join(map(lambda x: f"{x:.0f}.", test_wavelengths)) + "]"
    rf_save_path = f"{MODELS_PATH}/{train_ds_name}_{wavelengths_str}.random_forest"
    if os.path.exists(rf_save_path):
        with open(rf_save_path, "rb") as rf_file:
            rf = pickle.load(rf_file)
    else:
        rf = RandomForestRegressor(n_estimators=64, max_depth=16, n_jobs=10,
                                   min_samples_leaf=2)
        rf.fit(train_spectra.T, train_oxygenations)
        with open(rf_save_path, "wb") as rf_file:
            pickle.dump(rf, rf_file)

    return rf.predict(test_spectra.T)


def gbr_estimate(test_wavelengths, test_spectra, train_spectra, train_oxygenations,
                           MODELS_PATH, train_ds_name):
    wavelengths_str = "[" + "_".join(map(lambda x: f"{x:.0f}.", test_wavelengths)) + "]"
    rf_save_path = f"{MODELS_PATH}/{train_ds_name}_{wavelengths_str}.gbr"
    
    if os.path.exists(rf_save_path):
        with open(rf_save_path, "rb") as rf_file:
            gbr = pickle.load(rf_file)
    else:
        print("Tried to load:", rf_save_path)
        gbr = HistGradientBoostingRegressor(max_depth=16)
        gbr.fit(train_spectra.T, train_oxygenations)
        with open(rf_save_path, "wb") as rf_file:
            pickle.dump(gbr, rf_file)

    return gbr.predict(test_spectra.T)


