from simulation_code.load_model import load_model_from_excel

import numpy as np
import os

# Set mu_a values outside range available in tables to 0.
os.environ["PA_SIM_EXTRAPOLATE"] = "0"

# These parameters will be used for MCX simulation.
nx, ny, nz = 668, 400, 300
dx = 0.06  # mm

# Wavelengths to get reflectance measurements at.
WAVELENGTHS = np.arange(380, 905, 5)

# Melanosome volume fractions: varied logarithmically.
mvfs = np.logspace(np.log10(0.02), np.log10(0.4), 6)
# The six simulation parameters are assigned to the Fitzpatrick scale.
fps = ["I", "II", "III", "IV", "V", "VI"]


def get_layer_muas(tissue_model, wavelengths=WAVELENGTHS):
    # Get the layers from the tissue model, and extract the absorption at the given wavelengths.
    results = {}
    for tissue in tissue_model.tissues:
        results[tissue.name] = tissue.absorption(WAVELENGTHS)
    return results


reflectances = []
reflectances_685 = []
rgbs = []
dermis_mua = []
epidermis_mua = []
background_mua = []
itas = []

# Make a plot if desired.
# fig, ax = plt.subplots(figsize=(4, 3))

for mvf in mvfs:
    tissue = load_model_from_excel(
        "simulation_specification/forearm_tissue_model.xlsx",
        nx,
        ny,
        nz,
        dx,
        tissue_cz=-nz * dx / 3,
        variations=[("Epidermis", {"MelanosomeVolumeFraction": mvf})],
    )
    colour = tissue.get_layer_reflectance_colour(WAVELENGTHS)
    reflectance_685 = tissue.get_iad([685])[0][0]
    lab = tissue.lab
    ita = np.arctan((lab[0] - 50) / lab[2]) * 180 / np.pi
    itas.append(ita)
    _, r, _ = tissue._reflectance_visible
    muas = get_layer_muas(tissue)

    reflectances.append(r)
    reflectances_685.append(reflectance_685)
    rgbs.append(colour)
    dermis_mua.append(muas["Dermis"])
    epidermis_mua.append(muas["Epidermis"])
    background_mua.append(muas["TissueBackground"])

#     plt.plot(WAVELENGTHS, r, c=colour)
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Reflectance")
# plt.tight_layout()
# plt.show()

# Save the reflectance values to a numpy file.
np.savez(
    "../data/reflectance/intermediate_results",
    reflectances=reflectances,
    ita=itas,
    reflectances_685=reflectances_685,
    rgbs=rgbs,
    wavelengths=WAVELENGTHS,
    fp=fps,
    epidermis_mua=epidermis_mua,
    background_mua=background_mua,
    dermis_mua=dermis_mua,
)

print("Table values:")
print("ITAs:")
print(", ".join([f"{r:.2f}" for r in itas]))
print("Melanosome volume fractions:")
print(", ".join([f"{r:.3f}" for r in mvfs]))


def to_hex(x):
    a, b, c = (x * 255).astype(np.uint8)
    return "#%02x%02x%02x" % (a, b, c)


print("HEX colours:")
print([to_hex(x) for x in rgbs])

print("685 nm reflectances:")
print(", ".join([f"{x:.2f}" for x in reflectances_685]))
