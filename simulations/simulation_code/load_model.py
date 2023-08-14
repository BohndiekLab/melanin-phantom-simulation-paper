import pandas as pd
from .tissues import water_absorber, heavy_water_absorber, lipid_absorber, collagen_absorber, blood_absorber, Tissue, \
    mie_rayleigh_scatterer, dermis_scatterer, blood_scatterer, epidermis_scatterer, melanin_absorber
from .tissue_model import TissueModel


def get_tissue_from_row(row):
    row = row.copy()
    absorption = water_absorber(row["WaterVolumeFraction"])
    absorption += heavy_water_absorber(row["HeavyWaterVolumeFraction"])
    absorption += lipid_absorber(row["LipidVolumeFraction"])
    absorption += collagen_absorber(row["CollagenVolumeFraction"])
    absorption += blood_absorber(row["BloodOxygenation"], row["BloodVolumeFraction"])
    absorption += melanin_absorber(row["MelanosomeVolumeFraction"])

    scattering_type = row["ScatteringType"]
    if scattering_type == "MieRayleigh":
        scattering_mus, scattering_g = mie_rayleigh_scatterer(row["ScatteringAPrime"], row["ScatteringFRay"],
                                                              row["ScatteringBMie"])
    elif scattering_type == "Dermis":
        scattering_mus, scattering_g = dermis_scatterer()
    elif scattering_type == "Epidermis":
        scattering_mus, scattering_g = epidermis_scatterer()
    elif scattering_type == "Blood":
        scattering_mus, scattering_g = blood_scatterer()
    else:
        raise ValueError("Scattering type must be one of: 'MieRayleigh', 'Dermis', 'Epidermis', 'Blood'.")

    tissue = Tissue(absorption, scattering_mus, scattering_g, row["RefractiveIndex"], row["Layer"])
    return tissue


def load_model_from_excel(file: str, tissue_nx: int, tissue_ny: int, tissue_nz: int, tissue_dx: int, tissue_cx=0,
                          tissue_cy=0, tissue_cz=0, variations=None):
    df = pd.read_excel(file).set_index("Layer", drop=False)
    if "Background" in df["Layer"]:
        background = get_tissue_from_row(df.loc["Background"])
    t = TissueModel(tissue_nx, tissue_ny, tissue_nz, tissue_dx, tissue_cx, tissue_cy, tissue_cz,
                    background_properties=background)

    if variations is not None:
        for variation in variations:
            layer, updates = variation
            for v in updates:
                df.loc[layer, v] = updates[v]

    for _, row in df.iterrows():
        tissue = get_tissue_from_row(row)
        #
        if row["StructureType"] == "Layer":
            position = None
            axis = None

            # Determine which direction the cylinder is defined in.
            if not pd.isna(row["StructureX"]):
                position = row["StructureX"]
                axis = 0
            if not pd.isna(row["StructureY"]):
                position = row["StructureY"]
                axis = 1
            if not pd.isna(row["StructureZ"]):
                position = row["StructureZ"]
                axis = 2

            t.add_layer(position, row["StructureSize"], tissue, axis)
        elif row["StructureType"] == "Cylinder":
            position = []
            axis = None

            # Determine which direction the cylinder is defined in.
            if not pd.isna(row["StructureX"]):
                position.append(row["StructureX"])
            else:
                axis = 0
            if not pd.isna(row["StructureY"]):
                position.append(row["StructureY"])
            else:
                axis = 1
            if not pd.isna(row["StructureZ"]):
                position.append(row["StructureZ"])
            else:
                axis = 2

            t.add_cylinder(position[0], position[1], row["StructureSize"], tissue, axis)
    return t
