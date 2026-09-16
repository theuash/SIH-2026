"""Frozen M2<->M3 lesion_features schema. Single source of truth.

MATLAB equivalent: table with these 8 columns. Python: plain dict.
Do not add columns without telling M3 owner.
"""
COLUMNS = ["ma_count", "ma_area", "ex_count", "ex_area",
           "he_count", "he_area", "vessel_density", "nv_present"]

def empty():
    return {c: 0 for c in COLUMNS[:-1]} | {"nv_present": False}

def validate(d):
    missing = [c for c in COLUMNS if c not in d]
    if missing:
        raise ValueError(f"lesion_features missing columns: {missing}")
    return d
