import pandas as pd

def normalize_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.copy()
    mapping = {
        "Low Activity": "Low",
        "LowActivity": "Low",
        "Low_Activity": "Low",
        "Moderate Activity": "Moderate",
        "ModerateActivity": "Moderate",
        "High Activity": "High"
    }
    out[target] = out[target].replace(mapping)
    return out

def interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"CO2_ElectroChemicalSensor", "CO2_InfraredSensor"}.issubset(out.columns):
        out["CO2_ratio"] = out["CO2_ElectroChemicalSensor"] / (out["CO2_InfraredSensor"] + 1e-6)
    if {"MetalOxideSensor_Unit1", "MetalOxideSensor_Unit4"}.issubset(out.columns):
        out["MOx_1x4"] = out["MetalOxideSensor_Unit1"] * out["MetalOxideSensor_Unit4"]
    return out
