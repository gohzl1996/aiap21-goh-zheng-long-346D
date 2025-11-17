DB_PATH = "data/gas_monitoring.db"
TABLE_NAME = "gas_monitoring"

TARGET_COLUMN = "Activity Level"
SESSION_COLUMN = "Session ID"

NUMERIC_COLUMNS = [
    "Temperature", "Humidity", "CO2_InfraredSensor", "CO2_ElectroChemicalSensor",
    "MetalOxideSensor_Unit1", "MetalOxideSensor_Unit2", "MetalOxideSensor_Unit3",
    "MetalOxideSensor_Unit4"
]

# Treat 'Time of Day' as ordinal (no datetime parsing)
CATEGORICAL_COLUMNS = [
    "CO_GasSensor", "HVAC Operation Mode", "Ambient Light Level", "Time of Day"
]

# Based on EDA decisions (missingness and domain semantics)
MISSINGNESS_FLAG_COLUMNS = [
    "CO2_ElectroChemicalSensor", "MetalOxideSensor_Unit3", "CO_GasSensor", "Ambient Light Level"
]

# Outlier-prone numeric columns (Z-score flags)
OUTLIER_COLUMNS_Z = [
    "Temperature", "Humidity", "MetalOxideSensor_Unit2", "CO2_InfraredSensor",
    "MetalOxideSensor_Unit1", "MetalOxideSensor_Unit4"
]

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Domain-aware ordinality
ORDINAL_MAPS = {
    "CO_GasSensor": ["extremely low", "low", "medium", "high", "extremely high", "None"],
    "Ambient Light Level": ["very_dim", "dim", "moderate", "bright", "very_bright", "None"],
    "Time of Day": ["morning", "afternoon", "evening", "night"]
}
