
import pandas as pd
import numpy as np
import os
import json
import hashlib
from datetime import datetime

# Create project directories
for folder in [
    "datasets/v1",
    "datasets/v2",
    "metadata",
    "screenshots"
]:
    os.makedirs(folder, exist_ok=True)

# Target classes
classes = [
    "No_Disease",
    "Low_Risk",
    "Moderate_Risk",
    "High_Risk",
    "Severe_Disease"
]

probabilities = [0.50, 0.20, 0.12, 0.10, 0.08]

# Function to generate synthetic records
def generate_data(n, seed):
    np.random.seed(seed)

    risk = np.random.choice(
        classes, n, p=probabilities
    )

    level = pd.Series(risk).map({
        "No_Disease": 0,
        "Low_Risk": 1,
        "Moderate_Risk": 2,
        "High_Risk": 3,
        "Severe_Disease": 4
    }).values

    return pd.DataFrame({
        "Age": np.random.randint(18, 85, n),
        "Blood_Pressure": np.random.normal(
            115 + level * 8, 12, n
        ).round(2),
        "Blood_Glucose": np.random.normal(
            90 + level * 25, 15, n
        ).round(2),
        "Creatinine": np.random.normal(
            0.8 + level * 0.7, 0.3, n
        ).round(2),
        "Hemoglobin": np.random.normal(
            15 - level * 1.5, 1.5, n
        ).round(2),
        "BMI": np.random.normal(
            24 + level * 1.5, 4, n
        ).round(2),
        "Urea": np.random.normal(
            25 + level * 15, 8, n
        ).round(2),
        "Sodium": np.random.normal(
            140 - level * 1.5, 3, n
        ).round(2),
        "Gender": np.random.choice(
            ["Male", "Female"], n
        ),
        "Smoking": np.random.choice(
            ["Yes", "No"], n
        ),
        "Diabetes": np.random.choice(
            ["Yes", "No"], n
        ),
        "Hypertension": np.random.choice(
            ["Yes", "No"], n
        ),
        "Disease_Risk": risk
    })


# VERSION 1: Generate 1000 records
df_v1 = generate_data(1000, 42)

v1_path = "datasets/v1/kidney_disease_v1.csv"
df_v1.to_csv(v1_path, index=False)


# VERSION 2: Add 500 new records
new_df = generate_data(500, 100)

df_v2 = pd.concat(
    [df_v1, new_df],
    ignore_index=True
)

# Introduce controlled missing values
for column in [
    "Blood_Pressure",
    "Blood_Glucose",
    "BMI",
    "Smoking"
]:
    missing_indices = df_v2.sample(
        frac=0.03,
        random_state=42
    ).index

    df_v2.loc[missing_indices, column] = np.nan

v2_path = "datasets/v2/kidney_disease_v2.csv"
df_v2.to_csv(v2_path, index=False)


# Generate dataset metadata
def create_metadata(data, version, file_path, description):

    with open(file_path, "rb") as f:
        checksum = hashlib.sha256(f.read()).hexdigest()

    metadata = {
        "version": version,
        "dataset_name": "Kidney Disease Risk Prediction",
        "description": description,
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "features": list(data.columns),
        "missing_values": int(
            data.isnull().sum().sum()
        ),
        "target": "Disease_Risk",
        "file": file_path,
        "checksum": checksum,
        "created_at": datetime.now().isoformat()
    }

    with open(
        f"metadata/{version}_metadata.json", "w"
    ) as f:
        json.dump(metadata, f, indent=4)


create_metadata(
    df_v1,
    "v1",
    v1_path,
    "Initial synthetic dataset with 1000 records."
)

create_metadata(
    df_v2,
    "v2",
    v2_path,
    "Added 500 records and introduced controlled missing values."
)


# Verify dataset integrity
def verify_dataset(version):

    with open(
        f"metadata/{version}_metadata.json"
    ) as f:
        metadata = json.load(f)

    with open(metadata["file"], "rb") as f:
        current_hash = hashlib.sha256(f.read()).hexdigest()

    return current_hash == metadata["checksum"]


# Display results
print("Experiment 1 completed successfully!")
print("Version 1 Shape:", df_v1.shape)
print("Version 2 Shape:", df_v2.shape)

print("V1 Integrity:",
      "Verified" if verify_dataset("v1") else "Failed")

print("V2 Integrity:",
      "Verified" if verify_dataset("v2") else "Failed")

print("\nVersion 2 Missing Values:")
print(df_v2.isnull().sum())


