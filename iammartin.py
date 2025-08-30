from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os


# Module name mapping - short names
module_names = {
    1: "REF",
    2: "B5", 
    3: "B10",
    4: "B20",
    5: "Gl2.1S5",
    6: "Gl2.1S20", 
    7: "Gl1.5S5",
    8: "Gl1.5S20",
    9: "Blue2.1S",
    10: "Blue2.1L",
    11: "Blue1.5S", 
    12: "Blue1.5L",
    13: "BlueBa",
    14: "Gl2.1L5",
    15: "Gl2.1L20",
    16: "Gl1.5L5", 
    17: "Gl1.5L20",
    18: "BrownC",
    20: "BrownBaC",
    22: "BrownGIC",
    25: "GreenC",
    27: "GreenBaC",
    28: "GreenGIC"
}

# Choose which modules to plot (best and worst from each group)
module1_number = 2  # Best performing module from the group
module2_number = 4  # Worst performing module from the group
output_folder = "./"  # Update this path if needed

# Load reference module (M1) and both selected modules data
ref_file_path = os.path.join(output_folder, "M1_averaged.csv")
module1_file_path = os.path.join(output_folder, f"M{module1_number}_averaged.csv")
module2_file_path = os.path.join(output_folder, f"M{module2_number}_averaged.csv")

# Load all datasets
ref_df = pd.read_csv(ref_file_path)
module1_df = pd.read_csv(module1_file_path)
module2_df = pd.read_csv(module2_file_path)

# Ensure all dataframes exist with required columns
required_columns = ["Angle (°)", "Isc (A)"]
for df, name in [(ref_df, "Reference M1"), (module1_df, f"Module M{module1_number}"), (module2_df, f"Module M{module2_number}")]:
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"{name} must contain 'Angle (°)' and 'Isc (A)' columns.")

# Calculate the Angular Factor (gamma) for all modules
ref_isc_0 = ref_df.loc[ref_df["Angle (°)"] == 0, "Isc (A)"].values[0]  # Isc at 0° for reference
module1_isc_0 = module1_df.loc[module1_df["Angle (°)"] == 0, "Isc (A)"].values[0]  # Isc at 0° for module1
module2_isc_0 = module2_df.loc[module2_df["Angle (°)"] == 0, "Isc (A)"].values[0]  # Isc at 0° for module2

ref_df["gamma"] = ref_df["Isc (A)"] / ref_isc_0  # Normalized short-circuit current for reference
module1_df["gamma"] = module1_df["Isc (A)"] / module1_isc_0  # Normalized short-circuit current for module1
module2_df["gamma"] = module2_df["Isc (A)"] / module2_isc_0  # Normalized short-circuit current for module2

# Calculate the difference: each module - reference module
# This shows how each module differs from the reference baseline
module1_df["module_vs_ref_diff"] = module1_df["gamma"] - ref_df["gamma"]
module2_df["module_vs_ref_diff"] = module2_df["gamma"] - ref_df["gamma"]

def iam_model(theta, a_r):
    theta_rad = np.radians(theta)  # Convert angle to radians
    return (1 - np.exp(-np.cos(theta_rad) / a_r)) / (1 - np.exp(-1 / a_r))

# Fit IAM model to both modules
popt1, _ = curve_fit(iam_model, module1_df["Angle (°)"], module1_df["gamma"], p0=[0.05])
popt2, _ = curve_fit(iam_model, module2_df["Angle (°)"], module2_df["gamma"], p0=[0.05])

# Store fitted IAM values in new columns
module1_df["IAM_fitted"] = iam_model(module1_df["Angle (°)"], *popt1)
module2_df["IAM_fitted"] = iam_model(module2_df["Angle (°)"], *popt2)

# Get short names for the modules
module1_name = module_names.get(module1_number, f"M{module1_number}")
module2_name = module_names.get(module2_number, f"M{module2_number}")
ref_name = module_names.get(1, "M1")

# Create subplots to show both the original plot and the new comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Both modules fitted IAM comparison
ax1.plot(module1_df["Angle (°)"], module1_df["gamma"], "o-", label=f"{module1_name} Measured", color="blue", markersize=4)
ax1.plot(module1_df["Angle (°)"], module1_df["IAM_fitted"], "--", label=f"{module1_name} Fitted (a_r={popt1[0]:.3f})", color="blue", linewidth=2)
ax1.plot(module2_df["Angle (°)"], module2_df["gamma"], "s-", label=f"{module2_name} Measured", color="red", markersize=4)
ax1.plot(module2_df["Angle (°)"], module2_df["IAM_fitted"], "--", label=f"{module2_name} Fitted (a_r={popt2[0]:.3f})", color="red", linewidth=2)
ax1.set_xlabel("AOI (°)")
ax1.set_ylabel("Normalized IAM")
ax1.set_title(f"Best vs Worst Module IAM Comparison\n{module1_name} vs {module2_name}")
ax1.legend()
ax1.grid(True)

# Plot 2: Performance difference from reference module for both modules
ax2.plot(module1_df["Angle (°)"], module1_df["module_vs_ref_diff"], "o-", color="blue", linewidth=2, markersize=6, label=f"{module1_name} vs {ref_name}")
ax2.plot(module2_df["Angle (°)"], module2_df["module_vs_ref_diff"], "s-", color="red", linewidth=2, markersize=6, label=f"{module2_name} vs {ref_name}")
ax2.axhline(y=0, color="black", linestyle="--", alpha=0.7, label=f"Reference baseline ({ref_name})")
ax2.set_xlabel("AOI (°)")
ax2.set_ylabel("Module - REF Difference")
ax2.set_title(f"Performance Difference from Reference {ref_name}\n{module1_name} vs {module2_name} Comparison")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(f"{module1_name}_vs_{module2_name}_vs_{ref_name}_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ IAM models fitted:")
print(f"   - {module1_name}: a_r = {popt1[0]:.3f}")
print(f"   - {module2_name}: a_r = {popt2[0]:.3f}")
print(f"\n📊 {module1_name} vs Reference {ref_name} difference analysis:")
print(f"   - At 0°: Difference = {module1_df.loc[module1_df['Angle (°)'] == 0, 'module_vs_ref_diff'].values[0]:.3f}")
print(f"   - At 45°: Difference = {module1_df.loc[module1_df['Angle (°)'] == 45, 'module_vs_ref_diff'].values[0]:.3f}")
print(f"   - At 60°: Difference = {module1_df.loc[module1_df['Angle (°)'] == 60, 'module_vs_ref_diff'].values[0]:.3f}")
print(f"   - Mean difference across all angles: {module1_df['module_vs_ref_diff'].mean():.3f}")
print(f"   - Std deviation of difference: {module1_df['module_vs_ref_diff'].std():.3f}")
print(f"   - Max positive difference: {module1_df['module_vs_ref_diff'].max():.3f}")
print(f"   - Max negative difference: {module1_df['module_vs_ref_diff'].min():.3f}")

print(f"\n📊 {module2_name} vs Reference {ref_name} difference analysis:")
print(f"   - At 0°: Difference = {module2_df.loc[module2_df['Angle (°)'] == 0, 'module_vs_ref_diff'].values[0]:.3f}")
print(f"   - At 45°: Difference = {module2_df.loc[module2_df['Angle (°)'] == 45, 'module_vs_ref_diff'].values[0]:.3f}")
print(f"   - At 60°: Difference = {module2_df.loc[module2_df['Angle (°)'] == 60, 'module_vs_ref_diff'].values[0]:.3f}")
print(f"   - Mean difference across all angles: {module2_df['module_vs_ref_diff'].mean():.3f}")
print(f"   - Std deviation of difference: {module2_df['module_vs_ref_diff'].std():.3f}")
print(f"   - Max positive difference: {module2_df['module_vs_ref_diff'].max():.3f}")
print(f"   - Max negative difference: {module2_df['module_vs_ref_diff'].min():.3f}")