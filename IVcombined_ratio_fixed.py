import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def load_and_process_data(files, dataset_label):
    data_list = []
    isc_values = []  # Store Isc values for all files

    for file in files:
        try:
            df = pd.read_csv(file, skiprows=list(np.arange(1, 8)))  # Adjust skiprows if needed
            df.columns = df.columns.str.strip()  # Clean column names
            
            voltage = df["Ucorr[V]"].astype(float)
            current = df["Icorr[A]"].astype(float)

            # Find the index closest to V = 0
            zero_idx = np.abs(voltage).argmin()
            
            # Select 10 points before and 10 points after
            start_idx = max(0, zero_idx - 10)
            end_idx = min(len(voltage), zero_idx + 10)
            
            voltage_subset = voltage[start_idx:end_idx]
            current_subset = current[start_idx:end_idx]
            
            # Perform linear regression
            slope, intercept, _, _, _ = linregress(voltage_subset, current_subset)
            
            # Store Isc (current at V = 0)
            isc_values.append(intercept)
            
            # Filter data for plotting (only V >= 0)
            voltage_p = voltage[voltage >= 0].values
            current_p = current[voltage >= 0].values

            # Insert V=0 with estimated I_sc
            if voltage_p[0] > 0:  # Only if V=0 is missing
                voltage_p = np.insert(voltage_p, 0, 0)
                current_p = np.insert(current_p, 0, intercept)

            # Store for interpolation
            data_list.append((voltage_p, current_p))
        except FileNotFoundError:
            print(f"Warning: {file} not found, skipping...")
            continue
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    if not isc_values:
        print(f"No valid data found for {dataset_label}")
        return 0, 0

    # Compute mean and standard deviation for Isc
    Isc_mean = np.mean(isc_values)
    Isc_std = np.std(isc_values)
    print(f"Short-circuit current (Isc) for dataset {dataset_label}: {Isc_mean:.3f} A ± {Isc_std:.3f} A")

    return Isc_mean, Isc_std

# Define module numbers you want to process (1-23)
module_ids = list(range(1, 24))  # 1 to 23

isc_sample_list = []
isc_module_list = []
valid_modules = []

for i in module_ids:
    sample_files = [f"s{i}a_data.csv", f"s{i}b_data.csv", f"s{i}c_data.csv"]
    module_files = [f"m{i}a_data.csv", f"m{i}b_data.csv", f"m{i}c_data.csv"]

    Isc_sample, _ = load_and_process_data(sample_files, f"Sample {i}")
    Isc_module, _ = load_and_process_data(module_files, f"Module {i}")

    # Only include modules with valid data
    if Isc_sample > 0 and Isc_module > 0:
        isc_sample_list.append(Isc_sample)
        isc_module_list.append(Isc_module)
        valid_modules.append(i)

print(f"Successfully processed {len(valid_modules)} modules: {valid_modules}")

# Define colors for different groups
colors = {
    1: "black",      # REF
    2: "lightsalmon", # B5
    3: "lightsalmon", # B10
    4: "lightsalmon", # B20
    5: "plum",       # Gl2.1S5
    6: "plum",       # Gl2.1S20
    7: "plum",       # Gl1.5S5
    8: "plum",       # Gl1.5S20
    9: "lightblue",  # Blue2.1S
    10: "lightblue", # Blue2.1L
    11: "lightblue", # Blue1.5S
    12: "lightblue", # Blue1.5L
    13: "lightblue", # BlueBa
    14: "plum",      # Gl2.1L5
    15: "plum",      # Gl2.1L20
    16: "plum",      # Gl1.5L5
    17: "plum",      # Gl1.5L20
    18: "sienna",    # BrownC
    19: "sienna",    # BrownBaC
    20: "sienna",    # BrownGlC
    21: "green",     # GreenC
    22: "green",     # GreenBaC
    23: "green"      # GreenGlC
}

# Module short names mapping
module_short_names = {
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
    19: "BrownBaC",
    20: "BrownGlC",
    21: "GreenC",
    22: "GreenBaC",
    23: "GreenGlC"
}

# Compute Isc ratio
isc_ratios = np.array(isc_module_list) / np.array(isc_sample_list)

# Create labels and colors for valid modules only
module_labels = [module_short_names[i] for i in valid_modules]
module_colors = [colors[i] for i in valid_modules]

# Define the desired order for plotting (grouping similar modules together)
desired_order = [1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23]

# Filter to only include modules that were successfully processed
filtered_order = [i for i in desired_order if i in valid_modules]

# Reorder arrays/lists based on the filtered order
order_indices = [valid_modules.index(i) for i in filtered_order]
ordered_labels = [module_labels[i] for i in order_indices]
ordered_ratios = [isc_ratios[i] for i in order_indices]
ordered_colors = [module_colors[i] for i in order_indices]

# Plot
plt.figure(figsize=(14, 7))
bars = plt.bar(range(len(ordered_labels)), ordered_ratios, color=ordered_colors, alpha=0.7)

# Add value labels on top of bars
for i, bar in enumerate(bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", 
             ha="center", va="bottom", fontsize=10, fontweight='bold')

# Labels and title
plt.xlabel("Module", size=12)
plt.xticks(range(len(ordered_labels)), ordered_labels, rotation=45, ha='right', size=10)
plt.ylabel("Isc Ratio (Module / Sample)", size=12)
plt.title("Short-Circuit Current (Isc) Ratio by Module Type", size=14)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Add legend for color groups
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor="black", alpha=0.7, label="Reference"),
    plt.Rectangle((0,0),1,1, facecolor="lightsalmon", alpha=0.7, label="BaSO4"),
    plt.Rectangle((0,0),1,1, facecolor="plum", alpha=0.7, label="Glass"),
    plt.Rectangle((0,0),1,1, facecolor="lightblue", alpha=0.7, label="Blue"),
    plt.Rectangle((0,0),1,1, facecolor="sienna", alpha=0.7, label="Brown"),
    plt.Rectangle((0,0),1,1, facecolor="green", alpha=0.7, label="Green")
]
#plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save and show
plt.savefig("IV_Curve_Comparison_All_Modules.png", dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print(f"\nSummary of Isc Ratios:")
print(f"Mean ratio: {np.mean(isc_ratios):.3f}")
print(f"Std ratio: {np.std(isc_ratios):.3f}")
print(f"Min ratio: {np.min(isc_ratios):.3f}")
print(f"Max ratio: {np.max(isc_ratios):.3f}")

# Create a summary table
summary_df = pd.DataFrame({
    'Module': valid_modules,
    'Short_Name': [module_short_names[i] for i in valid_modules],
    'Sample_Isc': isc_sample_list,
    'Module_Isc': isc_module_list,
    'Ratio': isc_ratios
})

print(f"\nDetailed Results:")
print(summary_df.to_string(index=False)) 