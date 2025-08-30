import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import matplotlib.cm as cm

# Define which module to plot (change this to plot different modules)
MODULE_TO_PLOT = "m10"  # Change this to "m2", "m3", etc.

# Load reflectance file for the specific module
reflectance_file = f"{MODULE_TO_PLOT}_reflectance.csv"

# Check if file exists
if not os.path.exists(reflectance_file):
    print(f"Error: {reflectance_file} not found!")
    print("Available modules:")
    available_files = sorted(glob.glob("m*_reflectance.csv"))
    for file in available_files:
        module_name = os.path.basename(file).split("_")[0]
        print(f"  {module_name}")
    exit()

# Create single plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Define a colormap that represents the visible spectrum (400-700 nm)
wavelength_range = np.linspace(380, 700, 100)  # Generate 100 points between 400-700 nm
colors = cm.jet((wavelength_range - 380) / 300)  # Normalize to range 0-1 for colormap

# Function to get color for each wavelength
def get_wavelength_color(wavelength):
    norm_index = int((wavelength - 380) / 3)  # Scale to the colormap size
    norm_index = np.clip(norm_index, 0, len(colors) - 1)  # Ensure within range
    return colors[norm_index]

# Load and plot the specific module
df = pd.read_csv(reflectance_file)
module_name = os.path.basename(reflectance_file).split("_")[0]

# Assign a color to each wavelength
wavelength_colors = [get_wavelength_color(wl) for wl in df["Wavelength (nm)"]]

# Plot each point individually with its corresponding color
for j in range(len(df)):
    ax.plot(df["Wavelength (nm)"][j:j+2], df["Mean Reflectance"][j:j+2], 
             color=wavelength_colors[j], linewidth=2)

# Fill standard deviation area in gray (for clarity)
ax.fill_between(df["Wavelength (nm)"], df["Mean Reflectance"] - df["Std Dev"],
                 df["Mean Reflectance"] + df["Std Dev"], alpha=0.2, color="gray", 
                 label="±1 Standard Deviation")

# Customize the plot
ax.set_xlabel("Wavelength (nm)")
ax.set_xlim(380, 800)
ax.set_ylim(0, 0.4)
ax.set_ylabel("Reflectance")
ax.set_title(f"Reflectance Spectrum for Blue2.L")
ax.grid(True, alpha=0.3)
ax.legend()

# Adjust layout
plt.tight_layout()
plt.savefig(f"reflcurve_{module_name}_spectrum.png")
plt.show()

print(f"Plotted {module_name} and saved as reflcurve_{module_name}_spectrum.png")