import pandas as pd
import matplotlib.pyplot as plt

# File path (update if needed)
file_path = "sample1a.asc"

# Read the .asc file, skipping metadata lines
with open(file_path, "r") as file:
    lines = file.readlines()

# Find the starting line of the numerical data
for i, line in enumerate(lines):
    if line.startswith("Time[us]"):
        start_idx = i + 1
        break

# Read the data into a pandas DataFrame
df = pd.read_csv(file_path, skiprows=start_idx, delimiter=",", 
                 names=["Time[us]", "Umeas[V]", "Imeas[V]", "Irrad.[kW/m²]", 
                        "Ucorr[V]", "Icorr[A]", "Pcorr[W]"])

# Convert columns to numeric values
df = df.apply(pd.to_numeric, errors='coerce')

# Sort data by voltage (for a cleaner plot)
df = df.sort_values(by="Ucorr[V]", ascending=False)

# Plot IV Curve
plt.figure(figsize=(8,6))
plt.plot(df["Ucorr[V]"], df["Icorr[A]"], label="IV Curve", color="blue")
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.title("IV Curve of the PV Module")
plt.grid(True)
plt.legend()
plt.show()

# Plot Power Curve (Power vs. Voltage)
plt.figure(figsize=(8,6))
plt.plot(df["Ucorr[V]"], df["Pcorr[W]"], label="Power Curve", color="red")
plt.xlabel("Voltage (V)")
plt.ylabel("Power (W)")
plt.title("Power Curve of the PV Module")
plt.grid(True)
plt.legend()
plt.show()