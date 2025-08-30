import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import matplotlib.lines as mlines

# === TOGGLE: Set the feature to color by ===
color_feature = "glass_percentage"

isc_values = {
    "m1": 8.477, "m2": 8.501, "m3": 8.462, "m4": 8.453, "m5": 8.084,
    "m6": 6.447, "m7": 8.437, "m8": 8.442, "m9": 6.919, "m10": 7.235,
    "m11": 7.452, "m12": 7.437, "m13": 7.955, "m14": 8.418, "m15": 8.202,
    "m16": 8.483, "m17": 7.932, "m18": 6.563, "m19": 6.615, "m20": 6.661,
    "m21": 6.634, "m22": 6.531, "m23": 6.730, "m24": 7.859, "m25": 7.901,
    "m26": 7.967, "m27": 7.916, "m28": 7.926, "m29": 7.960
}

module_properties = {
    "m1": ("B", 0), "m2": ("B", 5), "m3": ("B", 10), "m4": ("B", 20),
    "m5": ("G", 5), "m6": ("G", 20), "m7": ("G", 5), "m8": ("G", 20),
    "m9": ("5", 9), "m10": ("5", 9), "m11": ("5", 9), "m12": ("5", 9),
    "m13": ("5", 9), "m14": ("G", 5), "m15": ("G", 20), "m16": ("G", 5), "m17": ("G", 20),
    "m18": ("MB5B", 0), "m19": ("MB5B", 0), "m20": ("MB5B", 20), "m21": ("MB5B", 20),
    "m22": ("MB5B", 20), "m23": ("MB5B", 20), "m24": ("MB7G", 0), "m25": ("MB7G", 0),
    "m26": ("MB7G", 20), "m27": ("MB7G", 20), "m28": ("MB7G", 20), "m29": ("MB7G", 20)
}

marker_styles = {"B": "s", "G": "o", "5": "D", "MB5B": "^", "MB7G": "v"}

refractive_index_map = {
    "m5": 2.1, "m6": 2.1, "m7": 1.5, "m8": 1.5, "m9": 2.1, "m10": 2.1, "m11": 1.5, "m12": 1.5, "m14": 2.1, "m15": 2.1,
    "m16": 1.5, "m17": 1.5, "m22": 1.5, "m23": 1.5, "m28": 1.5, "m29": 1.5
}
particle_size_map = {
    "m5": 7, "m6": 7, "m7": 10, "m8": 10, "m9": 7, "m10": 25, "m11": 10, "m12": 25, "m14": 25, "m15": 25,
    "m16": 15, "m17": 15, "m22": 10, "m23": 10, "m28": 10, "m29": 10
}

also_baso4 = {"m13", "m20", "m21", "m26", "m27"}

# === Load and process data ===
data = []
# Only process modules 18-29
modules_to_plot = ['m18', 'm19', 'm20', 'm21', 'm22', 'm23', 'm24', 'm25', 'm26', 'm27', 'm28', 'm29']

for file in sorted(glob.glob("./ratio/m*_ratio.csv")):
    module = os.path.basename(file).split("_")[0]
    if module not in isc_values or module not in modules_to_plot:
        continue

    df = pd.read_csv(file)
    df_visible = df[(df["Wavelength (nm)"] >= 380) & (df["Wavelength (nm)"] <= 800)]
    avg_ratio = df_visible["Diffuse-to-Total"].mean()

    mat, glass_pct = module_properties[module]
    marker = marker_styles[mat]

    if color_feature == "glass_percentage":
        color_val = glass_pct
    elif color_feature == "refractive_index":
        color_val = refractive_index_map.get(module, np.nan)
    elif color_feature == "particle_size":
        color_val = particle_size_map.get(module, np.nan)
    else:
        color_val = np.nan

    data.append((avg_ratio, isc_values[module], color_val, marker, module))

# === Prepare data for plotting ===
valid_data = [d for d in data if not np.isnan(d[2])]
avg_ratios = [d[0] for d in valid_data]
iscs = [d[1] for d in valid_data]
glass_vals = [d[2] for d in valid_data]
markers = [d[3] for d in valid_data]
modules = [d[4] for d in valid_data]

# === Plot ===
fig, ax = plt.subplots(figsize=(16, 8))
cmap = colormaps["viridis"]
norm = mcolors.Normalize(vmin=min(glass_vals), vmax=max(glass_vals))

for x, y, val, marker, mod in zip(avg_ratios, iscs, glass_vals, markers, modules):
    edge_color = "Black" if mod in also_baso4 else None

    if mod == "m1":
        ax.scatter(x, y, color="Red", marker=marker, edgecolors="r", 
                   s=400, linewidths=1.5)
    else:
        ax.scatter(x, y, c=[val], cmap=cmap, norm=norm, marker=marker,
                   edgecolors=edge_color, s=300, linewidths=1.5)
        # Adjust annotation position to avoid overlapping
        if mod in ['m28']:
            ax.annotate(mod, (x, y), textcoords="offset points", xytext=(5, 15),
                        ha='left', fontsize=9, color='black')
        elif mod in ['m27']:
            ax.annotate(mod, (x, y), textcoords="offset points", xytext=(-5, 15),
                        ha='left', fontsize=9, color='black')    
        else:
            ax.annotate(mod, (x, y), textcoords="offset points", xytext=(0, 15),
                        ha='left', fontsize=9, color='black')
# === Colorbar ===
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
ticks = sorted(set(glass_vals))
cbar = fig.colorbar(sm, ax=ax, ticks=ticks)
cbar.set_label("Filler Percentage (%)")
cbar.ax.set_yticklabels([f"{int(v)}%" for v in ticks])

# === Linear Fit ===
# Removed fitted line as requested
# slope, intercept = np.polyfit(avg_ratios, iscs, 1)
# fit_line = np.polyval([slope, intercept], avg_ratios)
# ax.plot(avg_ratios, fit_line, 'r-', label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")

# === Legend ===
legend_items = [
    mlines.Line2D([], [], color='black', marker=m, linestyle='None', label=label, markersize=8)
    for label, m in zip(["5% MB Brown", "7% MB Green"],
                        ["^", "v"])
]

# Add BaSO₄ indicator
legend_items.append(
    mlines.Line2D([], [], color='Black', marker='o', linestyle='None',
                  label="With BaSO₄", markersize=10, markerfacecolor='none')
)
ax.legend(handles=legend_items, loc='best')

ax.set_xlabel("Diffuse-to-Total Ratio")
ax.set_ylabel("Short-Circuit Current (Isc) [A]")
ax.set_title("Isc vs Diffuse-to-Total Ratio (Modules 18-29)", fontsize=16)
ax.grid(True)
fig.tight_layout()
plt.savefig("isctoDTT_modules_18_29_no_fit.png")
plt.show()