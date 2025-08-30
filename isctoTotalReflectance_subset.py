import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable

# Use a non-interactive backend to ensure saving works without a display
matplotlib.use("Agg")


def main() -> None:
    # === TOGGLE: Set the feature to color by ===
    # Supported: "glass_percentage"
    color_feature = "glass_percentage"

    # === Exclude specific modules (keep all others) ===
    excluded_modules = {"m19", "m21", "m23", "m24", "m26", "m29"}

    # === Input data ===
    base_dir = os.path.dirname(os.path.abspath(__file__))
    total_folder = os.path.join(base_dir, "total", "converted_csv")
    csv_path = os.path.join(total_folder, "integrated_reflectance_all.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}")

    total_reflectance = pd.read_csv(csv_path)

    # Short-circuit current values (A)
    isc_values = {
        "m1": 8.477, "m2": 8.501, "m3": 8.462, "m4": 8.453, "m5": 8.084,
        "m6": 6.447, "m7": 8.437, "m8": 8.442, "m9": 6.919, "m10": 7.235,
        "m11": 7.452, "m12": 7.437, "m13": 7.955, "m14": 8.418, "m15": 8.202,
        "m16": 8.483, "m17": 7.932, "m18": 6.563, "m19": 6.615, "m20": 6.661,
        "m21": 6.634, "m22": 6.531, "m23": 6.730, "m24": 7.859, "m25": 7.901,
        "m26": 7.967, "m27": 7.916, "m28": 7.926, "m29": 7.960,
    }

    # Material family and glass percentage mapping (from notebook)
    module_properties = {
        "m1": ("B", 0), "m2": ("B", 5), "m3": ("B", 10), "m4": ("B", 20),
        "m5": ("G", 5), "m6": ("G", 20), "m7": ("G", 5), "m8": ("G", 20),
        "m9": ("5", 9), "m10": ("5", 9), "m11": ("5", 9), "m12": ("5", 9),
        "m13": ("5", 9), "m14": ("G", 5), "m15": ("G", 20), "m16": ("G", 5), "m17": ("G", 20),
        "m18": ("MB5B", 0), "m19": ("MB5B", 0), "m20": ("MB5B", 20), "m21": ("MB5B", 20),
        "m22": ("MB5B", 20), "m23": ("MB5B", 20), "m24": ("MB7G", 0), "m25": ("MB7G", 0),
        "m26": ("MB7G", 20), "m27": ("MB7G", 20), "m28": ("MB7G", 20), "m29": ("MB7G", 20),
    }

    # Marker styles per family
    marker_styles = {"B": "s", "G": "o", "5": "D", "MB5B": "^", "MB7G": "v"}

    # Modules that also contain BaSO4 (to outline)
    also_baso4 = {"m13", "m20", "m21", "m26", "m27"}

    # === Data preparation ===
    data = []  # tuples: (reflectance, isc, color_val, marker, module, family)

    for _, row in total_reflectance.iterrows():
        module = str(row.get("Module", "")).strip()
        if module in excluded_modules:
            continue
        if module not in isc_values:
            continue
        if module not in module_properties:
            continue

        reflectance = float(row["Integrated Reflectance"])  # expects column exists
        family, glass_pct = module_properties[module]
        marker = marker_styles.get(family, "x")

        if color_feature == "glass_percentage":
            color_val = glass_pct
        else:
            color_val = glass_pct

        data.append((reflectance, isc_values[module], color_val, marker, module, family))

    if not data:
        raise RuntimeError("No data points collected. Check CSV content and filters.")

    # Unpack data
    reflectances = [d[0] for d in data]
    iscs = [d[1] for d in data]
    color_vals = [d[2] for d in data]

    # Normalize for continuous colormap
    vmin = float(np.nanmin(color_vals))
    vmax = float(np.nanmax(color_vals))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["viridis"]

    # === Plotting ===
    fig, ax = plt.subplots(figsize=(16, 8))

    for x, y, val, marker, mod, fam in data:
        edge_color = "black" if mod in also_baso4 else None

        if mod == "m1":
            ax.scatter(x, y, color="red", marker=marker, edgecolors="r", s=400, linewidths=1.5, label="m1 (reference)")
        else:
            ax.scatter(x, y, c=[val], cmap=cmap, norm=norm, marker=marker, edgecolors=edge_color, s=300, linewidths=1.5)

    # === Add annotations for specific modules ===
    for x, y, val, marker, mod, fam in data:
        if mod == "m17":
            ax.annotate("Gl1.5L20", (x, y), xytext=(10, 10), textcoords='offset points', 
                       fontsize=10, ha='left', va='bottom', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        elif mod == "m6":
            ax.annotate("Gl2.1S20", (x, y), xytext=(10, -15), textcoords='offset points', 
                       fontsize=10, ha='left', va='top', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        elif mod == "m13":
            ax.annotate("BlueBa", (x, y), xytext=(10, 10), textcoords='offset points', 
                       fontsize=10, ha='left', va='bottom', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    # === Fit line ===
    slope, intercept = np.polyfit(reflectances, iscs, 1)
    xs = np.linspace(min(reflectances), max(reflectances), 100)
    ys = slope * xs + intercept
    ax.plot(xs, ys, "r--", label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")

    # === Colorbar ===
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    ticks = sorted(set(color_vals))
    cbar = fig.colorbar(sm, ax=ax, ticks=ticks)
    cbar.set_label("Glass Percentage (%)")
    cbar.ax.set_yticklabels([f"{v:.0f}%" for v in ticks])

    # === Legend ===
    legend_items = [
        mlines.Line2D([], [], color='black', marker='s', linestyle='None', label='BaSO₄ Fillers', markersize=10),
        mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='Glass Fillers', markersize=10),
        mlines.Line2D([], [], color='black', marker='D', linestyle='None', label='5% MB Blue', markersize=10),
        mlines.Line2D([], [], color='black', marker='^', linestyle='None', label='5% MB Brown', markersize=10),
        mlines.Line2D([], [], color='black', marker='v', linestyle='None', label='7% MB Green', markersize=10),
    ]
    legend_items.append(
        mlines.Line2D([], [], color='Red', marker='s', linestyle='None', label='Reference', markersize=10, markeredgecolor='none')
    )
    legend_items.append(
        mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='With BaSO₄', markerfacecolor='none', markeredgecolor='black', markersize=12)
    )
    ax.legend(handles=legend_items, loc='best')

    # === Labels and Save ===
    ax.set_xlabel("Total Reflectance (Integrated)")
    ax.set_ylabel("Short-Circuit Current (Isc) [A]")
    ax.set_title("Isc vs Total Reflectance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(base_dir, "isctoTotalReflectance_filtered.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main() 