import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors as mcolors
from colour import Lab_to_XYZ, XYZ_to_sRGB
import matplotlib as mpl

# CONFIG
OUTPUT_DIR = "cielch"
SUBGROUPS = {
    'non_pigmented': [1,2,3,4,5,6,7,8,14,15,16,17],
    'blue': [9,10,11,12,13],
    'brown': [18,20,22],
    'green': [24,26,29],
}
MODULE_DESCRIPTIONS = {
    1: "Module 1 (REF)", 2: "Module 2 (B5)", 3: "Module 3 (B10)", 4: "Module 4 (B20)",
    5: "Module 5 (G2.1S5)", 6: "Module 6 (G2.1S20)", 7: "Module 7 (G1.5S5)", 8: "Module 8 (G1.5S20)",
    9: "Module 9 (Blue2.1S)", 10: "Module 10 (Blue2.1L)", 11: "Module 11 (Blue1.5S)", 12: "Module 12 (Blue1.5L)", 13: "Module 13 (BlueBa)",
    14: "Module 14 (G2.1L5)", 15: "Module 15 (G2.1L20)", 16: "Module 16 (G1.5L5)", 17: "Module 17 (G1.5L20)",
    18: "Module 18 (BrownC)", 20: "Module 20 (BrownBaC)", 22: "Module 22 (BrownGlC)",
    25: "Module 25 (GreenC)", 27: "Module 27 (GreenBaC)", 28: "Module 28 (GreenGlC)",
}

# Load precomputed hue/chroma/L* data
CSV_PATH = 'cielab/only_lightness_correction/hue_chroma_lab_results_lightness.csv'
df = pd.read_csv(CSV_PATH)

# Filter out non-numeric module entries and convert Module to numeric
df = df[pd.to_numeric(df['Module'], errors='coerce').notna()]
df['Module'] = pd.to_numeric(df['Module'])

def max_chroma_for_hue(L, hue_deg, n_steps=200):
    """
    Find the maximum in-gamut chroma for a given lightness and hue.
    """
    chroma_vals = np.linspace(0, 150, n_steps)
    a = chroma_vals * np.cos(np.radians(hue_deg))
    b = chroma_vals * np.sin(np.radians(hue_deg))
    Lab = np.stack([np.full_like(a, L), a, b], axis=-1)
    xyz = np.array([Lab_to_XYZ(lab) for lab in Lab])
    rgb = np.array([XYZ_to_sRGB(x) for x in xyz])
    in_gamut = ((rgb >= 0) & (rgb <= 1)).all(axis=1)
    if not np.any(in_gamut):
        return 0
    return chroma_vals[in_gamut][-1]

def create_lab_ab_color_wheel_background(theta, r, L_fixed=70):
    """
    Create a Lab color wheel background at fixed L* (e.g., 70),
    with a* and b* from -100 to 100 filling the circle.
    """
    # r: 0 (center) to 100 (edge)
    # theta: angle in radians
    a = r * np.cos(theta)
    b = r * np.sin(theta)
    Lab = np.stack([np.full_like(a, L_fixed), a, b], axis=-1)
    Lab_flat = Lab.reshape(-1, 3)
    xyz = np.array([Lab_to_XYZ(lab) for lab in Lab_flat])
    rgb = np.array([XYZ_to_sRGB(x) for x in xyz])
    rgb = np.clip(rgb, 0, 1)
    return rgb.reshape(a.shape + (3,))

def group_mean_ab_hue(df, groupby_col):
    grouped = df.groupby(groupby_col)
    mean_a = grouped['a'].mean()
    mean_b = grouped['b'].mean()
    mean_L = grouped['L'].mean()
    mean_chroma = np.sqrt(mean_a**2 + mean_b**2)
    mean_hue = (np.degrees(np.arctan2(mean_b, mean_a)) + 360) % 360
    result = pd.DataFrame({
        'L': mean_L,
        'a': mean_a,
        'b': mean_b,
        'Chroma': mean_chroma,
        'Hue': mean_hue
    })
    return result

# --- Plotting Functions ---
def plot_cielch_polar_trajectory(df, modules, group_name):
    # Check if plot already exists
    outname = f'cielch_polar_{group_name}'
    #if plot_exists(outname):
    #    return
        
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.set_facecolor("white")
    ax.set_theta_zero_location('E')  # 0° at right (East, +a*)
    ax.set_theta_direction(1)        # Counterclockwise
    
    # Create Lab (a*, b*) color wheel background at L*=70
    theta = np.linspace(0, 2 * np.pi, 360)
    r = np.linspace(0, 100, 200)
    T, R = np.meshgrid(theta, r)
    RGB = create_lab_ab_color_wheel_background(T, R, L_fixed=70)
    ax.pcolormesh(T, R, np.ones_like(R), color=RGB.reshape(-1, 3), shading='auto')
    
    chroma_levels = np.arange(20, 101, 20)
    for c in chroma_levels:
        ax.plot(np.linspace(0, 2 * np.pi, 500), np.full(500, c), '--', color='gray', lw=0.8, alpha=0.7, zorder=1)
    ax.set_yticklabels([])
    label_angle = np.radians(80)
    for c in chroma_levels:
        ax.text(label_angle, c, f"{c}", color='black', fontsize=9, va='center', ha='left', alpha=0.7)
    # --- Improved a*/b* axis label positions ---
    axis_label_radius = 110
    # Offsets in radius for each label (increased for more separation)
    offsets = [15, 15, 15, 15]  # [a*, b*, -a*, -b*]
    axis_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    axis_labels = ['a*', 'b*', '-a*', '-b*']
    axis_xy = [
        (axis_label_radius + offsets[0], 0),    # a* (right)
        (0, axis_label_radius + offsets[1]),    # b* (up)
        (-(axis_label_radius + offsets[2]), 0), # -a* (left)
        (0, -(axis_label_radius + offsets[3]))  # -b* (down)
    ]
    for angle, label, (x, y) in zip(axis_angles, axis_labels, axis_xy):
        ax.plot([angle, angle], [0, 100], color='k', linestyle=':', lw=1, alpha=0.7, zorder=2)
        ax.text(np.arctan2(y, x), np.hypot(x, y), label, fontsize=13, color='k', ha='center', va='center', fontweight='bold', rotation=0)
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '8', '<', '>', 'p', 'H', 'd', '|', '_']
    linestyles = ['-', '--', '-.', ':']
    n_markers = len(markers)
    n_linestyles = len(linestyles)
    for idx, module in enumerate(modules):
        sub = df[df['Module'] == module]
        sub_mean = group_mean_ab_hue(sub, 'Theta_i').sort_index()
        if not sub_mean.empty:
            hue_rad = np.radians(sub_mean['Hue'])
            chroma = sub_mean['Chroma']
            marker = markers[idx % n_markers]
            linestyle = linestyles[(idx // n_markers) % n_linestyles]
            ax.plot(hue_rad, chroma, marker=marker, linestyle=linestyle, label=MODULE_DESCRIPTIONS[module], color='black', linewidth=1, markersize=2)
    # Move the title higher up (about 1 cm)
    ax.set_title(f"CIE L*C*h° Polar Plot: {group_name}", fontsize=16, pad=60)
    ax.grid(False)
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(OUTPUT_DIR, f'{outname}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_cielch_polar_by_angle(df, modules, theta_i, title, outname, output_dir=None):
    """
    Plots modules at a single angle of incidence on the CIE L*C*h° polar plot.
    Color indicates lightness (L*), and markers differentiate modules.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if plot already exists
    #if plot_exists(outname, output_dir):
    #    return
        
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.set_facecolor("white")
    ax.set_theta_zero_location('E')  # 0° at right (East, +a*)
    ax.set_theta_direction(1)        # Counterclockwise
    
    # Create Lab (a*, b*) color wheel background at L*=70
    theta = np.linspace(0, 2 * np.pi, 360)
    r = np.linspace(0, 100, 200)
    T, R = np.meshgrid(theta, r)
    RGB = create_lab_ab_color_wheel_background(T, R, L_fixed=70)
    ax.pcolormesh(T, R, np.ones_like(R), color=RGB.reshape(-1, 3), shading='auto')
    
    # --- Add concentric chroma circles and labels ---
    chroma_levels = np.arange(20, 101, 20)
    for c in chroma_levels:
        ax.plot(np.linspace(0, 2 * np.pi, 500), np.full(500, c), '--', color='gray', lw=0.8, alpha=0.7, zorder=1)
    ax.set_yticklabels([])
    label_angle = np.radians(80)
    for c in chroma_levels:
        ax.text(label_angle, c, f"{c}", color='black', fontsize=9, va='center', ha='left', alpha=0.7)
    
    # --- Add a*/b* axis labels ---
    axis_label_radius = 110
    offsets = [10, 10, 10, 10]  # [a*, b*, -a*, -b*]
    axis_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    axis_labels = ['a*', 'b*', '-a*', '-b*']
    axis_xy = [
        (axis_label_radius + offsets[0], 0),
        (0, axis_label_radius + offsets[1]),
        (-(axis_label_radius + offsets[2]), 0),
        (0, -(axis_label_radius + offsets[3]))
    ]
    for angle, ax_label, (x, y) in zip(axis_angles, axis_labels, axis_xy):
        ax.plot([angle, angle], [0, 100], color='k', linestyle=':', lw=1, alpha=0.7, zorder=2)
        ax.text(np.arctan2(y, x), np.hypot(x, y), ax_label, fontsize=13, color='k', ha='center', va='center', fontweight='bold', rotation=0)
    
    # --- Plot data points for selected modules at the given angle ---
    # Filter data for the specific angle and modules
    data_at_angle = df[(df['Module'].isin(modules)) & (df['Theta_i'] == theta_i)]

    # --- Plot ALL individual points for selected modules at the given angle ---
    # Normalize lightness for colormap using all values
    norm_all = mcolors.Normalize(vmin=data_at_angle['L'].min() - 5, vmax=data_at_angle['L'].max() + 5)
    cmap = plt.get_cmap('gray')
    for idx, row in data_at_angle.iterrows():
        hue_rad = np.radians(row['Hue'])
        chroma = row['Chroma']
        lightness = row['L']
        ax.scatter(
            hue_rad, chroma,
            c=[[lightness]],  # Needs to be a sequence
            cmap=cmap,
            norm=norm_all,
            marker='o',
            edgecolor='none',
            s=30,
            alpha=0.4,
            zorder=5
        )

    # Aggregate data to get one representative point per module using CORRECT mean calculation
    aggregated_data = []
    for module in modules:
        module_data = data_at_angle[data_at_angle['Module'] == module]
        if not module_data.empty:
            mean_a = module_data['a'].mean()
            mean_b = module_data['b'].mean()
            mean_L = module_data['L'].mean()
            mean_chroma = np.sqrt(mean_a**2 + mean_b**2)
            mean_hue = (np.degrees(np.arctan2(mean_b, mean_a)) + 360) % 360
            aggregated_data.append({
                'Module': module,
                'L': mean_L,
                'a': mean_a,
                'b': mean_b,
                'Chroma': mean_chroma,
                'Hue': mean_hue
            })
    
    aggregated_data = pd.DataFrame(aggregated_data)

    # Normalize lightness for colormap using the aggregated values
    l_values = aggregated_data['L']
    norm = mcolors.Normalize(vmin=l_values.min() - 5, vmax=l_values.max() + 5)

    # Plot points for each module with a different marker
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '8', '<', '>', 'p', 'H', 'd', '|', '_']
    for idx, row in aggregated_data.iterrows():
        hue_rad = np.radians(row['Hue'])
        chroma = row['Chroma']
        lightness = row['L']
        module_num = int(row['Module'])
        point_label = MODULE_DESCRIPTIONS[module_num]
        marker = markers[idx % len(markers)]

        ax.scatter(hue_rad, chroma,
                   c=[lightness], # c needs to be a sequence
                   cmap=cmap,
                   norm=norm,
                   marker=marker,
                   edgecolor='black', s=100, label=point_label, zorder=10)

    # Set title and finalize
    ax.set_title(title, fontsize=16, pad=60)
    ax.grid(False)
    ax.set_ylim(0, 100)
    
    # Create and add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) # Dummy mappable
    plt.colorbar(sm, ax=ax, pad=0.1, label="Lightness (L*)")
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{outname}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_cielch_polar_module_trajectory(df, module_id, title, outname):
    """
    Plots the color trajectory of a single module across all incident angles.
    Color indicates lightness (L*), and points are connected to show the trajectory.
    """
    # Check if plot already exists
    #if plot_exists(outname):
    #     return
        
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.set_facecolor("white")
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    
    # Create Lab (a*, b*) color wheel background at L*=70
    theta_bg = np.linspace(0, 2 * np.pi, 360)
    r_bg = np.linspace(0, 100, 200)
    T, R = np.meshgrid(theta_bg, r_bg)
    
    # Use full CIE LCH color conversion
    RGB = create_lab_ab_color_wheel_background(T, R, L_fixed=70)
    ax.pcolormesh(T, R, np.ones_like(R), color=RGB.reshape(-1, 3), shading='auto')

    # --- Add concentric chroma circles and labels ---
    chroma_levels = np.arange(20, 101, 20)
    for c in chroma_levels:
        ax.plot(np.linspace(0, 2 * np.pi, 500), np.full(500, c), '--', color='gray', lw=0.8, alpha=0.7, zorder=1)
    ax.set_yticklabels([])
    label_angle = np.radians(80)
    for c in chroma_levels:
        ax.text(label_angle, c, f"{c}", color='black', fontsize=9, va='center', ha='left', alpha=0.7)

    # --- Add a*/b* axis labels ---
    axis_label_radius = 110
    offsets = [10, 10, 10, 10]
    axis_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    axis_labels = ['a*', 'b*', '-a*', '-b*']
    axis_xy = [
        (axis_label_radius + offsets[0], 0),
        (0, axis_label_radius + offsets[1]),
        (-(axis_label_radius + offsets[2]), 0),
        (0, -(axis_label_radius + offsets[3]))
    ]
    for angle, ax_label, (x, y) in zip(axis_angles, axis_labels, axis_xy):
        ax.plot([angle, angle], [0, 100], color='k', linestyle=':', lw=1, alpha=0.7, zorder=2)
        ax.text(np.arctan2(y, x), np.hypot(x, y), ax_label, fontsize=13, color='k', ha='center', va='center', fontweight='bold', rotation=0)

    # --- Plot data points for the selected module using CORRECT mean calculation ---
    module_data = df[df['Module'] == module_id]
    grouped = module_data.groupby('Theta_i')
    mean_a = grouped['a'].mean()
    mean_b = grouped['b'].mean()
    mean_L = grouped['L'].mean()
    mean_chroma = np.sqrt(mean_a**2 + mean_b**2)
    mean_hue = (np.degrees(np.arctan2(mean_b, mean_a)) + 360) % 360

    if mean_a.empty:
        print(f"Warning: No data found for module {module_id}. Skipping plot.")
        plt.close()
        return

    hue_rad = np.radians(mean_hue)
    chroma = mean_chroma
    lightness = mean_L

    # Normalize lightness for colormap
    norm = mcolors.Normalize(vmin=lightness.min() - 5, vmax=lightness.max() + 5)
    cmap = plt.get_cmap('gray')

    # Plot points colored by lightness
    #sc = ax.scatter(hue_rad, chroma, c=lightness, cmap=cmap, norm=norm, edgecolor='black', s=150, zorder=10)
    # Plot connecting line
    ax.plot(hue_rad, chroma, color='black', alpha=0.5, linestyle='-', linewidth=1.5, zorder=9)

    # Annotate each point with its angle
    #for angle, r_val, theta_val in zip(module_data.index, chroma, hue_rad):
    #    ax.annotate(f"{int(angle)}°", (theta_val, r_val), xytext=(10, 10), textcoords='offset points', fontsize=9, alpha=0.8, zorder=11)

    # Set title and finalize
    ax.set_title(title, fontsize=16, pad=60)
    ax.grid(False)
    ax.set_ylim(0, 100)
    #plt.colorbar(sc, ax=ax, pad=0.1, label="Lightness (L*)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{outname}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_cielch_zoom_snippet(df, module_id, outname):
    """
    Creates a minimalist, zoomed-in cartesian plot of a module's trajectory.
    This is intended as a snippet to be shown alongside a main plot.
    """
    # Check if plot already exists
    #if plot_exists(outname):
   #      return
        
    # Get the module's trajectory data using CORRECT mean calculation
    module_data = df[df['Module'] == module_id]
    grouped = module_data.groupby('Theta_i')
    mean_a = grouped['a'].mean()
    mean_b = grouped['b'].mean()
    mean_L = grouped['L'].mean()
    mean_chroma = np.sqrt(mean_a**2 + mean_b**2)
    mean_hue = (np.degrees(np.arctan2(mean_b, mean_a)) + 360) % 360

    if mean_a.empty:
        print(f"Warning: No data found for module {module_id} for snippet. Skipping plot.")
        plt.close()
        return

    a_coords = mean_a
    b_coords = mean_b
    lightness = mean_L

    # Determine plot limits with a small padding
    padding = 5
    xlim = (a_coords.min() - padding, a_coords.max() + padding)
    ylim = (b_coords.min() - padding, b_coords.max() + padding)

    # Create a small, clean figure
    fig, ax = plt.subplots(figsize=(3, 3))

    # Normalize lightness for colormap
    norm = mcolors.Normalize(vmin=lightness.min(), vmax=lightness.max())
    cmap = plt.get_cmap('gray')

    # Plot trajectory line and points
    ax.plot(a_coords, b_coords, color='black', alpha=0.5, linestyle='-', linewidth=1.5, zorder=1)
    ax.scatter(a_coords, b_coords, c=lightness, cmap=cmap, norm=norm, edgecolor='black', s=150, zorder=2)

    # Annotate each point
    for angle, a_val, b_val in zip(mean_a.index, a_coords, b_coords):
        ax.annotate(f"{int(angle)}°", (a_val, b_val), xytext=(8, 8), textcoords='offset points', fontsize=9, alpha=0.8, zorder=3)

    # Style the plot for a clean snippet
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')  # Hide all axes, ticks, and labels

    # Save the snippet with minimal whitespace
    plt.savefig(os.path.join(OUTPUT_DIR, f'{outname}.png'), dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_all_modules_at_45(df):
    output_dir = os.path.join(OUTPUT_DIR, 'Modules_at_45')
    os.makedirs(output_dir, exist_ok=True)
    all_modules = sorted(df['Module'].unique())
    for module in all_modules:
        plot_cielch_polar_by_angle(
            df,
            modules=[module],
            theta_i=45,
            title=f"CIE L*C*h° Polar Plot: Module {module} at θᵢ=45°",
            outname=f"module{module:02d}_theta45",
            output_dir=output_dir
        )

def plot_exists(outname, output_dir=None):
    """Check if a plot file already exists and skip generation if it does."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    plot_path = os.path.join(output_dir, f'{outname}.png')
    if os.path.exists(plot_path):
        print(f"Skipping {outname}.png - already exists")
        return True
    return False

def plot_blue_modules_overlayed_raw(df, output_dir=None):
    """
    Create overlayed CIELCH polar plot showing all blue modules with different grey colors for incident angles,
    using consistent symbols (circles) with better contrast and no connecting lines.
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'blue_modules_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    blue_modules = SUBGROUPS['blue']
    all_angles = sorted(df['Theta_i'].unique())
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 10))
    ax.set_facecolor("white")
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    
    # Create Lab color wheel background
    theta = np.linspace(0, 2 * np.pi, 360)
    r = np.linspace(0, 100, 200)
    T, R = np.meshgrid(theta, r)
    RGB = create_lab_ab_color_wheel_background(T, R, L_fixed=70)
    ax.pcolormesh(T, R, np.ones_like(R), color=RGB.reshape(-1, 3), shading='auto')
    
    # Add chroma circles and labels
    chroma_levels = np.arange(20, 101, 20)
    for c in chroma_levels:
        ax.plot(np.linspace(0, 2 * np.pi, 500), np.full(500, c), '--', color='gray', lw=0.8, alpha=0.7, zorder=1)
    ax.set_yticklabels([])
    label_angle = np.radians(80)
    for c in chroma_levels:
        ax.text(label_angle, c, f"{c}", color='black', fontsize=9, va='center', ha='left', alpha=0.7)
    
    # Add a*/b* axis labels
    axis_label_radius = 110
    offsets = [15, 15, 15, 15]
    axis_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    axis_labels = ['a*', 'b*', '-a*', '-b*']
    axis_xy = [
        (axis_label_radius + offsets[0], 0),
        (0, axis_label_radius + offsets[1]),
        (-(axis_label_radius + offsets[2]), 0),
        (0, -(axis_label_radius + offsets[3]))
    ]
    for angle, label, (x, y) in zip(axis_angles, axis_labels, axis_xy):
        ax.plot([angle, angle], [0, 100], color='k', linestyle=':', lw=1, alpha=0.7, zorder=2)
        ax.text(np.arctan2(y, x), np.hypot(x, y), label, fontsize=13, color='k', ha='center', va='center', fontweight='bold', rotation=0)
    
    # Use consistent symbols (circles) and better contrast grey colors
    grey_colors = ['#000000', '#333333', '#666666', '#999999', '#CCCCCC', '#FFFFFF', '#1A1A1A', '#4D4D4D']  # Better contrast
    
    # Plot for each incident angle with different grey colors
    for angle_idx, angle in enumerate(all_angles):
        angle_color = grey_colors[angle_idx % len(grey_colors)]
        
        # Plot data for each module at this incident angle
        for module_idx, module in enumerate(blue_modules):
            module_data = df[df['Module'] == module]
            angle_data = module_data[module_data['Theta_i'] == angle]
            
            if not angle_data.empty:
                hue_rad = np.radians(angle_data['Hue'])
                chroma = angle_data['Chroma']
                
                # Use consistent circle symbol, smaller size, no connecting lines
                ax.scatter(hue_rad, chroma, 
                          c=angle_color,
                          marker='o',
                          s=15,  # Smaller size to reduce overlap
                          alpha=0.8, 
                          edgecolors='black',
                          linewidth=0.2,
                          label=f'θᵢ={int(angle)}°' if module_idx == 0 else "",
                          zorder=5)
    
    ax.set_title("Blue Modules: Data Points by Incident Angle\n(Grey colors = angles, Circles = all modules)", fontsize=16, pad=60)
    ax.grid(False)
    ax.set_ylim(0, 100)
    
    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    plt.savefig(os.path.join(output_dir, 'blue_modules_overlayed_raw.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_blue_modules_individual_trajectories(df, output_dir=None):
    """
    Create individual CIELCH polar plots for each blue module showing their trajectories.
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'blue_modules_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    blue_modules = SUBGROUPS['blue']
    
    for module in blue_modules:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))
        ax.set_facecolor("white")
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        
        # Create Lab color wheel background
        theta = np.linspace(0, 2 * np.pi, 360)
        r = np.linspace(0, 100, 200)
        T, R = np.meshgrid(theta, r)
        RGB = create_lab_ab_color_wheel_background(T, R, L_fixed=70)
        ax.pcolormesh(T, R, np.ones_like(R), color=RGB.reshape(-1, 3), shading='auto')
        
        # Add chroma circles and labels
        chroma_levels = np.arange(20, 101, 20)
        for c in chroma_levels:
            ax.plot(np.linspace(0, 2 * np.pi, 500), np.full(500, c), '--', color='gray', lw=0.8, alpha=0.7, zorder=1)
        ax.set_yticklabels([])
        label_angle = np.radians(80)
        for c in chroma_levels:
            ax.text(label_angle, c, f"{c}", color='black', fontsize=9, va='center', ha='left', alpha=0.7)
        
        # Add a*/b* axis labels
        axis_label_radius = 110
        offsets = [15, 15, 15, 15]
        axis_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        axis_labels = ['a*', 'b*', '-a*', '-b*']
        axis_xy = [
            (axis_label_radius + offsets[0], 0),
            (0, axis_label_radius + offsets[1]),
            (-(axis_label_radius + offsets[2]), 0),
            (0, -(axis_label_radius + offsets[3]))
        ]
        for angle, label, (x, y) in zip(axis_angles, axis_labels, axis_xy):
            ax.plot([angle, angle], [0, 100], color='k', linestyle=':', lw=1, alpha=0.7, zorder=2)
            ax.text(np.arctan2(y, x), np.hypot(x, y), label, fontsize=13, color='k', ha='center', va='center', fontweight='bold', rotation=0)
        
        # Get module data
        module_data = df[df['Module'] == module]
        sc = None  # Initialize scatter plot object
        if not module_data.empty:
            # Plot raw data points
            hue_rad = np.radians(module_data['Hue'])
            chroma = module_data['Chroma']
            lightness = module_data['L']
            
            # Normalize lightness for better visualization
            norm = mcolors.Normalize(vmin=lightness.min() - 5, vmax=lightness.max() + 5)
            
            # Plot individual points
            sc = ax.scatter(hue_rad, chroma, 
                           c=lightness, 
                           cmap='viridis', 
                           norm=norm,
                           s=50, 
                           alpha=0.7, 
                           edgecolors='black',
                           linewidth=0.5,
                           zorder=5)
            
            # Plot trajectory line
            grouped = module_data.groupby('Theta_i')
            mean_a = grouped['a'].mean()
            mean_b = grouped['b'].mean()
            mean_chroma = np.sqrt(mean_a**2 + mean_b**2)
            mean_hue = (np.degrees(np.arctan2(mean_b, mean_a)) + 360) % 360
            
            ax.plot(np.radians(mean_hue), mean_chroma, 
                   color='red', 
                   linewidth=3, 
                   alpha=0.9,
                   label='Trajectory',
                   zorder=6)
            
            # Annotate angle points on trajectory
            for angle, r_val, theta_val in zip(mean_chroma.index, mean_chroma, np.radians(mean_hue)):
                ax.annotate(f"{int(angle)}°", (theta_val, r_val), 
                           xytext=(10, 10), textcoords='offset points', 
                           fontsize=8, alpha=0.8, zorder=11,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        else:
            print(f"Warning: No data found for module {module}")
        
        ax.set_title(f"{MODULE_DESCRIPTIONS[module]}\nColor Trajectory Across All Angles", fontsize=14, pad=60)
        ax.grid(False)
        ax.set_ylim(0, 100)
        
        # Add colorbar only if scatter plot exists
        if sc is not None:
            plt.colorbar(sc, ax=ax, pad=0.1, label="Lightness (L*)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'blue_module_{module:02d}_trajectory.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_blue_modules_by_angle_with_lightness(df, output_dir=None):
    """
    Create overlayed CIELCH polar plot showing all blue modules with different symbols for angles
    and lightness as color, providing more information without connecting lines.
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'blue_modules_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    blue_modules = SUBGROUPS['blue']
    all_angles = sorted(df['Theta_i'].unique())
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 10))
    ax.set_facecolor("white")
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    
    # Create Lab color wheel background
    theta = np.linspace(0, 2 * np.pi, 360)
    r = np.linspace(0, 100, 200)
    T, R = np.meshgrid(theta, r)
    RGB = create_lab_ab_color_wheel_background(T, R, L_fixed=70)
    ax.pcolormesh(T, R, np.ones_like(R), color=RGB.reshape(-1, 3), shading='auto')
    
    # Add chroma circles and labels
    chroma_levels = np.arange(20, 101, 20)
    for c in chroma_levels:
        ax.plot(np.linspace(0, 2 * np.pi, 500), np.full(500, c), '--', color='gray', lw=0.8, alpha=0.7, zorder=1)
    ax.set_yticklabels([])
    label_angle = np.radians(80)
    for c in chroma_levels:
        ax.text(label_angle, c, f"{c}", color='black', fontsize=9, va='center', ha='left', alpha=0.7)
    
    # Add a*/b* axis labels
    axis_label_radius = 110
    offsets = [15, 15, 15, 15]
    axis_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    axis_labels = ['a*', 'b*', '-a*', '-b*']
    axis_xy = [
        (axis_label_radius + offsets[0], 0),
        (0, axis_label_radius + offsets[1]),
        (-(axis_label_radius + offsets[2]), 0),
        (0, -(axis_label_radius + offsets[3]))
    ]
    for angle, label, (x, y) in zip(axis_angles, axis_labels, axis_xy):
        ax.plot([angle, angle], [0, 100], color='k', linestyle=':', lw=1, alpha=0.7, zorder=2)
        ax.text(np.arctan2(y, x), np.hypot(x, y), label, fontsize=13, color='k', ha='center', va='center', fontweight='bold', rotation=0)
    
    # Define different symbols for different angles
    symbols = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '8', '<', '>', 'p', 'H', 'd', '|', '_']
    
    # Get overall lightness range for consistent colormap
    blue_data = df[df['Module'].isin(blue_modules)]
    overall_norm = mcolors.Normalize(vmin=blue_data['L'].min() - 5, vmax=blue_data['L'].max() + 5)
    cmap = plt.get_cmap('viridis')
    
    # Plot for each incident angle with different symbols
    for angle_idx, angle in enumerate(all_angles):
        symbol = symbols[angle_idx % len(symbols)]
        
        # Plot data for each module at this incident angle
        for module_idx, module in enumerate(blue_modules):
            module_data = df[df['Module'] == module]
            angle_data = module_data[module_data['Theta_i'] == angle]
            
            if not angle_data.empty:
                hue_rad = np.radians(angle_data['Hue'])
                chroma = angle_data['Chroma']
                lightness = angle_data['L']
                
                # Use different symbols for angles, lightness as color, no connecting lines
                ax.scatter(hue_rad, chroma, 
                          c=lightness,
                          cmap=cmap,
                          norm=overall_norm,
                          marker=symbol,
                          s=20, 
                          alpha=0.8, 
                          edgecolors='black',
                          linewidth=0.3,
                          label=f'θᵢ={int(angle)}°' if module_idx == 0 else "",
                          zorder=5)
    
    ax.set_title("Blue Modules: Data Points by Incident Angle\n(Symbols = angles, Color = lightness)", fontsize=16, pad=60)
    ax.grid(False)
    ax.set_ylim(0, 100)
    
    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9)
    
    # Add colorbar for lightness
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=overall_norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.1, label="Lightness (L*)")
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    plt.savefig(os.path.join(output_dir, 'blue_modules_by_angle_with_lightness.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_blue_modules_comparison_summary(df, output_dir=None):
    """
    Create a summary comparison plot showing all blue modules side by side for easy comparison.
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'blue_modules_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    blue_modules = SUBGROUPS['blue']
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': 'polar'})
    axes = axes.flatten()
    
    # Define symbols and colors for different incident angles
    symbols = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '8', '<', '>', 'p', 'H', 'd', '|', '_']
    grey_colors = ['#000000', '#333333', '#666666', '#999999', '#CCCCCC', '#FFFFFF', '#1A1A1A', '#4D4D4D']
    all_angles = sorted(df['Theta_i'].unique())
    
    for idx, module in enumerate(blue_modules):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        ax.set_facecolor("white")
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        
        # Create Lab color wheel background
        theta = np.linspace(0, 2 * np.pi, 360)
        r = np.linspace(0, 100, 200)
        T, R = np.meshgrid(theta, r)
        RGB = create_lab_ab_color_wheel_background(T, R, L_fixed=70)
        ax.pcolormesh(T, R, np.ones_like(R), color=RGB.reshape(-1, 3), shading='auto')
        
        # Add chroma circles and labels
        chroma_levels = np.arange(20, 101, 20)
        for c in chroma_levels:
            ax.plot(np.linspace(0, 2 * np.pi, 500), np.full(500, c), '--', color='gray', lw=0.8, alpha=0.7, zorder=1)
        ax.set_yticklabels([])
        
        # Add a*/b* axis labels
        axis_label_radius = 110
        offsets = [15, 15, 15, 15]
        axis_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        axis_labels = ['a*', 'b*', '-a*', '-b*']
        axis_xy = [
            (axis_label_radius + offsets[0], 0),
            (0, axis_label_radius + offsets[1]),
            (-(axis_label_radius + offsets[2]), 0),
            (0, -(axis_label_radius + offsets[3]))
        ]
        for angle, label, (x, y) in zip(axis_angles, axis_labels, axis_xy):
            ax.plot([angle, angle], [0, 100], color='k', linestyle=':', lw=1, alpha=0.7, zorder=2)
            ax.text(np.arctan2(y, x), np.hypot(x, y), label, fontsize=10, color='k', ha='center', va='center', fontweight='bold', rotation=0)
        
        # Get module data
        module_data = df[df['Module'] == module]
        if not module_data.empty:
            # Plot data for each incident angle
            for angle_idx, angle in enumerate(all_angles):
                angle_data = module_data[module_data['Theta_i'] == angle]
                if not angle_data.empty:
                    hue_rad = np.radians(angle_data['Hue'])
                    chroma = angle_data['Chroma']
                    
                    symbol = symbols[angle_idx % len(symbols)]
                    color = grey_colors[angle_idx % len(grey_colors)]
                    
                    ax.scatter(hue_rad, chroma, 
                              c=color,
                              marker=symbol,
                              s=20, 
                              alpha=0.8, 
                              edgecolors='black',
                              linewidth=0.3,
                              zorder=5)
        
        ax.set_title(f"{MODULE_DESCRIPTIONS[module]}", fontsize=12, pad=40)
        ax.grid(False)
        ax.set_ylim(0, 100)
    
    # Hide unused subplots
    for idx in range(len(blue_modules), len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall legend
    legend_elements = []
    for angle_idx, angle in enumerate(all_angles):
        symbol = symbols[angle_idx % len(symbols)]
        color = grey_colors[angle_idx % len(grey_colors)]
        legend_elements.append(plt.Line2D([0], [0], marker=symbol, color='w', 
                                        markerfacecolor=color, markersize=8, 
                                        label=f'θᵢ={int(angle)}°'))
    
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)
    
    plt.suptitle("Blue Modules Comparison: Data Points by Incident Angle", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'blue_modules_comparison_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_lightness_filtering_analysis(df, output_dir=None):
    """
    Analyze and visualize the effect of lightness filtering on blue modules.
    Shows how much data gets removed with different L* thresholds.
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'blue_modules_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    blue_modules = SUBGROUPS['blue']
    lightness_thresholds = [20, 25, 30, 35, 40, 45, 50]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot 1: Data retention vs lightness threshold
    retention_data = []
    for threshold in lightness_thresholds:
        total_points = len(df[df['Module'].isin(blue_modules)])
        filtered_points = len(df[(df['Module'].isin(blue_modules)) & (df['L'] >= threshold)])
        retention = (filtered_points / total_points) * 100
        retention_data.append(retention)
    
    axes[0].plot(lightness_thresholds, retention_data, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Lightness Threshold (L*)')
    axes[0].set_ylabel('Data Retention (%)')
    axes[0].set_title('Data Retention vs Lightness Threshold\n(Blue Modules)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 105)
    
    # Plot 2: Lightness distribution for blue modules
    blue_data = df[df['Module'].isin(blue_modules)]
    axes[1].hist(blue_data['L'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1].set_xlabel('Lightness (L*)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Lightness Distribution\n(Blue Modules)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Before/after filtering comparison (L* >= 30)
    threshold = 30
    original_data = blue_data
    filtered_data = blue_data[blue_data['L'] >= threshold]
    
    # Original data
    axes[2].scatter(original_data['a'], original_data['b'], 
                   c=original_data['L'], cmap='viridis', 
                   s=30, alpha=0.6, label=f'Original ({len(original_data)} points)')
    axes[2].set_xlabel('a*')
    axes[2].set_ylabel('b*')
    axes[2].set_title(f'Before/After Lightness Filtering (L* ≥ {threshold})')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Filtered data
    axes[2].scatter(filtered_data['a'], filtered_data['b'], 
                   c=filtered_data['L'], cmap='viridis', 
                   s=50, alpha=0.8, edgecolors='red', linewidth=1,
                   label=f'Filtered ({len(filtered_data)} points)')
    axes[2].legend()
    
    # Plot 4: Module-wise data retention
    module_retention = []
    module_names = []
    for module in blue_modules:
        module_data = df[df['Module'] == module]
        total = len(module_data)
        filtered = len(module_data[module_data['L'] >= threshold])
        retention = (filtered / total) * 100 if total > 0 else 0
        module_retention.append(retention)
        module_names.append(f"Module {module}")
    
    bars = axes[3].bar(module_names, module_retention, color='skyblue', edgecolor='black')
    axes[3].set_xlabel('Module')
    axes[3].set_ylabel('Data Retention (%)')
    axes[3].set_title(f'Data Retention by Module\n(L* ≥ {threshold})')
    axes[3].tick_params(axis='x', rotation=45)
    axes[3].grid(True, alpha=0.3, axis='y')
    axes[3].set_ylim(0, 105)
    
    # Add value labels on bars
    for bar, value in zip(bars, module_retention):
        axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lightness_filtering_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_hue_trajectory_comparison(df, output_dir=None):
    """
    Compare hue trajectories across blue modules to see if they follow similar patterns.
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'blue_modules_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    blue_modules = SUBGROUPS['blue']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot 1: Hue vs Angle of Incidence
    for idx, module in enumerate(blue_modules):
        module_data = df[df['Module'] == module]
        if not module_data.empty:
            grouped = module_data.groupby('Theta_i')
            mean_hue = (np.degrees(np.arctan2(grouped['b'].mean(), grouped['a'].mean())) + 360) % 360
            
            axes[0].plot(mean_hue.index, mean_hue, 
                        marker='o', linewidth=2, markersize=6,
                        color=colors[idx % len(colors)],
                        label=MODULE_DESCRIPTIONS[module])
    
    axes[0].set_xlabel('Angle of Incidence (θᵢ)')
    axes[0].set_ylabel('Hue (degrees)')
    axes[0].set_title('Hue Trajectory Comparison\n(Blue Modules)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Chroma vs Angle of Incidence
    for idx, module in enumerate(blue_modules):
        module_data = df[df['Module'] == module]
        if not module_data.empty:
            grouped = module_data.groupby('Theta_i')
            mean_a = grouped['a'].mean()
            mean_b = grouped['b'].mean()
            mean_chroma = np.sqrt(mean_a**2 + mean_b**2)
            
            axes[1].plot(mean_chroma.index, mean_chroma, 
                        marker='s', linewidth=2, markersize=6,
                        color=colors[idx % len(colors)],
                        label=MODULE_DESCRIPTIONS[module])
    
    axes[1].set_xlabel('Angle of Incidence (θᵢ)')
    axes[1].set_ylabel('Chroma')
    axes[1].set_title('Chroma Trajectory Comparison\n(Blue Modules)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Lightness vs Angle of Incidence
    for idx, module in enumerate(blue_modules):
        module_data = df[df['Module'] == module]
        if not module_data.empty:
            grouped = module_data.groupby('Theta_i')
            mean_L = grouped['L'].mean()
            
            axes[2].plot(mean_L.index, mean_L, 
                        marker='^', linewidth=2, markersize=6,
                        color=colors[idx % len(colors)],
                        label=MODULE_DESCRIPTIONS[module])
    
    axes[2].set_xlabel('Angle of Incidence (θᵢ)')
    axes[2].set_ylabel('Lightness (L*)')
    axes[2].set_title('Lightness Trajectory Comparison\n(Blue Modules)')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Hue-Chroma relationship
    for idx, module in enumerate(blue_modules):
        module_data = df[df['Module'] == module]
        if not module_data.empty:
            grouped = module_data.groupby('Theta_i')
            mean_a = grouped['a'].mean()
            mean_b = grouped['b'].mean()
            mean_chroma = np.sqrt(mean_a**2 + mean_b**2)
            mean_hue = (np.degrees(np.arctan2(mean_b, mean_a)) + 360) % 360
            
            scatter = axes[3].scatter(mean_hue, mean_chroma, 
                                    c=mean_L.index, cmap='viridis',
                                    s=100, alpha=0.7,
                                    label=MODULE_DESCRIPTIONS[module])
            
            # Connect points with lines
            axes[3].plot(mean_hue, mean_chroma, 
                        color=colors[idx % len(colors)], 
                        linewidth=1, alpha=0.5)
    
    axes[3].set_xlabel('Hue (degrees)')
    axes[3].set_ylabel('Chroma')
    axes[3].set_title('Hue-Chroma Relationship\n(Blue Modules)')
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)
    
    # Add colorbar for angle information
    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array([])
    plt.colorbar(sm, ax=axes[3], pad=0.1, label="Angle of Incidence (θᵢ)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hue_trajectory_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def fit_trajectory_line_for_angle(angle_data, method='linear'):
    """
    Fit a trajectory line through raw data points for a specific incident angle.
    
    Parameters:
    - angle_data: DataFrame with columns ['a', 'b', 'L', 'Hue', 'Chroma']
    - method: 'linear' for linear fit, 'polynomial' for polynomial fit
    
    Returns:
    - fitted_hue: array of fitted hue values
    - fitted_chroma: array of fitted chroma values
    """
    if len(angle_data) < 2:
        return None, None
    
    # Extract a* and b* coordinates
    a_coords = angle_data['a'].values
    b_coords = angle_data['b'].values
    
    if method == 'linear':
        # Fit a line through a* vs b* points
        if len(a_coords) >= 2:
            # Use linear regression to fit line through a*-b* space
            coeffs = np.polyfit(a_coords, b_coords, 1)
            a_fit = np.linspace(a_coords.min() - 1, a_coords.max() + 1, 50)
            b_fit = coeffs[0] * a_fit + coeffs[1]
            
            # Convert back to polar coordinates
            fitted_chroma = np.sqrt(a_fit**2 + b_fit**2)
            fitted_hue = (np.degrees(np.arctan2(b_fit, a_fit)) + 360) % 360
            
            return fitted_hue, fitted_chroma
    
    elif method == 'polynomial':
        # Fit a polynomial through a* vs b* points
        if len(a_coords) >= 3:
            coeffs = np.polyfit(a_coords, b_coords, min(2, len(a_coords) - 1))
            a_fit = np.linspace(a_coords.min() - 1, a_coords.max() + 1, 50)
            b_fit = np.polyval(coeffs, a_fit)
            
            # Convert back to polar coordinates
            fitted_chroma = np.sqrt(a_fit**2 + b_fit**2)
            fitted_hue = (np.degrees(np.arctan2(b_fit, a_fit)) + 360) % 360
            
            return fitted_hue, fitted_chroma
    
    return None, None

def plot_blue_modules_with_fitted_trajectories(df, output_dir=None):
    """
    Create overlayed CIELCH polar plot showing all blue modules with fitted trajectory lines
    for each incident angle through the raw data points.
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'blue_modules_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    blue_modules = SUBGROUPS['blue']
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 10))
    ax.set_facecolor("white")
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    
    # Create Lab color wheel background
    theta = np.linspace(0, 2 * np.pi, 360)
    r = np.linspace(0, 100, 200)
    T, R = np.meshgrid(theta, r)
    RGB = create_lab_ab_color_wheel_background(T, R, L_fixed=70)
    ax.pcolormesh(T, R, np.ones_like(R), color=RGB.reshape(-1, 3), shading='auto')
    
    # Add chroma circles and labels
    chroma_levels = np.arange(20, 101, 20)
    for c in chroma_levels:
        ax.plot(np.linspace(0, 2 * np.pi, 500), np.full(500, c), '--', color='gray', lw=0.8, alpha=0.7, zorder=1)
    ax.set_yticklabels([])
    label_angle = np.radians(80)
    for c in chroma_levels:
        ax.text(label_angle, c, f"{c}", color='black', fontsize=9, va='center', ha='left', alpha=0.7)
    
    # Add a*/b* axis labels
    axis_label_radius = 110
    offsets = [15, 15, 15, 15]
    axis_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    axis_labels = ['a*', 'b*', '-a*', '-b*']
    axis_xy = [
        (axis_label_radius + offsets[0], 0),
        (0, axis_label_radius + offsets[1]),
        (-(axis_label_radius + offsets[2]), 0),
        (0, -(axis_label_radius + offsets[3]))
    ]
    for angle, label, (x, y) in zip(axis_angles, axis_labels, axis_xy):
        ax.plot([angle, angle], [0, 100], color='k', linestyle=':', lw=1, alpha=0.7, zorder=2)
        ax.text(np.arctan2(y, x), np.hypot(x, y), label, fontsize=13, color='k', ha='center', va='center', fontweight='bold', rotation=0)
    
    # Color scheme for blue modules
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
    markers = ['o', 's', '^', 'D', 'v']
    
    # Get all unique incident angles
    all_angles = sorted(df['Theta_i'].unique())
    
    # Plot for each blue module
    for idx, module in enumerate(blue_modules):
        module_data = df[df['Module'] == module]
        if not module_data.empty:
            # Plot raw data points with transparency
            hue_rad = np.radians(module_data['Hue'])
            chroma = module_data['Chroma']
            lightness = module_data['L']
            
            ax.scatter(hue_rad, chroma, 
                      c=lightness, 
                      cmap='viridis', 
                      marker=markers[idx % len(markers)],
                      s=20, 
                      alpha=0.4, 
                      edgecolors=colors[idx % len(colors)],
                      linewidth=0.3,
                      label=f"{MODULE_DESCRIPTIONS[module]} (raw)",
                      zorder=5)
            
            # Fit and plot trajectory lines for each incident angle
            for angle in all_angles:
                angle_data = module_data[module_data['Theta_i'] == angle]
                if len(angle_data) >= 2:  # Need at least 2 points to fit a line
                    fitted_hue, fitted_chroma = fit_trajectory_line_for_angle(angle_data, method='linear')
                    
                    if fitted_hue is not None and fitted_chroma is not None:
                        # Plot fitted trajectory line
                        ax.plot(np.radians(fitted_hue), fitted_chroma, 
                               color=colors[idx % len(colors)], 
                               linewidth=1.5, 
                               alpha=0.7,
                               zorder=6)
                        
                        # Add angle label at the midpoint of the trajectory
                        mid_idx = len(fitted_hue) // 2
                        ax.annotate(f"{int(angle)}°", 
                                   (np.radians(fitted_hue[mid_idx]), fitted_chroma[mid_idx]),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=7, alpha=0.8, zorder=11,
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax.set_title("Blue Modules: Raw Data with Fitted Trajectories\n(One Line per Incident Angle)", fontsize=16, pad=60)
    ax.grid(False)
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9)
    
    # Add colorbar for lightness
    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.1, label="Lightness (L*)")
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    plt.savefig(os.path.join(output_dir, 'blue_modules_fitted_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_individual_module_fitted_trajectories(df, output_dir=None):
    """
    Create individual CIELCH polar plots for each blue module showing data points
    for each incident angle with different symbols and lightness coloring.
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'blue_modules_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    blue_modules = SUBGROUPS['blue']
    all_angles = sorted(df['Theta_i'].unique())
    
    # Define symbols for different incident angles
    symbols = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '8', '<', '>', 'p', 'H', 'd', '|', '_']
    
    for module in blue_modules:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))
        ax.set_facecolor("white")
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        
        # Create Lab color wheel background
        theta = np.linspace(0, 2 * np.pi, 360)
        r = np.linspace(0, 100, 200)
        T, R = np.meshgrid(theta, r)
        RGB = create_lab_ab_color_wheel_background(T, R, L_fixed=70)
        ax.pcolormesh(T, R, np.ones_like(R), color=RGB.reshape(-1, 3), shading='auto')
        
        # Add chroma circles and labels
        chroma_levels = np.arange(20, 101, 20)
        for c in chroma_levels:
            ax.plot(np.linspace(0, 2 * np.pi, 500), np.full(500, c), '--', color='gray', lw=0.8, alpha=0.7, zorder=1)
        ax.set_yticklabels([])
        label_angle = np.radians(80)
        for c in chroma_levels:
            ax.text(label_angle, c, f"{c}", color='black', fontsize=9, va='center', ha='left', alpha=0.7)
        
        # Add a*/b* axis labels
        axis_label_radius = 110
        offsets = [15, 15, 15, 15]
        axis_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        axis_labels = ['a*', 'b*', '-a*', '-b*']
        axis_xy = [
            (axis_label_radius + offsets[0], 0),
            (0, axis_label_radius + offsets[1]),
            (-(axis_label_radius + offsets[2]), 0),
            (0, -(axis_label_radius + offsets[3]))
        ]
        for angle, label, (x, y) in zip(axis_angles, axis_labels, axis_xy):
            ax.plot([angle, angle], [0, 100], color='k', linestyle=':', lw=1, alpha=0.7, zorder=2)
            ax.text(np.arctan2(y, x), np.hypot(x, y), label, fontsize=13, color='k', ha='center', va='center', fontweight='bold', rotation=0)
        
        # Get module data
        module_data = df[df['Module'] == module]
        if not module_data.empty:
            # Get overall lightness range for consistent colormap
            overall_norm = mcolors.Normalize(vmin=0, vmax=100)
            cmap = plt.get_cmap('gray')
            
            # Plot data for each incident angle with different symbols and lightness coloring
            for idx, angle in enumerate(all_angles):
                angle_data = module_data[module_data['Theta_i'] == angle]
                if not angle_data.empty:
                    hue_rad = np.radians(angle_data['Hue'])
                    chroma = angle_data['Chroma']
                    lightness = angle_data['L']
                    
                    # Use different symbol for each angle
                    symbol = symbols[idx % len(symbols)]
                    
                    # Plot data points with lightness coloring
                    ax.scatter(hue_rad, chroma, 
                              c=lightness,
                              cmap=cmap,
                              norm=overall_norm,
                              marker=symbol,
                              s=20,  # Smaller symbols
                              alpha=0.6,  # Semi-transparent to show overlap
                              edgecolors='black',
                              linewidth=0.3,
                              label=f'θᵢ={int(angle)}°',
                              zorder=5)
        
        ax.set_title(f"{MODULE_DESCRIPTIONS[module]}\nData Points by Incident Angle (Symbols = angles, Color = lightness)", fontsize=14, pad=60)
        ax.grid(False)
        ax.set_ylim(0, 100)
        
        # Add legend with lightness information
        legend_elements = []
        for idx, angle in enumerate(all_angles):
            symbol = symbols[idx % len(symbols)]
            legend_elements.append(plt.Line2D([0], [0], marker=symbol, color='w', 
                                            markerfacecolor='gray', markersize=8, 
                                            label=f'θᵢ={int(angle)}°'))
        
        # Add lightness information to legend
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                        markerfacecolor='white', markersize=8, 
                                        label='L* > 80 (white)'))
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                        markerfacecolor='gray', markersize=8, 
                                        label='L* 20-80 (grey)'))
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                        markerfacecolor='black', markersize=8, 
                                        label='L* < 20 (black)'))
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9)
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.97])
        plt.savefig(os.path.join(output_dir, f'blue_module_{module:02d}_connected_data.png'), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    print("Generating CIELCH polar plots for blue modules analysis...")
    
    # Generate improved blue modules analysis
    print("1. Creating overlayed raw data plot (improved)...")
    plot_blue_modules_overlayed_raw(df)
    
    print("2. Creating overlayed plot with lightness as color...")
    plot_blue_modules_by_angle_with_lightness(df)
    
    print("3. Creating comparison summary plot...")
    plot_blue_modules_comparison_summary(df)
    
    print("4. Creating individual trajectory plots...")
    plot_blue_modules_individual_trajectories(df)
    
    print("5. Creating individual connected data plots...")
    plot_individual_module_fitted_trajectories(df)
    
    print("6. Analyzing lightness filtering effects...")
    plot_lightness_filtering_analysis(df)
    
    print("7. Comparing hue trajectories...")
    plot_hue_trajectory_comparison(df)
    
    # Original plots (keeping for reference)
    print("8. Generating original subgroup plots...")
    for group_name, modules in SUBGROUPS.items():
        plot_cielch_polar_trajectory(df, modules, group_name)
    
    print("9. Generating example plots...")
    # Example: plot multiple modules from the 'blue' group at 60°
    plot_cielch_polar_by_angle(
        df=df,
        modules=SUBGROUPS['blue'],
        theta_i=60,
        title="Blue Modules (60°)",
        outname="cielch_polar_blue_60deg"
    )

    # Example: plot a single module at 15°
    plot_cielch_polar_by_angle(
        df=df,
        modules=[17],
        theta_i=15,
        title="Module 17 (15°)",
        outname="cielch_polar_Module17_15deg"
    )

    # Example: plot the full trajectory of a single module
    plot_cielch_polar_module_trajectory(
        df=df,
        module_id=17,
        title="Module 17 Trajectory",
        outname="cielch_polar_module17_trajectory"
    )

    # Example: create a zoomed-in snippet
    plot_cielch_zoom_snippet(
        df=df,
        module_id=17,
        outname="cielch_snippet_module17"
    )

    # Example: plot all modules at 45°
    plot_all_modules_at_45(df)
    
    print("All plots generated successfully!") 