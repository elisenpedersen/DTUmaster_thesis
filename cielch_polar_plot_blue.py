"""
Blue Modules Analysis - Original Working Version
This script generates CIELCH polar plots for blue module analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors as mcolors
from colour import Lab_to_XYZ, XYZ_to_sRGB

# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')

# Configuration
OUTPUT_DIR = "cielch/blue_modules_analysis"
BLUE_MODULES = [9, 10, 11, 12, 13]
MODULE_DESCRIPTIONS = {
    9: "Module 9 (Blue2.1S)", 
    10: "Module 10 (Blue2.1L)", 
    11: "Module 11 (Blue1.5S)", 
    12: "Module 12 (Blue1.5L)", 
    13: "Module 13 (BlueBa)"
}

# Load data
CSV_PATH = 'cielab/only_lightness_correction/hue_chroma_lab_results_lightness.csv'
df = pd.read_csv(CSV_PATH)

# Filter out non-numeric module entries and convert Module to numeric
df = df[pd.to_numeric(df['Module'], errors='coerce').notna()]
df['Module'] = pd.to_numeric(df['Module'])

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

def plot_blue_module_connected_data(df, module_id, output_dir):
    """Create connected data plot for a single blue module."""
    print(f"    Creating plot for {MODULE_DESCRIPTIONS[module_id]}...")
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))
    
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
    module_data = df[df['Module'] == module_id]
    if not module_data.empty:
        all_angles = sorted(module_data['Theta_i'].unique())
        
        # Define symbols for different incident angles
        symbols = ['o', 's', '^', 'D', 'v']
        
        # Get overall lightness range for consistent colormap
        overall_norm = plt.Normalize(vmin=0, vmax=100)
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
                          s=20,
                          alpha=0.6,
                          edgecolors='black',
                          linewidth=0.3,
                          label=f'θᵢ={int(angle)}°',
                          zorder=5)
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9)
    
    ax.set_title(f"{MODULE_DESCRIPTIONS[module_id]}\nData Points by Incident Angle", fontsize=14, pad=60)
    ax.grid(False)
    ax.set_ylim(0, 100)
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    plt.savefig(os.path.join(output_dir, f'blue_module_{module_id:02d}_connected_data.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Plot saved for {MODULE_DESCRIPTIONS[module_id]}")

def plot_blue_modules_overlayed_raw(df, output_dir):
    """Create overlayed plot showing all blue modules."""
    print("    Creating overlayed plot...")
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 10))
    
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
    
    all_angles = sorted(df['Theta_i'].unique())
    
    # Use consistent symbols (circles) and grey colors
    grey_colors = ['#000000', '#333333', '#666666', '#999999', '#CCCCCC']
    
    # Plot for each incident angle with different grey colors
    for angle_idx, angle in enumerate(all_angles):
        angle_color = grey_colors[angle_idx % len(grey_colors)]
        
        # Plot data for each module at this incident angle
        for module_idx, module in enumerate(BLUE_MODULES):
            module_data = df[df['Module'] == module]
            angle_data = module_data[module_data['Theta_i'] == angle]
            
            if not angle_data.empty:
                hue_rad = np.radians(angle_data['Hue'])
                chroma = angle_data['Chroma']
                
                ax.scatter(hue_rad, chroma, 
                          c=angle_color,
                          marker='o',
                          s=15,
                          alpha=0.8, 
                          edgecolors='black',
                          linewidth=0.2,
                          label=f'θᵢ={int(angle)}°' if module_idx == 0 else "",
                          zorder=5)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9)
    
    ax.set_title("Blue Modules: Data Points by Incident Angle", fontsize=16, pad=60)
    ax.grid(False)
    ax.set_ylim(0, 100)
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    plt.savefig(os.path.join(output_dir, 'blue_modules_overlayed_raw.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Overlayed plot saved")

def plot_hue_trajectory_comparison(df, output_dir):
    """Create hue trajectory comparison plot."""
    print("    Creating hue trajectory comparison...")
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 10))
    
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
    
    all_angles = sorted(df['Theta_i'].unique())
    
    # Plot trajectory for each blue module
    for module in BLUE_MODULES:
        module_data = df[df['Module'] == module]
        if not module_data.empty:
            # Group by incident angle and calculate mean
            grouped = module_data.groupby('Theta_i')
            mean_a = grouped['a'].mean()
            mean_b = grouped['b'].mean()
            mean_chroma = np.sqrt(mean_a**2 + mean_b**2)
            mean_hue = (np.degrees(np.arctan2(mean_b, mean_a)) + 360) % 360
            
            # Plot trajectory line
            hue_rad = np.radians(mean_hue)
            ax.plot(hue_rad, mean_chroma, 
                   marker='o', linewidth=2, markersize=6,
                   label=MODULE_DESCRIPTIONS[module], zorder=6)
            
            # Annotate angle points
            for angle, r_val, theta_val in zip(mean_chroma.index, mean_chroma, hue_rad):
                ax.annotate(f"{int(angle)}°", (theta_val, r_val), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, alpha=0.8, zorder=11)
    
    ax.set_title("Blue Modules: Hue Trajectory Comparison", fontsize=16, pad=60)
    ax.grid(False)
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    plt.savefig(os.path.join(output_dir, 'hue_trajectory_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Hue trajectory comparison saved")

def main():
    """Main function to generate all required plots."""
    print("BLUE MODULES ANALYSIS - ORIGINAL VERSION")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    print("1. Generating connected data plots for each blue module...")
    for module in BLUE_MODULES:
        print(f"   Processing {MODULE_DESCRIPTIONS[module]} (Module {module})...")
        plot_blue_module_connected_data(df, module, OUTPUT_DIR)
    
    print("2. Generating overlayed raw data plot...")
    plot_blue_modules_overlayed_raw(df, OUTPUT_DIR)
    
    print("3. Generating hue trajectory comparison plot...")
    plot_hue_trajectory_comparison(df, OUTPUT_DIR)
    
    print("\nAll plots generated successfully!")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for module in BLUE_MODULES:
        print(f"  - blue_module_{module:02d}_connected_data.png")
    print("  - blue_modules_overlayed_raw.png")
    print("  - hue_trajectory_comparison.png")

if __name__ == '__main__':
    main() 