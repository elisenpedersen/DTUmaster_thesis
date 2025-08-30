"""
Clean Green Modules Analysis - New Version
This script generates only the essential plots for green module analysis:
1. Connected data plots for each green module
2. Overlayed raw data plot
3. Hue trajectory comparison plot

All plots use the same high-quality LCH polar plot function to ensure consistency.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors as mcolors
from colour import Lab_to_XYZ, XYZ_to_sRGB

# Configuration
OUTPUT_DIR = "LAB/all_modules_lab_analysis/HUE_chroma/cielch"
GREEN_MODULES = [24, 25, 26, 27, 28, 29]
MODULE_DESCRIPTIONS = {
    24: "Module 24 (GreenC)", 
    25: "Module 25 (GreenCx)", 
    26: "Module 26 (GreenBaC)", 
    27: "Module 27 (GreenBaCx)", 
    28: "Module 28 (GreenGlCx)", 
    29: "Module 29 (GreenGlC)"
}

# Load data
CSV_PATH = 'LAB/all_modules_lab_analysis/HUE_chroma/cielab/only_lightness_correction/hue_chroma_lab_results_lightness_allcolors.csv'
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

def create_lch_polar_plot(ax, title="", show_legend=True):
    """
    Create a standardized LCH polar plot with color wheel background.
    This is the core plotting function used by all plots.
    """
    ax.set_facecolor("white")
    ax.set_theta_zero_location('E')  # 0° at right (East, +a*)
    ax.set_theta_direction(1)        # Counterclockwise
    
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
    
    if title:
        ax.set_title(title, fontsize=14, pad=60)
    
    ax.grid(False)
    ax.set_ylim(0, 100)
    
    return ax

def plot_green_module_connected_data(df, module_id, output_dir):
    """
    Create connected data plot for a single green module.
    This is the 'perfect plot' that you referenced.
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))
    
    # Create standardized LCH plot
    create_lch_polar_plot(ax, f"{MODULE_DESCRIPTIONS[module_id]}\nData Points by Incident Angle (Symbols = angles, Color = lightness)")
    
    # Get module data
    module_data = df[df['Module'] == module_id]
    if not module_data.empty:
        all_angles = sorted(module_data['Theta_i'].unique())
        
        # Define symbols for different incident angles
        symbols = ['o', 's', '^', 'D', 'v']
        
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
    plt.savefig(os.path.join(output_dir, f'green_module_{module_id:02d}_connected_data.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_green_modules_overlayed_raw(df, output_dir):
    """
    Create overlayed CIELCH polar plot showing all green modules with different grey colors for incident angles.
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 10))
    
    # Create standardized LCH plot
    create_lch_polar_plot(ax, "Green Modules: Data Points by Incident Angle\n(Grey colors = angles, Circles = all modules)")
    
    all_angles = sorted(df['Theta_i'].unique())
    
    # Use consistent symbols (circles) and better contrast grey colors
    grey_colors = ['#000000', '#333333', '#666666', '#999999', '#CCCCCC', '#FFFFFF', '#1A1A1A', '#4D4D4D']
    
    # Plot for each incident angle with different grey colors
    for angle_idx, angle in enumerate(all_angles):
        angle_color = grey_colors[angle_idx % len(grey_colors)]
        
        # Plot data for each module at this incident angle
        for module_idx, module in enumerate(GREEN_MODULES):
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
    
    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    plt.savefig(os.path.join(output_dir, 'green_modules_overlayed_raw.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_hue_trajectory_comparison(df, output_dir):
    """
    Compare hue trajectories across green modules to see if they follow similar patterns.
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
    
    # Plot 1: Hue vs Angle of Incidence
    for idx, module in enumerate(GREEN_MODULES):
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
    axes[0].set_title('Hue Trajectory Comparison\n(Green Modules)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Chroma vs Angle of Incidence
    for idx, module in enumerate(GREEN_MODULES):
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
    axes[1].set_title('Chroma Trajectory Comparison\n(Green Modules)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Lightness vs Angle of Incidence
    for idx, module in enumerate(GREEN_MODULES):
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
    axes[2].set_title('Lightness Trajectory Comparison\n(Green Modules)')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Hue-Chroma relationship
    for idx, module in enumerate(GREEN_MODULES):
        module_data = df[df['Module'] == module]
        if not module_data.empty:
            grouped = module_data.groupby('Theta_i')
            mean_a = grouped['a'].mean()
            mean_b = grouped['b'].mean()
            mean_chroma = np.sqrt(mean_a**2 + mean_b**2)
            mean_hue = (np.degrees(np.arctan2(mean_b, mean_a)) + 360) % 360
            mean_L = grouped['L'].mean()
            
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
    axes[3].set_title('Hue-Chroma Relationship\n(Green Modules)')
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)
    
    # Add colorbar for angle information
    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array([])
    plt.colorbar(sm, ax=axes[3], pad=0.1, label="Angle of Incidence (θᵢ)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hue_trajectory_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to generate all required plots."""
    print("GREEN MODULES ANALYSIS - NEW VERSION")
    print("=" * 50)
    
    # Create output directory
    output_dir = os.path.join(OUTPUT_DIR, 'green_modules_analysis_new')
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. Generating connected data plots for each green module...")
    for module in GREEN_MODULES:
        print(f"   Processing {MODULE_DESCRIPTIONS[module]}...")
        plot_green_module_connected_data(df, module, output_dir)
    
    print("2. Generating overlayed raw data plot...")
    plot_green_modules_overlayed_raw(df, output_dir)
    
    print("3. Generating hue trajectory comparison plot...")
    plot_hue_trajectory_comparison(df, output_dir)
    
    print("\nAll plots generated successfully!")
    print(f"Results saved in: {output_dir}")
    print("\nGenerated files:")
    for module in GREEN_MODULES:
        print(f"  - green_module_{module:02d}_connected_data.png")
    print("  - green_modules_overlayed_raw.png")
    print("  - hue_trajectory_comparison.png")

if __name__ == '__main__':
    main() 