"""
Simplified Blue Modules Analysis - Without Complex Color Wheel
This script generates the essential plots for blue module analysis with simplified background.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')

# Configuration
OUTPUT_DIR = "cielch/blue_modules_analysis_simple"
BLUE_MODULES = [1, 9, 10, 11, 12, 13]
MODULE_DESCRIPTIONS = {
    1: "Module REF",
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

# Verify the conversion worked
print(f"Data loaded successfully: {df.shape}")
print(f"Module column type: {df['Module'].dtype}")
print(f"Available modules: {sorted(df['Module'].unique())}")
print(f"Module 1 data points: {len(df[df['Module'] == 1])}")

def create_simple_lch_polar_plot(ax, title=""):
    """Create a simple LCH polar plot without complex color wheel."""
    ax.set_facecolor("white")
    ax.set_theta_zero_location('E')  # 0° at right (East, +a*)
    ax.set_theta_direction(1)        # Counterclockwise
    
    # Add simple chroma circles
    chroma_levels = np.arange(20, 101, 20)
    for c in chroma_levels:
        ax.plot(np.linspace(0, 2 * np.pi, 500), np.full(500, c), '--', color='gray', lw=0.8, alpha=0.7, zorder=1)
    
    # Add a*/b* axis labels
    axis_label_radius = 110
    axis_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    axis_labels = ['a*', 'b*', '-a*', '-b*']
    axis_xy = [
        (axis_label_radius + 15, 0),
        (0, axis_label_radius + 15),
        (-(axis_label_radius + 15), 0),
        (0, -(axis_label_radius + 15))
    ]
    for angle, label, (x, y) in zip(axis_angles, axis_labels, axis_xy):
        ax.plot([angle, angle], [0, 100], color='k', linestyle=':', lw=1, alpha=0.7, zorder=2)
        ax.text(np.arctan2(y, x), np.hypot(x, y), label, fontsize=13, color='k', ha='center', va='center', fontweight='bold', rotation=0)
    
    if title:
        ax.set_title(title, fontsize=14, pad=60)
    
    ax.grid(False)
    ax.set_ylim(0, 100)
    
    return ax

def plot_blue_module_connected_data(df, module_id, output_dir):
    """Create connected data plot for a single blue module."""
    print(f"    Creating plot for {MODULE_DESCRIPTIONS[module_id]}...")
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))
    
    # Create simple LCH plot
    create_simple_lch_polar_plot(ax, f"{MODULE_DESCRIPTIONS[module_id]}\nData Points by Incident Angle")
    
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
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    plt.savefig(os.path.join(output_dir, f'blue_module_{module_id:02d}_connected_data.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Plot saved for {MODULE_DESCRIPTIONS[module_id]}")

def plot_blue_modules_overlayed_raw(df, output_dir):
    """Create overlayed plot showing all blue modules."""
    print("    Creating overlayed plot...")
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 10))
    
    # Create simple LCH plot
    create_simple_lch_polar_plot(ax, "Blue Modules: Data Points by Incident Angle")
    
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
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    plt.savefig(os.path.join(output_dir, 'blue_modules_overlayed_raw.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Overlayed plot saved")

def main():
    """Main function to generate all required plots."""
    print("BLUE MODULES ANALYSIS - SIMPLIFIED VERSION")
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
    
    print("\nAll plots generated successfully!")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for module in BLUE_MODULES:
        print(f"  - blue_module_{module:02d}_connected_data.png")
    print("  - blue_modules_overlayed_raw.png")

if __name__ == '__main__':
    main()
