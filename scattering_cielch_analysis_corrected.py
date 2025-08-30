import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colour import SpectralDistribution, sd_to_XYZ, XYZ_to_Lab, MSDS_CMFS, SDS_ILLUMINANTS, Lab_to_XYZ, XYZ_to_sRGB
from matplotlib import colors as mcolors

# CONFIGURATION - CHANGE THESE VALUES TO ANALYZE DIFFERENT MODULES
MODULE_ID = 1  # Change this to any module number you want
SINGLE_ANGLE_THETA_I = 30  # Change this to 0, 15, 30, 45, or 60 for single angle plot

ANGLES = [0, 15, 30, 45, 60]
OUTPUT_DIR = "LAB/scattering_cielch_corrected"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Module descriptions from reference script
MODULE_DESCRIPTIONS = {
    1: "REF", 2: "B5", 3: "B10", 4: "B20",
    5: "G2.1S5", 6: "G2.1S20", 7: "G1.5S5", 8: "G1.5S20",
    9: "Blue2.1S", 10: "Blue2.1L", 11: "Blue1.5S", 12: "Blue1.5L", 13: "BlueBa",
    14: "G2.1L5", 15: "G2.1L20", 16: "G1.5L5", 17: "G1.5L20",
    18: "BrownC", 20: "BrownBaC", 22: "BrownGlC",
    25: "GreenC", 27: "GreenBaC", 28: "GreenGlC",
}

# Angular zones for color grouping based on angular deviation from specular
ANGULAR_ZONES = {
    'specular': {'deviation_range': (0, 15), 'description': 'Specular (0-15° deviation)', 'color': 'white'},
    'forward_scatter': {'deviation_range': (15, 45), 'description': 'Forward scatter (15-45° deviation)', 'color': 'grey'},
    'backscatter': {'deviation_range': (45, 90), 'description': 'Backscatter (45-90° deviation)', 'color': 'black'},
    'residual': {'deviation_range': (90, 180), 'description': 'Residual scatter (>90° deviation)', 'color': 'red'}
}

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def spectrum_to_lab(wavelengths, spectrum):
    std_wl = np.arange(360, 781, 1)
    spectrum_interp = np.interp(std_wl, wavelengths, spectrum)
    sd = SpectralDistribution(dict(zip(std_wl, spectrum_interp)))
    cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    illuminant = SDS_ILLUMINANTS['D65']
    XYZ = sd_to_XYZ(sd, cmfs, illuminant)
    Lab = XYZ_to_Lab(XYZ / 100)
    return Lab

def calculate_angular_deviation_from_specular(theta_i, theta_r, phi_r):
    """
    Calculate the angular deviation from the ideal specular reflection direction.
    
    Args:
        theta_i: Incident angle in degrees
        theta_r: Reflection angle in degrees  
        phi_r: Reflection azimuth angle in degrees
    
    Returns:
        Angular deviation in degrees
    """
    # Convert to radians
    theta_i_rad = np.radians(theta_i)
    theta_r_rad = np.radians(theta_r)
    phi_r_rad = np.radians(phi_r)
    
    # For specular reflection:
    # - θᵣ = θᵢ (same zenith angle)
    # - φᵣ = 180° (opposite azimuth to φᵢ = 0°)
    
    # Specular direction vector: [sin(θᵢ), 0, cos(θᵢ)] rotated by 180° in xy-plane
    # This becomes: [-sin(θᵢ), 0, cos(θᵢ)]
    R_specular = np.array([
        -np.sin(theta_i_rad),  # x component (negative because φ = 180°)
        0.0,                   # y component (0 because φ = 180°)
        np.cos(theta_i_rad)    # z component (same as incident)
    ])
    
    # Measured reflection direction vector
    R_measured = np.array([
        np.sin(theta_r_rad) * np.cos(phi_r_rad),  # x component
        np.sin(theta_r_rad) * np.sin(phi_r_rad),  # y component  
        np.cos(theta_r_rad)                       # z component
    ])
    
    # Calculate dot product
    dot_product = np.dot(R_measured, R_specular)
    
    # Clamp dot product to valid range [-1, 1] to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate angular deviation
    angular_deviation = np.degrees(np.arccos(dot_product))
    
    return angular_deviation

def classify_reflection_by_deviation(angular_deviation):
    """Classify reflection based on angular deviation from specular."""
    classifications = {}
    
    for zone, config in ANGULAR_ZONES.items():
        min_deviation, max_deviation = config['deviation_range']
        classifications[f'zone_{zone}'] = min_deviation <= angular_deviation <= max_deviation
    
    return classifications

def calculate_weighted_zone_ratios(df):
    """
    Calculate the weighted ratio of reflected light going into each angular zone.

    Here:
df['total_brdf'] = The actual BRDF intensity at each measurement point
df[zone_col] = Boolean indicator (1 if point belongs to zone, 0 otherwise)
df[weighted_col] = Weighted contribution = intensity times zone membership
Why This Matters:
The total_brdf values represent the actual amount of light reflected in each direction. 
By weighting by these values, we're calculating:
"How much of the total reflected light goes into each angular zone?"
Rather than just counting measurement points
This gives us the true light distribution across specular, forward scatter, 
and backscatter regions, which is much more meaningful than just point counts!
The data represents real physical measurements of how different solar panel 
materials scatter light, and the weighting ensures we're analyzing the actual 
light intensity distribution rather than just spatial distribution of measurement points.
    
    Args:
        df: DataFrame with 'total_brdf' and zone classification columns
    
    Returns:
        Dictionary with zone ratios and summary statistics
    """
    # Create weighted columns for each zone
    weighted_data = {}
    zone_ratios = {}
    
    # Calculate weighted contributions for each zone
    for zone in ANGULAR_ZONES.keys():
        zone_col = f'zone_{zone}'
        weighted_col = f'weighted_{zone}'
        
        # Create weighted column: total_brdf * zone_indicator
        df[weighted_col] = df['total_brdf'] * df[zone_col]
        weighted_data[zone] = df[weighted_col]
        
        # Calculate ratio: sum of weighted values / sum of total brdf
        total_weighted = df[weighted_col].sum()
        total_brdf = df['total_brdf'].sum()
        
        if total_brdf > 0:
            ratio = total_weighted / total_brdf
        else:
            ratio = 0.0
            
        zone_ratios[zone] = {
            'ratio': ratio,
            'total_weighted': total_weighted,
            'total_brdf': total_brdf,
            'point_count': df[zone_col].sum()
        }
    
    return zone_ratios, weighted_data

def create_lab_ab_color_wheel_background(theta, r, L_fixed=70):
    a = r * np.cos(theta)
    b = r * np.sin(theta)
    Lab = np.stack([np.full_like(a, L_fixed), a, b], axis=-1)
    Lab_flat = Lab.reshape(-1, 3)
    xyz = np.array([Lab_to_XYZ(lab) for lab in Lab_flat])
    rgb = np.array([XYZ_to_sRGB(x) for x in xyz])
    rgb = np.clip(rgb, 0, 1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    return rgb.reshape(a.shape + (3,))

def analyze_module_data():
    """Analyze data for the specified module with corrected angular deviation classification."""
    all_lab = []
    
    for angle in ANGLES:
        pkl_path = os.path.join('..', 'brdf_plots', f'Module{MODULE_ID}', f'Module{MODULE_ID}_theta{angle}_phi0.pkl')
        if not os.path.exists(pkl_path):
            print(f"Missing: {pkl_path}")
            continue
                
        df = load_pkl(pkl_path)
        wavelengths = np.linspace(400, 700, len(df['spec_brdf'].iloc[0]))
        
        module_name = MODULE_DESCRIPTIONS.get(MODULE_ID, f'Module {MODULE_ID}')
        print(f"Processing {module_name}, θᵢ={angle}°: {len(df)} measurements")
        
        for idx, row in df.iterrows():
            theta_r = row['theta_r']
            phi_r = row.get('phi_r', 0)  # Default to 0 if not available
            spectrum = np.array(row['spec_brdf'])
            Lab = spectrum_to_lab(wavelengths, spectrum)
            
            # Calculate angular deviation from specular using dot product
            angular_deviation = calculate_angular_deviation_from_specular(angle, theta_r, phi_r)
            
            # Debug: Print some values for θᵢ = 30° to understand the issue
            # if angle == 30 and idx < 5:  # Print first 5 measurements for θᵢ = 30°
            #     print(f"  Debug θᵢ=30°: θᵣ={theta_r:.1f}°, φᵣ={phi_r:.1f}°, deviation={angular_deviation:.1f}°")
            
            # Enhanced filtering: avoid extreme specular artifacts
            #if angular_deviation <= 15 and Lab[0] < 10:  # Very low L* in specular region
            #    continue
            
            # Get angular classifications based on deviation
            classifications = classify_reflection_by_deviation(angular_deviation)
            
            # Calculate hue and chroma using the same method as CIELCH polar plot
            hue = (np.degrees(np.arctan2(Lab[2], Lab[1])) + 360) % 360
            chroma = np.sqrt(Lab[1]**2 + Lab[2]**2)
            
            # Store data with classifications
            data_point = {
                'Module': MODULE_ID,
                'Theta_i': angle,
                'Theta_r': theta_r,
                'Phi_r': phi_r,
                'Angular_Deviation': angular_deviation,
                'L': Lab[0],
                'a': Lab[1],
                'b': Lab[2],
                'Hue': hue,
                'Chroma': chroma,
                'total_brdf': row.get('total_brdf', np.nan)
            }
            
            # Add classification flags
            for classification, is_member in classifications.items():
                data_point[classification] = is_member
            
            all_lab.append(data_point)
    
    df_result = pd.DataFrame(all_lab)
    module_name = MODULE_DESCRIPTIONS.get(MODULE_ID, f'Module {MODULE_ID}')
    print(f"Total data points for {module_name}: {len(df_result)}")
    print(f"Data shape: {df_result.shape}")
    
    # Print angular deviation statistics
    if len(df_result) > 0:
        print(f"Angular deviation statistics:")
        print(f"  Mean: {df_result['Angular_Deviation'].mean():.2f}°")
        print(f"  Std: {df_result['Angular_Deviation'].std():.2f}°")
        print(f"  Min: {df_result['Angular_Deviation'].min():.2f}°")
        print(f"  Max: {df_result['Angular_Deviation'].max():.2f}°")
        
        # Print zone distribution
        for zone in ANGULAR_ZONES.keys():
            count = df_result[f'zone_{zone}'].sum()
            percentage = (count / len(df_result)) * 100
            print(f"  {zone}: {count} points ({percentage:.1f}%)")
    
    return df_result

def plot_module_cielch_single_angle(df, theta_i=30):
    """Plot CIE LCH for a single incident angle with angular zone coloring."""
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))
    ax.set_facecolor("white")
    ax.set_theta_zero_location('E')  # 0° at right (East, +a*)
    ax.set_theta_direction(1)        # Counterclockwise
    
    # Create Lab (a*, b*) color wheel background at L*=70
    theta = np.linspace(0, 2 * np.pi, 360)
    r = np.linspace(0, 100, 200)
    T, R = np.meshgrid(theta, r)
    RGB = create_lab_ab_color_wheel_background(T, R, L_fixed=70)
    ax.pcolormesh(T, R, np.ones_like(R), color=RGB.reshape(-1, 3), shading='auto')
    
    # Filter data for the specific incident angle
    angle_data = df[df['Theta_i'] == theta_i]
    
    if len(angle_data) == 0:
        print(f"No data found for θᵢ = {theta_i}°")
        return
    
    # Plot data points colored by angular zone
    for zone, config in ANGULAR_ZONES.items():
        zone_col = f'zone_{zone}'
        zone_data = angle_data[angle_data[zone_col] == True]
        
        if len(zone_data) > 0:
            hue_rad = np.radians(zone_data['Hue'])
            chroma = zone_data['Chroma']
            
            # Create scatter plot with angular zone color
            ax.scatter(hue_rad, chroma, 
                      c=config['color'], 
                      s=30, 
                      alpha=0.7,
                      edgecolors='black',  # Add back black edges for symbols
                      linewidth=0.5,
                      label=f"{config['description']}\n({len(zone_data)} points)")
    
    # Add concentric circles for chroma levels
    chroma_levels = np.arange(20, 101, 20)
    for c in chroma_levels:
        ax.plot(np.linspace(0, 2 * np.pi, 500), np.full(500, c), '--', color='gray', lw=0.8, alpha=0.7, zorder=1)
    
     # Add axis labels
    axis_label_radius = 110
    offsets = [5, 5, 20, 5]
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
    
    # Get module description for title
    module_desc = MODULE_DESCRIPTIONS.get(MODULE_ID, f"Module {MODULE_ID}")
    ax.set_title(f'{module_desc} - CIE L*C*h° Polar Plot - θᵢ = {theta_i}°\n(Angular Deviation Classification)', fontsize=16, pad=60)
    ax.grid(False)
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(OUTPUT_DIR, f'm{MODULE_ID}_cielch_theta{theta_i}_corrected.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_module_cielch_all_angles(df):
    """Plot CIE LCH for all incident angles with different symbols."""
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))
    ax.set_facecolor("white")
    ax.set_theta_zero_location('E')  # 0° at right (East, +a*)
    ax.set_theta_direction(1)        # Counterclockwise
    
    # Create Lab (a*, b*) color wheel background at L*=70
    theta = np.linspace(0, 2 * np.pi, 360)
    r = np.linspace(0, 100, 200)
    T, R = np.meshgrid(theta, r)
    RGB = create_lab_ab_color_wheel_background(T, R, L_fixed=70)
    ax.pcolormesh(T, R, np.ones_like(R), color=RGB.reshape(-1, 3), shading='auto')

    # Define markers for different incident angles
    markers = ['o', 's', '^', 'D', 'v']
    
    # Add angular zone legend entries
    for zone, config in ANGULAR_ZONES.items():
        ax.scatter([], [], c=config['color'], s=30, alpha=0.7, edgecolors='black', linewidth=0.5, label=config['description'])
    
    # Plot all data points with reduced density
    for idx, theta_i in enumerate(ANGLES):
        angle_data = df[df['Theta_i'] == theta_i]
        if len(angle_data) == 0:
            continue
        
        for zone, config in ANGULAR_ZONES.items():
            zone_col = f'zone_{zone}'
            zone_data = angle_data[angle_data[zone_col] == True]
            if len(zone_data) > 0:
                hue_rad = np.radians(zone_data['Hue'])
                chroma = zone_data['Chroma']
                ax.scatter(hue_rad, chroma, 
                           c=config['color'], 
                           s=15, 
                           alpha=0.5, 
                           marker=markers[idx],
                           edgecolors='black', 
                           linewidth=0.3,
                           label='_nolegend_')  # Prevent duplicate labels

    # Add symbol legend for incident angles
    for idx, theta_i in enumerate(ANGLES):
        ax.scatter([], [], c='gray', s=15, alpha=0.5, marker=markers[idx],
                   edgecolors='black', linewidth=0.3, label=f'θᵢ = {theta_i}°')
    
    # Add chroma rings
    chroma_levels = np.arange(20, 101, 20)
    for c in chroma_levels:
        ax.plot(np.linspace(0, 2 * np.pi, 500), np.full(500, c), '--',
                color='gray', lw=0.8, alpha=0.7, zorder=1)
    
    # Axis labels (a*, b*)
    axis_label_radius = 110
    offsets = [5, 5, 10, 5]
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
        ax.text(np.arctan2(y, x), np.hypot(x, y), label,
                fontsize=13, color='k', ha='center', va='center', fontweight='bold', rotation=0)
    
    # Get module description for title
    module_desc = MODULE_DESCRIPTIONS.get(MODULE_ID, f"Module {MODULE_ID}")
    ax.set_title(f'{module_desc} - CIE L*C*h° Polar Plot - All Incident Angles\n(Angular Deviation Classification)', fontsize=20, pad=60)
    ax.grid(False)
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(OUTPUT_DIR, f'm{MODULE_ID}_cielch_all_angles_corrected.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_weighted_zone_distribution(df):
    """
    Analyze the weighted distribution of reflected light across angular zones.
    Creates both overall analysis and per-incident-angle analysis.
    """
    module_name = MODULE_DESCRIPTIONS.get(MODULE_ID, f'Module {MODULE_ID}')
    print(f"\n=== Weighted Zone Distribution Analysis for {module_name} ===")
    
    # Overall analysis (all incident angles combined)
    print(f"\nOverall Analysis (All Incident Angles):")
    zone_ratios, weighted_data = calculate_weighted_zone_ratios(df)
    
    for zone, stats in zone_ratios.items():
        print(f"  {zone}: {stats['ratio']:.3f} ({stats['ratio']*100:.1f}%) - {stats['point_count']} points")
    
    # Per-incident-angle analysis
    print(f"\nPer-Incident-Angle Analysis:")
    angle_zone_ratios = {}
    
    for theta_i in ANGLES:
        angle_data = df[df['Theta_i'] == theta_i]
        if len(angle_data) == 0:
            continue
            
        print(f"\n  θᵢ = {theta_i}°:")
        angle_ratios, _ = calculate_weighted_zone_ratios(angle_data)
        angle_zone_ratios[theta_i] = angle_ratios
        
        for zone, stats in angle_ratios.items():
            print(f"    {zone}: {stats['ratio']:.3f} ({stats['ratio']*100:.1f}%) - {stats['point_count']} points")
    
    # Save detailed results to CSV
    results_data = []
    
    # Overall results
    for zone, stats in zone_ratios.items():
        results_data.append({
            'Module': MODULE_ID,
            'Theta_i': 'All',
            'Zone': zone,
            'Ratio': stats['ratio'],
            'Ratio_Percent': stats['ratio'] * 100,
            'Total_Weighted': stats['total_weighted'],
            'Total_BRDF': stats['total_brdf'],
            'Point_Count': stats['point_count']
        })
    
    # Per-angle results
    for theta_i, angle_ratios in angle_zone_ratios.items():
        for zone, stats in angle_ratios.items():
            results_data.append({
                'Module': MODULE_ID,
                'Theta_i': theta_i,
                'Zone': zone,
                'Ratio': stats['ratio'],
                'Ratio_Percent': stats['ratio'] * 100,
                'Total_Weighted': stats['total_weighted'],
                'Total_BRDF': stats['total_brdf'],
                'Point_Count': stats['point_count']
            })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(OUTPUT_DIR, f'm{MODULE_ID}_weighted_zone_analysis.csv'), index=False)
    print(f"\nDetailed results saved to: {OUTPUT_DIR}/m{MODULE_ID}_weighted_zone_analysis.csv")
    
    return zone_ratios, angle_zone_ratios

def plot_weighted_zone_distribution(df):
    """
    Create a bar plot showing the weighted distribution of reflected light across zones.
    """
    # Calculate overall ratios
    zone_ratios, _ = calculate_weighted_zone_ratios(df)
    
    # Prepare data for plotting
    zones = list(zone_ratios.keys())
    ratios = [zone_ratios[zone]['ratio'] * 100 for zone in zones]  # Convert to percentage
    colors = [ANGULAR_ZONES[zone]['color'] for zone in zones]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall distribution
    bars1 = ax1.bar(zones, ratios, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_title(f'Module {MODULE_ID} - Overall Weighted Zone Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Percentage of Total Reflected Light (%)', fontsize=12)
    ax1.set_xlabel('Angular Zone', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, ratio in zip(bars1, ratios):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{ratio:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Per-incident-angle distribution
    angle_data = {}
    for theta_i in ANGLES:
        angle_df = df[df['Theta_i'] == theta_i]
        if len(angle_df) > 0:
            angle_ratios, _ = calculate_weighted_zone_ratios(angle_df)
            angle_data[theta_i] = [angle_ratios[zone]['ratio'] * 100 for zone in zones]
    
    if angle_data:
        x = np.arange(len(zones))
        width = 0.15
        multiplier = 0
        
        for theta_i, ratios in angle_data.items():
            offset = width * multiplier
            bars2 = ax2.bar(x + offset, ratios, width, label=f'θᵢ = {theta_i}°', alpha=0.7)
            multiplier += 1
        
        ax2.set_title(f'Module {MODULE_ID} - Weighted Zone Distribution by Incident Angle', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Percentage of Total Reflected Light (%)', fontsize=12)
        ax2.set_xlabel('Angular Zone', fontsize=12)
        ax2.set_xticks(x + width * (len(angle_data) - 1) / 2)
        ax2.set_xticklabels(zones)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'm{MODULE_ID}_weighted_zone_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Weighted zone distribution plot saved to: {OUTPUT_DIR}/m{MODULE_ID}_weighted_zone_distribution.png")

def main():
    """Main function to run the analysis."""
    module_name = MODULE_DESCRIPTIONS.get(MODULE_ID, f'Module {MODULE_ID}')
    print(f"Starting {module_name} corrected CIE LCH analysis...")
    print(f"Module: {module_name}")
    print(f"Single angle plot: θᵢ = {SINGLE_ANGLE_THETA_I}°")
    print(f"Using angular deviation from specular for zone classification")
    
    # Load and analyze module data
    df = analyze_module_data()
    
    # Save the module dataset
    df.to_csv(os.path.join(OUTPUT_DIR, f'm{MODULE_ID}_cielch_data_corrected.csv'), index=False)
    print(f"{module_name} data saved to {OUTPUT_DIR}/m{MODULE_ID}_cielch_data_corrected.csv")
    
    # Generate plots
    print(f"\nGenerating plots for {module_name}...")
    
    # Plot for single incident angle
    plot_module_cielch_single_angle(df, theta_i=SINGLE_ANGLE_THETA_I)
    print(f"Generated single angle plot (θᵢ = {SINGLE_ANGLE_THETA_I}°)")
    
    # Plot for all incident angles
    plot_module_cielch_all_angles(df)
    print(f"Generated all angles plot")
    
    # Perform weighted zone analysis
    print(f"\nPerforming weighted zone analysis...")
    zone_ratios, angle_zone_ratios = analyze_weighted_zone_distribution(df)
    
    # Create weighted zone distribution plots
    plot_weighted_zone_distribution(df)
    
    print(f"\n{module_name} corrected CIE LCH analysis complete! Results saved in {OUTPUT_DIR}")

if __name__ == '__main__':
    main() 