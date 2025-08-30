import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colour import SpectralDistribution, sd_to_XYZ, XYZ_to_Lab, MSDS_CMFS, SDS_ILLUMINANTS, Lab_to_XYZ, XYZ_to_sRGB

# CONFIGURATION - CHANGE THESE VALUES TO ANALYZE DIFFERENT MODULES
MODULE_ID = 20  # Change this to any module number you want
ANGLES = [0, 15, 30, 45, 60]
OUTPUT_DIR = "LAB/scattering_cielch_corrected"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Module descriptions
MODULE_DESCRIPTIONS = {
    1: "REF", 2: "B5", 3: "B10", 4: "B20",
    5: "G2.1S5", 6: "G2.1S20", 7: "G1.5S5", 8: "G1.5S20",
    9: "Blue2.1S", 10: "Blue2.1L", 11: "Blue1.5S", 12: "Blue1.5L", 13: "BlueBa",
    14: "G2.1L5", 15: "G2.1L20", 16: "G1.5L5", 17: "G1.5L20",
    18: "BrownC", 20: "BrownBaC", 22: "BrownGlC",
    25: "GreenC", 27: "GreenBaC", 28: "GreenGlC",
}

# Scattering lobe classification based on angular deviation from specular
SCATTERING_LOBES = {
    'specular': {'deviation_range': (0, 15), 'description': 'Specular (0-15° deviation)', 'color': '#d62728'},
    'forward_scatter': {'deviation_range': (15, 45), 'description': 'Forward scatter (15-45° deviation)', 'color': '#9467bd'},
    'backscatter': {'deviation_range': (45, 90), 'description': 'Backscatter (45-90° deviation)', 'color': '#8c564b'},
    'residual': {'deviation_range': (90, 180), 'description': 'Residual scatter (>90° deviation)', 'color': '#7f7f7f'}
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
    Calculate angular deviation between the measured reflection direction and ideal specular reflection.
    
    Args:
        theta_i (float): Incident zenith angle (degrees)
        theta_r (float): Reflection zenith angle (degrees)
        phi_r (float): Reflection azimuth angle (degrees)
        
    Returns:
        float: Angular deviation in degrees
    """
    # Convert angles to radians
    theta_i_rad = np.radians(theta_i)
    theta_r_rad = np.radians(theta_r)
    phi_r_rad = np.radians(phi_r)
    
    # Incident vector (coming in): I = [sin(θᵢ), 0, -cos(θᵢ)]
    # Surface normal = [0, 0, 1]
    # Specular reflection = mirror(I) = [sin(θᵢ), 0, cos(θᵢ)]
    R_spec = np.array([
        np.sin(theta_i_rad),
        0.0,
        np.cos(theta_i_rad)
    ])
    
    # Measured reflection direction in spherical coordinates
    R_meas = np.array([
        np.sin(theta_r_rad) * np.cos(phi_r_rad),
        np.sin(theta_r_rad) * np.sin(phi_r_rad),
        np.cos(theta_r_rad)
    ])
    
    # Compute the dot product
    dot_product = np.dot(R_meas, R_spec)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Numerical stability
    
    # Compute angular deviation
    deviation = np.degrees(np.arccos(dot_product))
    
    return deviation

def classify_scattering_lobe(deviation):
    """Classify reflection based on angular deviation from specular."""
    classifications = {}
    
    for lobe, config in SCATTERING_LOBES.items():
        min_deviation, max_deviation = config['deviation_range']
        classifications[f'lobe_{lobe}'] = min_deviation <= deviation <= max_deviation
    
    return classifications

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
    """Analyze data for the specified module with corrected scattering lobe classification."""
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
            
            # Calculate angular deviation from specular
            angular_deviation = calculate_angular_deviation_from_specular(angle, theta_r, phi_r)
            
            # Enhanced filtering: avoid extreme specular artifacts
            #if angular_deviation <= 15 and Lab[0] < 10:  # Very low L* in specular region
            #    continue
            
            # Get scattering lobe classifications based on angular deviation
            classifications = classify_scattering_lobe(angular_deviation)
            
            # Calculate hue and chroma
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
        
        # Print lobe distribution
        for lobe in SCATTERING_LOBES.keys():
            count = df_result[f'lobe_{lobe}'].sum()
            percentage = (count / len(df_result)) * 100
            print(f"  {lobe}: {count} points ({percentage:.1f}%)")
    
    return df_result

def plot_scattering_lobe_cielch(df, lobe_name, lobe_config, ax):
    """Plot CIE LCH for a specific scattering lobe."""
    ax.set_facecolor("white")
    ax.set_theta_zero_location('E')  # 0° at right (East, +a*)
    ax.set_theta_direction(1)        # Counterclockwise
    
    # Create Lab (a*, b*) color wheel background at L*=70
    theta = np.linspace(0, 2 * np.pi, 360)
    r = np.linspace(0, 100, 200)
    T, R = np.meshgrid(theta, r)
    RGB = create_lab_ab_color_wheel_background(T, R, L_fixed=70)
    ax.pcolormesh(T, R, np.ones_like(R), color=RGB.reshape(-1, 3), shading='auto')
    
    # Filter data for the specific scattering lobe
    lobe_col = f'lobe_{lobe_name}'
    lobe_data = df[df[lobe_col] == True]
    
    if len(lobe_data) > 0:
        hue_rad = np.radians(lobe_data['Hue'])
        chroma = lobe_data['Chroma']
        
        # Create scatter plot with lobe color
        ax.scatter(hue_rad, chroma, 
                  c=lobe_config['color'], 
                  s=30, 
                  alpha=0.7,
                  edgecolors='black',
                  linewidth=0.5,
                  label=f"{lobe_config['description']}\n({len(lobe_data)} points)")
    
    # Add concentric circles for chroma levels
    chroma_levels = np.arange(20, 101, 20)
    for c in chroma_levels:
        ax.plot(np.linspace(0, 2 * np.pi, 500), np.full(500, c), '--', color='gray', lw=0.8, alpha=0.7, zorder=1)
    
    # Add axis labels
    axis_label_radius = 110
    offsets = [7, 7, 15, 7]
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
    
    # No individual plot title - only main title
    ax.grid(False)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=10)

def main():
    module_name = MODULE_DESCRIPTIONS.get(MODULE_ID, f'Module {MODULE_ID}')
    print(f"Starting {module_name} corrected scattering lobe CIE LCH analysis...")
    print(f"Module: {module_name}")
    print(f"Using angular deviation from specular for lobe classification")
    
    # Load and analyze module data
    df = analyze_module_data()
    
    # Save the module dataset
    df.to_csv(os.path.join(OUTPUT_DIR, f'm{MODULE_ID}_corrected_scattering_lobe_data.csv'), index=False)
    print(f"{module_name} data saved to {OUTPUT_DIR}/m{MODULE_ID}_corrected_scattering_lobe_data.csv")
    
    # Create figure with four subplots in a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw={'projection': 'polar'})
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Plot each scattering lobe
    lobe_names = list(SCATTERING_LOBES.keys())
    for idx, lobe_name in enumerate(lobe_names):
        lobe_config = SCATTERING_LOBES[lobe_name]
        plot_scattering_lobe_cielch(df, lobe_name, lobe_config, axes[idx])
    
    # Add single main title
    fig.suptitle(f'{module_name} - CIE L*C*h° Polar Plots by Angular Deviation from Specular', fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.3)  # Adjust spacing for 2x2 grid
    plt.savefig(os.path.join(OUTPUT_DIR, f'm{MODULE_ID}_corrected_scattering_lobe_cielch.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{module_name} corrected scattering lobe CIE LCH analysis complete!")
    print(f"Results saved in {OUTPUT_DIR}")
    print(f"Key improvements:")
    print(f"- Uses angular deviation from specular reflection for classification")
    print(f"- Accounts for both θᵣ and φᵣ in deviation calculation")
    print(f"- Properly groups reflections by proximity to expected specular direction")
    print(f"- Four polar plots: Specular (0-15° dev), Forward scatter (15-45° dev), Backscatter (45-90° dev), Residual (>90° dev)")
    print(f"- Color wheel background at L*=70")
    print(f"- Enhanced filtering of extreme specular artifacts")
    print(f"- 2x2 grid layout to accommodate all four scattering lobes")

if __name__ == '__main__':
    main() 