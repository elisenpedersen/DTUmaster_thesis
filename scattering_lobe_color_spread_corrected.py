import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colour import SpectralDistribution, sd_to_XYZ, XYZ_to_Lab, MSDS_CMFS, SDS_ILLUMINANTS

# CONFIGURATION
ANGLES = [0, 15, 30, 45, 60]
OUTPUT_DIR = "LAB/scattering_cielch_corrected"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Module descriptions
MODULE_DESCRIPTIONS = {
    1: "Module 1 (REF)", 2: "Module 2 (B5)", 3: "Module 3 (B10)", 4: "Module 4 (B20)",
    5: "Module 5 (G2.1S5)", 6: "Module 6 (G2.1S20)", 7: "Module 7 (G1.5S5)", 8: "Module 8 (G1.5S20)",
    9: "Module 9 (Blue2.1S)", 10: "Module 10 (Blue2.1L)", 11: "Module 11 (Blue1.5S)", 12: "Module 12 (Blue1.5L)", 13: "Module 13 (BlueBa)",
    14: "Module 14 (G2.1L5)", 15: "Module 15 (G2.1L20)", 16: "Module 16 (G1.5L5)", 17: "Module 17 (G1.5L20)",
    18: "Module 18 (BrownC)", 20: "Module 20 (BrownBaC)", 22: "Module 22 (BrownGlC)",
    24: "Module 24 (GreenC)", 25: "Module 25 (GreenC)", 26: "Module 26 (GreenBaC)", 29: "Module 29 (GreenGlC)",
}

# Module colors for plotting
MODULE_COLORS = {
    1: 'black', 9: 'blue', 18: 'brown', 25: 'green'
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
    
    for lobe, config in SCATTERING_LOBES.items():
        min_deviation, max_deviation = config['deviation_range']
        classifications[f'lobe_{lobe}'] = min_deviation <= angular_deviation <= max_deviation
    
    return classifications

def analyze_modules_data(modules):
    """Analyze data for specified modules with corrected angular deviation classification."""
    all_lab = []
    
    for module_id in modules:
        for angle in ANGLES:
            pkl_path = os.path.join('brdf_plots', f'Module{module_id}', f'Module{module_id}_theta{angle}_phi0.pkl')
            if not os.path.exists(pkl_path):
                print(f"Missing: {pkl_path}")
                continue
                    
            df = load_pkl(pkl_path)
            wavelengths = np.linspace(400, 700, len(df['spec_brdf'].iloc[0]))
            
            print(f"Processing Module {module_id}, θᵢ={angle}°: {len(df)} measurements")
            
            for idx, row in df.iterrows():
                theta_r = row['theta_r']
                phi_r = row.get('phi_r', 0)  # Default to 0 if not available
                spectrum = np.array(row['spec_brdf'])
                Lab = spectrum_to_lab(wavelengths, spectrum)
                
                # Calculate angular deviation from specular
                angular_deviation = calculate_angular_deviation_from_specular(angle, theta_r, phi_r)
                
                # Enhanced filtering: avoid extreme specular artifacts
                if angular_deviation <= 15 and Lab[0] < 10:  # Very low L* in specular region
                    continue
                
                # Get scattering lobe classifications based on angular deviation
                classifications = classify_reflection_by_deviation(angular_deviation)
                
                # Calculate hue and chroma
                hue = (np.degrees(np.arctan2(Lab[2], Lab[1])) + 360) % 360
                chroma = np.sqrt(Lab[1]**2 + Lab[2]**2)
                
                # Store data with classifications
                data_point = {
                    'Module': module_id,
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
    print(f"Total data points: {len(df_result)}")
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

def plot_scattering_lobe_color_spread(df, modules, title, output_name):
    """Plot color spread in a*b* space for different scattering lobes."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    lobe_names = list(SCATTERING_LOBES.keys())
    
    for idx, lobe in enumerate(lobe_names):
        lobe_col = f'lobe_{lobe}'
        lobe_data = df[(df[lobe_col] == True) & (df['Module'].isin(modules))]
        
        ax = axes[idx]
        
        # Plot each module with different colors
        for module in modules:
            module_data = lobe_data[lobe_data['Module'] == module]
            if len(module_data) > 0:
                ax.scatter(module_data['a'], module_data['b'], 
                          alpha=0.6, s=30,
                          label=f"{MODULE_DESCRIPTIONS[module]} ({len(module_data)} points)",
                          color=MODULE_COLORS[module])
        
        ax.set_xlabel('a* (green-red)')
        ax.set_ylabel('b* (blue-yellow)')
        ax.set_title(f'{SCATTERING_LOBES[lobe]["description"]}\nColor Spread in a*b* Space')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_aspect('equal')
        
        # Add legend to each subplot
        ax.legend(fontsize=8)
    
    plt.suptitle(f"{title}\n(Angular Deviation Classification)", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.3)  # Adjust spacing for 2x2 grid
    plt.savefig(os.path.join(OUTPUT_DIR, f'{output_name}_corrected.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Starting corrected scattering lobe color spread analysis...")
    print("Using angular deviation from specular for lobe classification")
    
    # Representative modules (same as the angular analysis)
    representative_modules = [1, 9, 18, 25]  # REF, Blue, Brown, Green
    
    print(f"Analyzing representative modules: {representative_modules}")
    
    # Load and analyze data with corrected angular deviation classification
    df = analyze_modules_data(representative_modules)
    
    # Save the dataset
    df.to_csv(os.path.join(OUTPUT_DIR, 'scattering_lobe_color_data_corrected.csv'), index=False)
    print(f"Data saved to {OUTPUT_DIR}/scattering_lobe_color_data_corrected.csv")
    
    # Generate corrected scattering lobe color spread plot
    plot_scattering_lobe_color_spread(
        df, representative_modules,
        "Color Spread by Scattering Lobe (Representative Modules)",
        "scattering_lobe_color_spread_representative"
    )
    
    print(f"\nCorrected scattering lobe color spread analysis complete!")
    print(f"Results saved in {OUTPUT_DIR}")
    print(f"Key improvements:")
    print(f"- Uses angular deviation from specular reflection for lobe classification")
    print(f"- Accounts for both θᵣ and φᵣ in deviation calculation")
    print(f"- Properly groups reflections by proximity to expected specular direction")
    print(f"- Same representative modules: REF (black), Blue (blue), Brown (brown), Green (green)")
    print(f"- Color spread in a*b* space for each scattering lobe")
    print(f"- Enhanced filtering of extreme specular artifacts")
    print(f"- More accurate and physically meaningful color stability analysis")

if __name__ == '__main__':
    main() 