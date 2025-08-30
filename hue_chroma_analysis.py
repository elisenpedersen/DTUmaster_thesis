import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colour import SpectralDistribution, sd_to_XYZ, XYZ_to_Lab, MSDS_CMFS, SDS_ILLUMINANTS, Lab_to_XYZ, XYZ_to_sRGB
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks

# CONFIGURATION
ALL_MODULES = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,25,27,28]
NON_PIGMENTED = [1,2,3,4,5,6,7,8,14,15,16,17]
BLUE_MODULES = [9,10,11,12,13]
BROWN_MODULES = [18,20,22]
GREEN_MODULES = [25,27,28]
SUBSETS = {
    'all': ALL_MODULES,
    'non_pigmented': NON_PIGMENTED,
    'blue': BLUE_MODULES,
    'brown': BROWN_MODULES,
    'green': GREEN_MODULES,
}
MODULE_DESCRIPTIONS = {
    1: "Module REF", 2: "Module B5", 3: "Module B10", 4: "Module B20",
    5: "Module G2.1S5", 6: "Module G2.1S20", 7: "Module G1.5S5", 8: "Module G1.5S20",
    9: "Module Blue2.1S", 10: "Module Blue2.1L", 11: "Module Blue1.5S", 12: "Module Blue1.5L", 13: "Module BlueBa",
    14: "Module G2.1L5", 15: "Module G2.1L20", 16: "Module G1.5L5", 17: "Module G1.5L20",
    18: "Module BrownC", 20: "Module BrownBaC", 22: "Module BrownGlC",
    25: "Module GreenC", 27: "Module GreenBaC", 28: "Module GreenGlC",
}
MODULE_COLORS = {
    1: 'black', 2: 'black', 3: 'black', 4: 'black', 5: 'black', 6: 'black', 7: 'black', 8: 'black',
    14: 'black', 15: 'black', 16: 'black', 17: 'black',
    9: '#1f77b4', 10: '#1f77b4', 11: '#1f77b4', 12: '#1f77b4', 13: '#1f77b4',
    18: '#8c564b', 20: '#8c564b', 22: '#8c564b',
    25: '#2ca02c', 27: '#2ca02c', 28: '#2ca02c',
}
ANGLES = [0, 15, 30, 45, 60] 
#change directory
OUTPUT_DIR = "cielab/nonspecular/lightness_correction"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def analyze_all():
    all_lab = []
    for module in ALL_MODULES:
        for angle in ANGLES:
            pkl_path = os.path.join('../../../brdf_plots', f'Module{module}', f'Module{module}_theta{angle}_phi0.pkl')
            if not os.path.exists(pkl_path):
                print(f"Missing: {pkl_path}")
                continue
            df = load_pkl(pkl_path)
            wavelengths = np.linspace(400, 700, len(df['spec_brdf'].iloc[0]))
            
            # Identify specular peaks using BRDF values
            if 'total_brdf' in df.columns:
                brdf_values = df['total_brdf'].values
                # Find local maxima (peaks)
                peaks, _ = find_peaks(brdf_values, height=np.mean(brdf_values))
                # Calculate the mean and std of BRDF values excluding peaks
                non_peak_mask = ~np.isin(np.arange(len(brdf_values)), peaks)
                mean_brdf = np.mean(brdf_values[non_peak_mask])
                std_brdf = np.std(brdf_values[non_peak_mask])
                # Define specular threshold as mean + 2*std
                specular_threshold = mean_brdf + 4 * std_brdf
                # Create mask for non-specular points
                non_specular_mask = brdf_values < specular_threshold
                print(f"Module {module}, θᵢ={angle}°: {np.sum(non_specular_mask)}/{len(brdf_values)} non-specular points")
            else:
                # If no total_brdf column, process all points
                non_specular_mask = np.ones(len(df), dtype=bool)
                print(f"Module {module}, θᵢ={angle}°: No BRDF data, processing all {len(df)} points")
            
            for idx, row in df.iterrows(): #CHANGE THIS TO SWAP BETWEEN SPECULAR AND NON-SPECULAR
                # Comment out the line below to include specular points:
                if not non_specular_mask[idx]:
                    continue
                    
                spectrum = np.array(row['spec_brdf'])
                Lab = spectrum_to_lab(wavelengths, spectrum)
                
                # Skip low-lightness measurements (likely specular artifacts):
                if Lab[0] < 15:  # L* < 15 (changed from 20)
                    continue
                
                all_lab.append({
                    'Module': module,
                    'Theta_i': angle,
                    'L': Lab[0],
                    'a': Lab[1],
                    'b': Lab[2]
                })
                
    df = pd.DataFrame(all_lab)
    # Calculate hue and chroma
    df['Hue'] = (np.degrees(np.arctan2(df['b'], df['a'])) + 360) % 360
    df['Chroma'] = np.sqrt(df['a']**2 + df['b']**2)

    # --- DIAGNOSTIC PRINT ---
    # Print the Lab and Hue values for a specific green module to check its color data
   # green_module_check = df[(df['Module'] == 24) & (df['Theta_i'] == 0)]
   # if not green_module_check.empty:
   #     print("\n--- Diagnostic Check for Green Module 24 at 0° ---")
   #     print(green_module_check[['L', 'a', 'b', 'Hue']].to_string())
   #     print("----------------------------------------------------\n")
        
    return df

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

# --- Plotting stubs ---
def plot_hue_vs_angle(df, modules, subset_name):
    plt.figure(figsize=(10, 6))
    for module in modules:
        sub = df[df['Module'] == module]
        sub_mean = group_mean_ab_hue(sub, 'Theta_i').sort_index()
        if not sub_mean.empty:
            plt.plot(sub_mean.index, sub_mean['Hue'], 'o-', label=MODULE_DESCRIPTIONS[module], color=MODULE_COLORS[module], linewidth=2)
    plt.xlabel('Incident Angle (deg)')
    plt.ylabel('Hue Angle (degrees)')
    plt.title(f'Hue Angle vs. Incident Angle ({subset_name})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'hue_vs_angle_{subset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_chroma_vs_angle(df, modules, subset_name):
    plt.figure(figsize=(10, 6))
    for module in modules:
        sub = df[df['Module'] == module]
        sub_mean = group_mean_ab_hue(sub, 'Theta_i').sort_index()
        if not sub_mean.empty:
            plt.plot(sub_mean.index, sub_mean['Chroma'], 'o-', label=MODULE_DESCRIPTIONS[module], color=MODULE_COLORS[module], linewidth=2)
    plt.xlabel('Incident Angle (deg)')
    plt.ylabel('Chroma (C*)')
    plt.title(f'Chroma vs. Incident Angle ({subset_name})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'chroma_vs_angle_{subset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_L_vs_angle(df, modules, subset_name):
    plt.figure(figsize=(10, 6))
    for module in modules:
        sub = df[df['Module'] == module]
        sub_mean = group_mean_ab_hue(sub, 'Theta_i').sort_index()
        if not sub_mean.empty:
            plt.plot(sub_mean.index, sub_mean['L'], 'o-', label=MODULE_DESCRIPTIONS[module], color=MODULE_COLORS[module], linewidth=2)
    plt.xlabel('Incident Angle (deg)')
    plt.ylabel('L* (Lightness)')
    plt.title(f'L* vs. Incident Angle ({subset_name})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'L_vs_angle_{subset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_L_vs_hue(df, modules, subset_name):
    plt.figure(figsize=(10, 6))
    for module in modules:
        sub = df[df['Module'] == module]
        sub_mean = group_mean_ab_hue(sub, 'Theta_i').sort_values('Hue')
        if not sub_mean.empty:
            plt.plot(sub_mean['Hue'], sub_mean['L'], 'o-', label=MODULE_DESCRIPTIONS[module], color=MODULE_COLORS[module], linewidth=2)
    plt.xlabel('Hue Angle (degrees)')
    plt.ylabel('L* (Lightness)')
    plt.title(f'L* vs. Hue Angle ({subset_name})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'L_vs_hue_{subset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_ab_trajectory(df, modules, subset_name):
    plt.figure(figsize=(8, 8))
    for module in modules:
        sub = df[df['Module'] == module]
        sub_mean = group_mean_ab_hue(sub, 'Theta_i').sort_index()
        if not sub_mean.empty:
            plt.plot(sub_mean['a'], sub_mean['b'], 'o-', label=MODULE_DESCRIPTIONS[module], color=MODULE_COLORS[module], linewidth=2)
            for idx, row in sub_mean.iterrows():
                plt.annotate(f"{int(idx)}°", (row['a'], row['b']), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    plt.xlabel('a* (green-red)')
    plt.ylabel('b* (blue-yellow)')
    plt.title(f'a*b* Trajectory ({subset_name})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'ab_trajectory_{subset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_3d_hue_chroma_L(df, modules, subset_name):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for module in modules:
        sub = df[df['Module'] == module]
        sub_mean = group_mean_ab_hue(sub, 'Theta_i').sort_index()
        if not sub_mean.empty:
            hue = sub_mean['Hue']
            chroma = sub_mean['Chroma']
            L = sub_mean['L']
            ax.plot(hue, chroma, L, 'o-', label=MODULE_DESCRIPTIONS[module], color=MODULE_COLORS[module], linewidth=2)
    ax.set_xlabel('Hue Angle (degrees)')
    ax.set_ylabel('Chroma (C*)')
    ax.set_zlabel('L* (Lightness)')
    ax.set_title(f'3D Hue-Chroma-L* Trajectory ({subset_name})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'3d_hue_chroma_L_{subset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

"""    
def main():
    df = analyze_all()
    df.to_csv(os.path.join(OUTPUT_DIR, 'hue_chroma_lab_results_nonspecular.csv'), index=False)
    print(f'Hue/Chroma analysis complete (non-specular). Results saved in {OUTPUT_DIR}.')
"""
def main():
    df = analyze_all()
    df.to_csv(os.path.join(OUTPUT_DIR, 'hue_chroma_lab_results_nonspecular_lightness_correction.csv'), index=False)
    for subset_name, modules in SUBSETS.items():
        plot_hue_vs_angle(df, modules, subset_name)
        plot_chroma_vs_angle(df, modules, subset_name)
        plot_L_vs_angle(df, modules, subset_name)
        plot_L_vs_hue(df, modules, subset_name)
        plot_ab_trajectory(df, modules, subset_name)
        plot_3d_hue_chroma_L(df, modules, subset_name)
    print(f'Hue/Chroma analysis complete. Results saved in {OUTPUT_DIR}.')

if __name__ == '__main__':
    main() 