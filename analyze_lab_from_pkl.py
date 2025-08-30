import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colour import SpectralDistribution, sd_to_XYZ, XYZ_to_Lab, MSDS_CMFS, SDS_ILLUMINANTS, delta_E
import matplotlib.cm as cm

# CONFIGURATION
# Module definitions (23 modules)
NON_PIGMENTED = [1,2,3,4,5,6,7,8,14,15,16,17]
BLUE_MODULES = [9,10,11,12,13]
BROWN_MODULES = [18,20,22]
GREEN_MODULES = [25,27,28]

MODULES = NON_PIGMENTED + BLUE_MODULES + BROWN_MODULES + GREEN_MODULES
ANGLES = [0, 15, 30, 45, 60]
MODULE_DESCRIPTIONS = {
    1: "REF", 2: "B5", 3: "B10", 4: "B20",
    5: "G2.1S5", 6: "G2.1S20", 7: "G1.5S5", 8: "G1.5S20",
    9: "Blue2.1S", 10: "Blue2.1L", 11: "Blue1.5S", 12: "Blue1.5L", 13: "BlueBa",
    14: "G2.1L5", 15: "G2.1L20", 16: "G1.5L5", 17: "G1.5L20",
    18: "BrownC", 20: "BrownBaC", 22: "BrownGlC",
    25: "GreenC", 27: "GreenBaC", 28: "GreenGlC",
}
OUTPUT_DIR = "LAB/lab_analysis_from_pkl"
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
    for module in MODULES:
        for angle in ANGLES:
            pkl_path = os.path.join('brdf_plots', f'Module{module}', f'Module{module}_theta{angle}_phi0.pkl')
            if not os.path.exists(pkl_path):
                print(f"Missing: {pkl_path}")
                continue
            df = load_pkl(pkl_path)
            # ASSUMPTION: All rows have the same wavelength grid, e.g. 400-700nm, 1nm step
            # If you have a wavelength column, use it here instead
            wavelengths = np.linspace(400, 700, len(df['spec_brdf'].iloc[0]))
            for idx, row in df.iterrows():
                spectrum = np.array(row['spec_brdf'])
                Lab = spectrum_to_lab(wavelengths, spectrum)
                all_lab.append({
                    'Module': module,
                    'Theta_i': angle,
                    'L': Lab[0],
                    'a': Lab[1],
                    'b': Lab[2]
                })
    return pd.DataFrame(all_lab)

def plot_lab_scatter(df, theta_of_interest=None):
    plt.figure(figsize=(8, 8))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(len(MODULES))]
    
    for idx, module in enumerate(MODULES):
        # Get data for this module
        sub = df[df['Module'] == module]
        if theta_of_interest is not None:
            sub = sub[sub['Theta_i'] == theta_of_interest]
        
        # Group by theta_i and calculate mean for each angle
        sub = sub.groupby('Theta_i').agg({
            'a': 'mean',
            'b': 'mean'
        }).reset_index()
        
        # Sort by angle for better line connection
        sub = sub.sort_values('Theta_i')
        
        # Plot scatter points
        plt.scatter(sub['a'], sub['b'], label=MODULE_DESCRIPTIONS[module], 
                   alpha=0.7, color=colors[idx], s=100)  # Increased size and opacity
        
        # Add line connecting points
        plt.plot(sub['a'], sub['b'], '-', color=colors[idx], alpha=0.3, linewidth=1)
        
      
    plt.xlabel('a* (green-red)')
    plt.ylabel('b* (blue-yellow)')
    title = 'LAB Color Distribution'
    if theta_of_interest is not None:
        title += f' (θᵢ={theta_of_interest}°)'
    else:
        title += ' (all angles)'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    fname = f'lab_scatter_theta_{theta_of_interest}.png' if theta_of_interest is not None else 'lab_scatter_all_angles.png'
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
    plt.close()

def plot_lab_scatter_with_angles(df, theta_of_interest=None):
    """
    Create a scatter plot showing averaged points (one per theta_i) for each module.
    Each module has a consistent color, and each point is labeled with its theta value.
    """
    plt.figure(figsize=(10, 10))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(len(MODULES))]
    
    for idx, module in enumerate(MODULES):
        # Get data for this module
        sub = df[df['Module'] == module]
        
        # Group by theta_i and calculate mean for each angle
        sub = sub.groupby('Theta_i').agg({
            'a': 'mean',
            'b': 'mean'
        }).reset_index()
        
        # Sort by angle for better visualization
        sub = sub.sort_values('Theta_i')
        
        # Plot averaged points for this module
        plt.scatter(sub['a'], sub['b'], label=MODULE_DESCRIPTIONS[module], 
                   alpha=0.7, color=colors[idx], s=100)
        
        # Add angle labels next to points
        for _, row in sub.iterrows():
            plt.annotate(f"{row['Theta_i']}°", 
                        (row['a'], row['b']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    plt.xlabel('a* (green-red)')
    plt.ylabel('b* (blue-yellow)')
    title = 'LAB Color Distribution - Averaged Points'
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
    plt.grid(True, alpha=0.3)
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    fname = 'lab_scatter_averaged_points.png'
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches='tight')
    plt.close()

def plot_lab_scatter_raw(df, theta_of_interest=None):
    """
    Create a scatter plot showing all raw LAB measurements without averaging.
    Each point represents an individual measurement.
    """
    plt.figure(figsize=(8, 8))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(len(MODULES))]
    
    for idx, module in enumerate(MODULES):
        sub = df[df['Module'] == module]
        if theta_of_interest is not None:
            sub = sub[sub['Theta_i'] == theta_of_interest]
        plt.scatter(sub['a'], sub['b'], label=MODULE_DESCRIPTIONS[module], 
                   alpha=0.5, color=colors[idx], s=30)  # Smaller points for raw data
    
    plt.xlabel('a* (green-red)')
    plt.ylabel('b* (blue-yellow)')
    title = 'Raw LAB Color Distribution'
    if theta_of_interest is not None:
        title += f' (θᵢ={theta_of_interest}°)'
    else:
        title += ' (all angles)'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    fname = f'lab_scatter_raw_theta_{theta_of_interest}.png' if theta_of_interest is not None else 'lab_scatter_raw_all_angles.png'
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
    plt.close()

def plot_lab_trajectories(df):
    plt.figure(figsize=(8, 8))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(len(MODULES))]
    for idx, module in enumerate(MODULES):
        sub = df[df['Module'] == module].groupby('Theta_i').mean().sort_index()
        plt.plot(sub['a'], sub['b'], 'o-', label=MODULE_DESCRIPTIONS[module], color=colors[idx])
    plt.xlabel('a* (green-red)')
    plt.ylabel('b* (blue-yellow)')
    plt.title('LAB Trajectories Across Angles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'lab_trajectories.png'), dpi=300)
    plt.close()

def plot_l_vs_angle(df):
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(len(MODULES))]
    for idx, module in enumerate(MODULES):
        sub = df[df['Module'] == module].groupby('Theta_i').mean().sort_index()
        plt.plot(sub.index, sub['L'], 'o-', label=MODULE_DESCRIPTIONS[module], color=colors[idx])
    plt.xlabel('Incident Angle (deg)')
    plt.ylabel('L* (Lightness)')
    plt.title('L* Evolution with Incident Angle')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'L_vs_angle.png'), dpi=300)
    plt.close()

def calculate_centroid_shift_lab(df):
    """
    Calculate centroid (mean Lab) and DeltaE shifts for each module and angle.
    Returns a DataFrame with Module, Theta_i, centroid_L, centroid_a, centroid_b, DeltaE_to_prev, DeltaE_to_ref.
    """
    results = []
    for module in df['Module'].unique():
        sub = df[df['Module'] == module]
        grouped = sub.groupby('Theta_i')
        centroids = {}
        for theta_i, group in grouped:
            labs = group[['L', 'a', 'b']].values
            centroid = labs.mean(axis=0)
            centroids[theta_i] = centroid
        sorted_thetas = sorted(centroids.keys())
        ref_centroid = centroids[sorted_thetas[0]]
        prev_centroid = None
        for theta_i in sorted_thetas:
            centroid = centroids[theta_i]
            DeltaE_to_ref = np.linalg.norm(centroid - ref_centroid)
            DeltaE_to_prev = np.linalg.norm(centroid - prev_centroid) if prev_centroid is not None else 0
            results.append({
                'Module': module,
                'Theta_i': theta_i,
                'centroid_L': centroid[0],
                'centroid_a': centroid[1],
                'centroid_b': centroid[2],
                'DeltaE_to_prev': DeltaE_to_prev,
                'DeltaE_to_ref': DeltaE_to_ref
            })
            prev_centroid = centroid
    return pd.DataFrame(results)

def summarize_color_stability(centroid_df):
    """
    Summarize color stability for each module: min, max, mean DeltaE_to_ref.
    Save as CSV and print the summary.
    """
    summary = centroid_df.groupby('Module')['DeltaE_to_ref'].agg(['min', 'max', 'mean']).reset_index()
    summary = summary.rename(columns={'min': 'DeltaE_min', 'max': 'DeltaE_max', 'mean': 'DeltaE_mean'})
    summary_path = os.path.join(OUTPUT_DIR, 'lab_centroid_stability_summary.csv')
    summary.to_csv(summary_path, index=False)
    print('Color stability summary:')
    print(summary)
    print(f'Saved summary to {summary_path}')

def plot_color_stability_bar(summary_df):
    """
    Plot a bar chart of DeltaE_max for each module.
    """
    plt.figure(figsize=(10,6))
    modules = summary_df['Module'].astype(str)
    plt.bar(modules, summary_df['DeltaE_max'], color='gray', edgecolor='black')
    plt.xlabel('Module')
    plt.ylabel('Max ΔE to Reference')
    plt.title('Maximum Centroid Color Shift (ΔE) per Module')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'lab_centroid_stability_barplot.png'), dpi=300)
    plt.close()
    print(f'Saved bar plot to {os.path.join(OUTPUT_DIR, "lab_centroid_stability_barplot.png")}')

def calculate_normalized_deltaE(df, reference_module=1):
    """
    Calculate ΔE while removing glass glare effects on lightness by comparing to the reference module.
    This isolates chromaticity differences by ignoring L* variations caused by glass glare.
    """
    results = []
    
    # Get reference module data (Module 1)
    ref_data = df[df['Module'] == reference_module].groupby('Theta_i').agg({
        'L': 'mean',
        'a': 'mean', 
        'b': 'mean'
    }).reset_index()
    
    for module in df['Module'].unique():
        if module == reference_module:
            continue
            
        sub = df[df['Module'] == module]
        grouped = sub.groupby('Theta_i')
        
        for theta_i, group in grouped:
            # Get current module's mean values
            current_L = group['L'].mean()
            current_a = group['a'].mean()
            current_b = group['b'].mean()
            
            # Get reference module's values for this angle
            ref_row = ref_data[ref_data['Theta_i'] == theta_i]
            if ref_row.empty:
                continue
                
            ref_L = ref_row['L'].iloc[0]
            ref_a = ref_row['a'].iloc[0]
            ref_b = ref_row['b'].iloc[0]
            
            # Method 1: Calculate ΔE in a*b* plane only (chromaticity) - removes glass glare effect
            deltaE_chromaticity = np.sqrt(
                (current_a - ref_a)**2 +  # a* difference
                (current_b - ref_b)**2    # b* difference
            )
            
            # Method 2: Full ΔE (for comparison) - includes all differences
            deltaE_full = np.sqrt(
                (current_L - ref_L)**2 +  # L* difference
                (current_a - ref_a)**2 +  # a* difference
                (current_b - ref_b)**2    # b* difference
            )
            
            results.append({
                'Module': module,
                'Theta_i': theta_i,
                'current_L': current_L,
                'current_a': current_a,
                'current_b': current_b,
                'ref_L': ref_L,
                'ref_a': ref_a,
                'ref_b': ref_b,
                'deltaE_chromaticity': deltaE_chromaticity,
                'deltaE_full': deltaE_full
            })
    
    return pd.DataFrame(results)

def plot_normalized_deltaE_comparison(normalized_df):
    """
    Plot the different ΔE calculations for comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Chromaticity ΔE (a*b* only) - removes glass glare effect
    for module in normalized_df['Module'].unique():
        sub = normalized_df[normalized_df['Module'] == module].sort_values('Theta_i')
        ax1.plot(sub['Theta_i'], sub['deltaE_chromaticity'], 'o-', 
                label=f'Module {module}', alpha=0.7)
    ax1.set_xlabel('Incident Angle (deg)')
    ax1.set_ylabel('ΔE (Chromaticity Only)')
    ax1.set_title('ΔE in a*b* Plane\n(Removes Glass Glare Effect)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Full ΔE (for comparison) - includes all differences
    for module in normalized_df['Module'].unique():
        sub = normalized_df[normalized_df['Module'] == module].sort_values('Theta_i')
        ax2.plot(sub['Theta_i'], sub['deltaE_full'], 'o-', 
                label=f'Module {module}', alpha=0.7)
    ax2.set_xlabel('Incident Angle (deg)')
    ax2.set_ylabel('ΔE (Full)')
    ax2.set_title('Full ΔE (L* + a* + b*)\n(Includes Glass Glare Effect)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'normalized_deltaE_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def summarize_normalized_stability(normalized_df):
    """
    Summarize the normalized color stability metrics.
    """
    summary = normalized_df.groupby('Module').agg({
        'deltaE_chromaticity': ['min', 'max', 'mean'],
        'deltaE_full': ['min', 'max', 'mean']
    }).round(3)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    
    summary_path = os.path.join(OUTPUT_DIR, 'normalized_deltaE_summary.csv')
    summary.to_csv(summary_path, index=False)
    print('Normalized color stability summary:')
    print(summary)
    print(f'Saved summary to {summary_path}')
    return summary

def calculate_glass_glare_corrected_deltaE(df, reference_module=1):
    """
    Calculate ΔE with glass glare correction by subtracting Module 1's L* variations with angle.
    This corrects for systematic glass glare effects that affect lightness across all modules.
    """
    results = []
    
    # Get reference module data (Module 1) to calculate L* correction factors
    ref_data = df[df['Module'] == reference_module].groupby('Theta_i').agg({
        'L': 'mean',
        'a': 'mean', 
        'b': 'mean'
    }).reset_index()
    
    # Calculate L* correction factors (how much L* changes with angle for Module 1)
    # Use 0° as the baseline for L* correction
    baseline_L = ref_data[ref_data['Theta_i'] == 0]['L'].iloc[0]
    ref_data['L_correction'] = ref_data['L'] - baseline_L
    
    for module in df['Module'].unique():
        if module == reference_module:
            continue
            
        sub = df[df['Module'] == module]
        grouped = sub.groupby('Theta_i')
        
        for theta_i, group in grouped:
            # Get current module's mean values
            current_L = group['L'].mean()
            current_a = group['a'].mean()
            current_b = group['b'].mean()
            
            # Get reference module's values for this angle
            ref_row = ref_data[ref_data['Theta_i'] == theta_i]
            if ref_row.empty:
                continue
                
            ref_L = ref_row['L'].iloc[0]
            ref_a = ref_row['a'].iloc[0]
            ref_b = ref_row['b'].iloc[0]
            L_correction = ref_row['L_correction'].iloc[0]
            
            # Apply glass glare correction to current module's L*
            corrected_L = current_L - L_correction
            
            # Calculate ΔE with glass glare correction
            # Compare corrected L* to baseline L* (0° reference)
            # This removes systematic glass glare effects and shows true color differences
            deltaE_glare_corrected = np.sqrt(
                (corrected_L - baseline_L)**2 +  # L* difference (corrected vs baseline)
                (current_a - ref_a)**2 +         # a* difference
                (current_b - ref_b)**2           # b* difference
            )
            
            # Also calculate chromaticity ΔE for comparison
            deltaE_chromaticity = np.sqrt(
                (current_a - ref_a)**2 +  # a* difference
                (current_b - ref_b)**2    # b* difference
            )
            
            # Full ΔE (for comparison) - includes all differences
            deltaE_full = np.sqrt(
                (current_L - ref_L)**2 +  # L* difference
                (current_a - ref_a)**2 +  # a* difference
                (current_b - ref_b)**2    # b* difference
            )
            
            results.append({
                'Module': module,
                'Theta_i': theta_i,
                'current_L': current_L,
                'current_a': current_a,
                'current_b': current_b,
                'ref_L': ref_L,
                'ref_a': ref_a,
                'ref_b': ref_b,
                'L_correction': L_correction,
                'corrected_L': corrected_L,
                'baseline_L': baseline_L,
                'deltaE_glare_corrected': deltaE_glare_corrected,
                'deltaE_chromaticity': deltaE_chromaticity,
                'deltaE_full': deltaE_full
            })
    
    return pd.DataFrame(results)

def plot_glass_glare_corrected_comparison(corrected_df):
    """
    Plot the different ΔE calculations including glass glare correction.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Glass glare corrected ΔE
    for module in corrected_df['Module'].unique():
        sub = corrected_df[corrected_df['Module'] == module].sort_values('Theta_i')
        ax1.plot(sub['Theta_i'], sub['deltaE_glare_corrected'], 'o-', 
                label=f'Module {module}', alpha=0.7)
    ax1.set_xlabel('Incident Angle (deg)')
    ax1.set_ylabel('ΔE (Glass Glare Corrected)')
    ax1.set_title('ΔE with Glass Glare Correction\n(L* normalized to 0° baseline)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Chromaticity ΔE (a*b* only)
    for module in corrected_df['Module'].unique():
        sub = corrected_df[corrected_df['Module'] == module].sort_values('Theta_i')
        ax2.plot(sub['Theta_i'], sub['deltaE_chromaticity'], 'o-', 
                label=f'Module {module}', alpha=0.7)
    ax2.set_xlabel('Incident Angle (deg)')
    ax2.set_ylabel('ΔE (Chromaticity Only)')
    ax2.set_title('ΔE in a*b* Plane\n(Ignores L* completely)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'glass_glare_corrected_deltaE_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def summarize_glass_glare_corrected_stability(corrected_df):
    """
    Summarize the glass glare corrected color stability metrics.
    """
    summary = corrected_df.groupby('Module').agg({
        'deltaE_glare_corrected': ['min', 'max', 'mean'],
        'deltaE_chromaticity': ['min', 'max', 'mean']
    }).round(3)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    
    summary_path = os.path.join(OUTPUT_DIR, 'glass_glare_corrected_deltaE_summary.csv')
    summary.to_csv(summary_path, index=False)
    print('Glass glare corrected color stability summary:')
    print(summary)
    print(f'Saved summary to {summary_path}')
    return summary

def plot_L_correction_factors(df, reference_module=1):
    """
    Plot the L* correction factors from Module 1 to visualize glass glare effects.
    """
    # Get reference module data (Module 1) to calculate L* correction factors
    ref_data = df[df['Module'] == reference_module].groupby('Theta_i').agg({
        'L': 'mean',
        'a': 'mean', 
        'b': 'mean'
    }).reset_index()
    
    # Calculate L* correction factors (how much L* changes with angle for Module 1)
    baseline_L = ref_data[ref_data['Theta_i'] == 0]['L'].iloc[0]
    ref_data['L_correction'] = ref_data['L'] - baseline_L
    
    plt.figure(figsize=(10, 6))
    
    # Plot L* values vs angle
    plt.subplot(1, 2, 1)
    plt.plot(ref_data['Theta_i'], ref_data['L'], 'o-', linewidth=2, markersize=8, 
             label=f'{MODULE_DESCRIPTIONS[reference_module]}', color='blue')
    plt.axhline(y=baseline_L, color='red', linestyle='--', alpha=0.7, 
                label=f'Baseline L* (0°) = {baseline_L:.2f}')
    plt.xlabel('Incident Angle (deg)')
    plt.ylabel('L* (Lightness)')
    plt.title('L* vs Incident Angle\n(Module 1 Reference)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot L* correction factors
    plt.subplot(1, 2, 2)
    plt.plot(ref_data['Theta_i'], ref_data['L_correction'], 'o-', linewidth=2, markersize=8, 
             color='green', label='L* Correction Factor')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No correction (0°)')
    plt.xlabel('Incident Angle (deg)')
    plt.ylabel('L* Correction Factor')
    plt.title('Glass Glare Correction Factors\n(Subtracted from other modules)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'L_correction_factors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'L* correction factors saved to {os.path.join(OUTPUT_DIR, "L_correction_factors.png")}')
    print('L* correction factors by angle:')
    for _, row in ref_data.iterrows():
        print(f"  {row['Theta_i']}°: {row['L_correction']:.3f}")

def calculate_deltaE_from_csv_over_angles(csv_path, reference_module=1):
    """
    Calculate Delta E over incident angles using the CSV dataset without any glare correction.
    This function loads the CSV data and calculates Delta E between each module and the reference module
    at each incident angle, showing how color differences change with viewing angle.
    
    Args:
        csv_path: Path to the hue_chroma_lab_results_lightness.csv file
        reference_module: Module to use as reference (default: 1)
    
    Returns:
        DataFrame with Delta E values for each module-angle combination
    """
    # Load the CSV data
    df = pd.read_csv(csv_path)
    
    # Filter to only include modules in our MODULES list
    # The CSV already has numeric modules, so we can filter directly
    df = df[df['Module'].isin(MODULES)]
    # Convert to string for consistency with the rest of the code
    df['Module'] = df['Module'].astype(str)
    
    # Debug: print column names and first few rows
    print(f"CSV columns: {df.columns.tolist()}")
    print(f"First few rows after filtering:")
    print(df.head())
    print(f"Module column unique values after filtering: {sorted(df['Module'].unique())}")
    
    # Get unique modules and angles
    modules = df['Module'].unique()
    angles = sorted(df['Theta_i'].unique())
    
    print(f"Found {len(modules)} modules: {sorted(modules)}")
    print(f"Found {len(angles)} angles: {angles}")
    
    # Convert reference module to string to match CSV format
    reference_module_str = str(reference_module)
    print(f"Using reference module: '{reference_module_str}' (type: {type(reference_module_str)})")
    
    results = []
    
    # Calculate mean LAB values for each module-angle combination
    for module in modules:
        for angle in angles:
            # Get data for this module and angle
            module_data = df[(df['Module'] == module) & (df['Theta_i'] == angle)]
            
            if module_data.empty:
                print(f"No data for module {module} at angle {angle}")
                continue
                
            # Calculate mean LAB values
            mean_L = module_data['L'].mean()
            mean_a = module_data['a'].mean()
            mean_b = module_data['b'].mean()
            
            # Get reference module data for this angle
            # Handle both string and numeric reference module
            ref_data = df[(df['Module'] == reference_module_str) & (df['Theta_i'] == angle)]
            
            if ref_data.empty:
                print(f"No reference data for module '{reference_module_str}' at angle {angle}")
                continue
                
            # Calculate mean LAB values for reference module
            ref_L = ref_data['L'].mean()
            ref_a = ref_data['a'].mean()
            ref_b = ref_data['b'].mean()
            
            # Calculate Delta E (Euclidean distance in LAB space)
            deltaE = np.sqrt(
                (mean_L - ref_L)**2 + 
                (mean_a - ref_a)**2 + 
                (mean_b - ref_b)**2
            )
            
            # Also calculate chromaticity Delta E (a*b* only)
            deltaE_chromaticity = np.sqrt(
                (mean_a - ref_a)**2 + 
                (mean_b - ref_b)**2
            )
            
            # Convert module number to short name
            module_int = int(module)
            module_name = MODULE_DESCRIPTIONS.get(module_int, f"Module_{module}")
            
            results.append({
                'Module': module_name,
                'Theta_i': angle,
                'Module_L': mean_L,
                'Module_a': mean_a,
                'Module_b': mean_b,
                'Ref_L': ref_L,
                'Ref_a': ref_a,
                'Ref_b': ref_b,
                'DeltaE': deltaE,
                'DeltaE_chromaticity': deltaE_chromaticity
            })
            
            print(f"Calculated Delta E for module {module} at angle {angle}: {deltaE:.3f}")
    
    result_df = pd.DataFrame(results)
    print(f"Generated {len(result_df)} Delta E calculations")
    if len(result_df) > 0:
        print(f"Result columns: {result_df.columns.tolist()}")
        print(f"Sample results:")
        print(result_df.head())
    else:
        print("No results generated - check reference module matching")
    
    return result_df

def plot_deltaE_over_angles(deltaE_df, output_dir):
    """
    Plot Delta E over incident angles for all modules, separated by colored and non-pigmented.
    
    Args:
        deltaE_df: DataFrame with Delta E results
        output_dir: Directory to save plots
    """
    # Get unique modules (excluding reference module)
    all_modules = deltaE_df['Module'].unique()
    modules = [m for m in all_modules if m != 'REF']
    
    # Separate modules into colored and non-pigmented
    colored_modules = []
    non_pigmented_modules = []
    
    for module in modules:
        if any(color in module for color in ['Blue', 'Brown', 'Green']):
            colored_modules.append(module)
        else:
            non_pigmented_modules.append(module)
    
    # Plot 1: Non-pigmented modules
    plt.figure(figsize=(12, 8))
    
    for module in non_pigmented_modules:
        module_data = deltaE_df[deltaE_df['Module'] == module].sort_values('Theta_i')
        plt.plot(module_data['Theta_i'], module_data['DeltaE'], 'o-', 
                label=f'{module}', linewidth=2, markersize=6, alpha=0.8)
    
    plt.xlabel('Incident Angle (deg)')
    plt.ylabel('ΔE')
    plt.title('Centroid Color Shift ΔE - Non-Pigmented Modules\n(With Lightness Filtering and Non-Specular Peaks)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deltaE_over_angles_non_pigmented.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Colored modules
    plt.figure(figsize=(12, 8))
    
    for module in colored_modules:
        module_data = deltaE_df[deltaE_df['Module'] == module].sort_values('Theta_i')
        plt.plot(module_data['Theta_i'], module_data['DeltaE'], 'o-', 
                label=f'{module}', linewidth=2, markersize=6, alpha=0.8)
    
    plt.xlabel('Incident Angle (deg)')
    plt.ylabel('ΔE')
    plt.title('Centroid Color Shift ΔE - Colored Modules\n(With Lightness Filtering and Non-Specular Peaks)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deltaE_over_angles_colored.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: All modules together (original plot)
    plt.figure(figsize=(12, 8))
    
    for module in modules:
        module_data = deltaE_df[deltaE_df['Module'] == module].sort_values('Theta_i')
        plt.plot(module_data['Theta_i'], module_data['DeltaE'], 'o-', 
                label=f'{module}', linewidth=2, markersize=6, alpha=0.8)
    
    plt.xlabel('Incident Angle (deg)')
    plt.ylabel('ΔE')
    plt.title('Centroid Color Shift ΔE - All Modules\n(With Lightness Filtering and Non-Specular Peaks)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deltaE_over_angles_full.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Chromaticity Delta E over angles (a*b* only)
    plt.figure(figsize=(12, 8))
    
    for module in modules:
        module_data = deltaE_df[deltaE_df['Module'] == module].sort_values('Theta_i')
        plt.plot(module_data['Theta_i'], module_data['DeltaE_chromaticity'], 'o-', 
                label=f'{module}', linewidth=2, markersize=6, alpha=0.8)
    
    plt.xlabel('Incident Angle (deg)')
    plt.ylabel('ΔE (Chromaticity Only)')
    plt.title('Chromaticity Delta E vs Incident Angle\n(With Lightness Filtering and Non-Specular Peaks)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deltaE_over_angles_chromaticity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Comparison of both Delta E types
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Full Delta E
    for module in modules:
        module_data = deltaE_df[deltaE_df['Module'] == module].sort_values('Theta_i')
        ax1.plot(module_data['Theta_i'], module_data['DeltaE'], 'o-', 
                label=f'{module}', linewidth=2, markersize=6, alpha=0.8)
    ax1.set_xlabel('Incident Angle (deg)')
    ax1.set_ylabel('ΔE')
    ax1.set_title('Centroid Color Shift ΔE')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Chromaticity Delta E
    for module in modules:
        module_data = deltaE_df[deltaE_df['Module'] == module].sort_values('Theta_i')
        ax2.plot(module_data['Theta_i'], module_data['DeltaE_chromaticity'], 'o-', 
                label=f'{module}', linewidth=2, markersize=6, alpha=0.8)
    ax2.set_xlabel('Incident Angle (deg)')
    ax2.set_ylabel('ΔE (Chromaticity Only)')
    ax2.set_title('Chromaticity Delta E vs Incident Angle')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deltaE_over_angles_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Delta E over angles plots saved to {output_dir}")

def summarize_deltaE_over_angles(deltaE_df, output_dir):
    """
    Summarize Delta E over angles analysis.
    
    Args:
        deltaE_df: DataFrame with Delta E results
        output_dir: Directory to save summary
    """
    # Calculate summary statistics for each module
    summary = deltaE_df.groupby('Module').agg({
        'DeltaE': ['min', 'max', 'mean', 'std'],
        'DeltaE_chromaticity': ['min', 'max', 'mean', 'std']
    }).round(3)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    
    # Sort by module name for better readability
    # Custom sorting to maintain logical order
    def module_sort_key(module_name):
        # Define custom sort order for module names
        order_map = {
            'REF': 0, 'B5': 1, 'B10': 2, 'B20': 3,
            'G2.1S5': 4, 'G2.1S20': 5, 'G1.5S5': 6, 'G1.5S20': 7,
            'Blue2.1S': 8, 'Blue2.1L': 9, 'Blue1.5S': 10, 'Blue1.5L': 11, 'BlueBa': 12,
            'G2.1L5': 13, 'G2.1L20': 14, 'G1.5L5': 15, 'G1.5L20': 16,
            'BrownC': 17, 'BrownBaC': 18, 'BrownGlC': 19,
            'GreenC': 20, 'GreenBaC': 21, 'GreenGlC': 22
        }
        return order_map.get(module_name, 999)  # Unknown modules at the end
    
    summary = summary.sort_values('Module', key=lambda x: x.map(module_sort_key))
    
    # Save summary
    summary_path = os.path.join(output_dir, 'deltaE_over_angles_summary.csv')
    summary.to_csv(summary_path, index=False)
    
    print('Delta E over angles summary:')
    print(summary)
    print(f'Saved summary to {summary_path}')
    
    return summary

def main():
    df = analyze_all()
    df.to_csv(os.path.join(OUTPUT_DIR, 'lab_results.csv'), index=False)
    plot_lab_scatter(df, theta_of_interest=30)
    plot_lab_scatter_with_angles(df)
    plot_lab_scatter_raw(df, theta_of_interest=30)
    plot_lab_trajectories(df)
    plot_l_vs_angle(df)
    
    # Calculate normalized ΔE using Module 1 (REF) as the reference
    print(f"Using {MODULE_DESCRIPTIONS[1]} as reference module for ΔE normalization")
    normalized_df = calculate_normalized_deltaE(df, reference_module=1)
    normalized_df.to_csv(os.path.join(OUTPUT_DIR, 'normalized_deltaE_results.csv'), index=False)
    plot_normalized_deltaE_comparison(normalized_df)
    summarize_normalized_stability(normalized_df)
    
    # Original centroid analysis
    centroid_df = calculate_centroid_shift_lab(df)
    centroid_df.to_csv(os.path.join(OUTPUT_DIR, 'lab_centroid_shifts.csv'), index=False)
    summarize_color_stability(centroid_df)
    summary = pd.read_csv(os.path.join(OUTPUT_DIR, 'lab_centroid_stability_summary.csv'))
    plot_color_stability_bar(summary)
    
    # Calculate glass glare corrected ΔE
    print(f"Using {MODULE_DESCRIPTIONS[1]} as reference module for glass glare correction")
    corrected_df = calculate_glass_glare_corrected_deltaE(df, reference_module=1)
    corrected_df.to_csv(os.path.join(OUTPUT_DIR, 'glass_glare_corrected_deltaE_results.csv'), index=False)
    plot_glass_glare_corrected_comparison(corrected_df)
    summarize_glass_glare_corrected_stability(corrected_df)
    
    # Plot L* correction factors
    plot_L_correction_factors(df, reference_module=1)
    
    # NEW: Calculate Delta E over incident angles using CSV dataset (no glare correction)
    print("\n" + "="*60)
    print("ANALYZING DELTA E OVER INCIDENT ANGLES FROM CSV DATASET")
    print("(No glare correction applied)")
    print("="*60)
    
    csv_path = 'LAB/all_modules_lab_analysis/HUE_chroma/cielab/nonspecular/lightness_correction/hue_chroma_lab_results_nonspecular_lightness_correction.csv'
    
    if os.path.exists(csv_path):
        print(f"Loading CSV data from: {csv_path}")
        deltaE_csv_df = calculate_deltaE_from_csv_over_angles(csv_path, reference_module=1)
        deltaE_csv_df.to_csv(os.path.join(OUTPUT_DIR, 'deltaE_over_angles_from_csv.csv'), index=False)
        
        # Create plots
        plot_deltaE_over_angles(deltaE_csv_df, OUTPUT_DIR)
        
        # Create summary
        csv_summary = summarize_deltaE_over_angles(deltaE_csv_df, OUTPUT_DIR)
        
        print(f"CSV-based Delta E analysis complete. Results saved to {OUTPUT_DIR}")
    else:
        print(f"Warning: CSV file not found at {csv_path}")
        print("Skipping CSV-based Delta E analysis")
    
    print(f'LAB analysis complete. Plots and data saved in {OUTPUT_DIR}.')

if __name__ == '__main__':
    main() 