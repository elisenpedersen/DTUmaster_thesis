"""
Improved DeltaE Centroid Shift Analysis with Angular Deviation Filtering
This script implements enhanced filtering to improve DeltaE calculations by:
1. Using only backscatter data (45-90° angular deviation)
2. Avoiding specular artifacts
3. Providing more reliable color stability assessment
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colour import SpectralDistribution, sd_to_XYZ, XYZ_to_Lab, MSDS_CMFS, SDS_ILLUMINANTS

# CONFIGURATION
MODULES = [1, 2, 3, 4, 5, 7, 8, 14, 15, 16, 17]
ANGLES = [0, 15, 30, 45, 60]
OUTPUT_DIR = "LAB/lab_analysis_filtered"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODULE_DESCRIPTIONS = {
    1: "Module 1 (REF)", 2: "Module 2 (B5)", 3: "Module 3 (B10)", 4: "Module 4 (B20)",
    5: "Module 5 (G2.1S5)", 7: "Module 7 (G1.5S5)", 8: "Module 8 (G1.5S20)",
    14: "Module 14 (G2.1L5)", 15: "Module 15 (G2.1L20)", 16: "Module 16 (G1.5L5)", 17: "Module 17 (G1.5L20)"
}

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def spectrum_to_lab(wavelengths, spectrum):
    std_wl = np.arange(360, 781, 1)
    std_spectrum = np.interp(std_wl, wavelengths, spectrum)
    sd = SpectralDistribution(dict(zip(std_wl, std_spectrum)))
    XYZ = sd_to_XYZ(sd, MSDS_CMFS['CIE 1931 2 Degree Standard Observer'], SDS_ILLUMINANTS['D65'])
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
    
    # Calculate the dot product between measured reflection and specular direction
    # Specular direction: [sin(theta_i), 0, cos(theta_i)]
    # Measured direction: [sin(theta_r)*cos(phi_r), sin(theta_r)*sin(phi_r), cos(theta_r)]
    
    dot_product = (np.sin(theta_r_rad) * np.cos(phi_r_rad) * np.sin(theta_i_rad) + 
                   np.cos(theta_r_rad) * np.cos(theta_i_rad))
    
    # Clamp dot product to valid range [-1, 1] to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate angular deviation
    angular_deviation = np.degrees(np.arccos(dot_product))
    
    return angular_deviation

def analyze_all_with_angular_data():
    """Load all data including angular information for filtering."""
    all_lab = []
    for module in MODULES:
        for angle in ANGLES:
            pkl_path = os.path.join('brdf_plots', f'Module{module}', f'Module{module}_theta{angle}_phi0.pkl')
            if not os.path.exists(pkl_path):
                print(f"Missing: {pkl_path}")
                continue
            df = load_pkl(pkl_path)
            wavelengths = np.linspace(400, 700, len(df['spec_brdf'].iloc[0]))
            for idx, row in df.iterrows():
                spectrum = np.array(row['spec_brdf'])
                Lab = spectrum_to_lab(wavelengths, spectrum)
                all_lab.append({
                    'Module': module,
                    'Theta_i': angle,
                    'theta_r': row['theta_r'],
                    'phi_r': row.get('phi_r', 0),  # Default to 0 if not available
                    'L': Lab[0],
                    'a': Lab[1],
                    'b': Lab[2]
                })
    return pd.DataFrame(all_lab)

def calculate_centroid_shift_lab_filtered(df):
    """
    Calculate centroid (mean Lab) and DeltaE shifts for each module and angle.
    Uses filtered data based on angular deviation to avoid specular artifacts.
    Only includes backscatter data (45-90° deviation) for most stable measurements.
    """
    results = []
    
    for module in df['Module'].unique():
        sub = df[df['Module'] == module]
        grouped = sub.groupby('Theta_i')
        centroids = {}
        
        for theta_i, group in grouped:
            # Apply angular deviation filtering
            filtered_group = []
            
            for idx, row in group.iterrows():
                theta_r = row['theta_r']
                phi_r = row.get('phi_r', 0)
                
                # Calculate angular deviation from specular
                angular_deviation = calculate_angular_deviation_from_specular(theta_i, theta_r, phi_r)
                
                # Filter: only include backscatter data (45-90° deviation) for most stable measurements
                # Also exclude extreme specular artifacts
                if (angular_deviation >= 45 and angular_deviation <= 90 and 
                    not (angular_deviation <= 15 and row['L'] < 10)):
                    filtered_group.append(row)
            
            if filtered_group:
                filtered_df = pd.DataFrame(filtered_group)
                labs = filtered_df[['L', 'a', 'b']].values
                centroid = labs.mean(axis=0)
                centroids[theta_i] = centroid
                print(f"Module {module}, θᵢ={theta_i}°: {len(filtered_group)}/{len(group)} points used (backscatter filtered)")
            else:
                print(f"Module {module}, θᵢ={theta_i}°: No valid backscatter data found")
                continue
        
        if not centroids:
            print(f"Module {module}: No valid centroids calculated")
            continue
            
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

def calculate_centroid_shift_lab_unfiltered(df):
    """Original centroid calculation without filtering for comparison."""
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

def summarize_color_stability(centroid_df, suffix=""):
    """Summarize color stability for each module."""
    summary = centroid_df.groupby('Module')['DeltaE_to_ref'].agg(['min', 'max', 'mean']).reset_index()
    summary = summary.rename(columns={'min': 'DeltaE_min', 'max': 'DeltaE_max', 'mean': 'DeltaE_mean'})
    summary_path = os.path.join(OUTPUT_DIR, f'lab_centroid_stability_summary{suffix}.csv')
    summary.to_csv(summary_path, index=False)
    print(f'Color stability summary{suffix}:')
    print(summary)
    print(f'Saved summary to {summary_path}')
    return summary

def plot_color_stability_comparison(unfiltered_summary, filtered_summary):
    """Create comparison plot between filtered and unfiltered results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Unfiltered results
    modules = unfiltered_summary['Module'].astype(str)
    max_deltaE_unfiltered = unfiltered_summary['DeltaE_max']
    bars1 = ax1.bar(modules, max_deltaE_unfiltered, color='lightcoral', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Module')
    ax1.set_ylabel('Max ΔE to Reference')
    ax1.set_title('Unfiltered Analysis (All Data)')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, max_deltaE_unfiltered):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Filtered results
    max_deltaE_filtered = filtered_summary['DeltaE_max']
    bars2 = ax2.bar(modules, max_deltaE_filtered, color='lightblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Module')
    ax2.set_ylabel('Max ΔE to Reference')
    ax2.set_title('Filtered Analysis (Backscatter Only)')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, max_deltaE_filtered):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'centroid_stability_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved comparison plot to {os.path.join(OUTPUT_DIR, "centroid_stability_comparison.png")}')

def plot_improvement_analysis(unfiltered_summary, filtered_summary):
    """Analyze and plot the improvement from filtering."""
    # Merge summaries for comparison
    comparison = unfiltered_summary.merge(filtered_summary, on='Module', suffixes=('_unfiltered', '_filtered'))
    comparison['improvement'] = comparison['DeltaE_max_unfiltered'] - comparison['DeltaE_max_filtered']
    comparison['improvement_percent'] = (comparison['improvement'] / comparison['DeltaE_max_unfiltered']) * 100
    
    # Create improvement plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Absolute improvement
    modules = comparison['Module'].astype(str)
    bars1 = ax1.bar(modules, comparison['improvement'], 
                   color=['green' if x > 0 else 'red' for x in comparison['improvement']], 
                   edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Module')
    ax1.set_ylabel('ΔE Improvement (Unfiltered - Filtered)')
    ax1.set_title('Absolute Improvement in Max ΔE')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Percentage improvement
    bars2 = ax2.bar(modules, comparison['improvement_percent'], 
                   color=['green' if x > 0 else 'red' for x in comparison['improvement_percent']], 
                   edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Module')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Percentage Improvement in Max ΔE')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'filtering_improvement_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print improvement summary
    print("\n" + "="*60)
    print("FILTERING IMPROVEMENT ANALYSIS")
    print("="*60)
    print("Modules with improved stability (lower ΔE after filtering):")
    improved = comparison[comparison['improvement'] > 0]
    if not improved.empty:
        for _, row in improved.iterrows():
            print(f"  Module {row['Module']}: {row['improvement']:.2f} ΔE improvement ({row['improvement_percent']:.1f}%)")
    else:
        print("  No modules showed improvement")
    
    print("\nModules with worse stability (higher ΔE after filtering):")
    worsened = comparison[comparison['improvement'] < 0]
    if not worsened.empty:
        for _, row in worsened.iterrows():
            print(f"  Module {row['Module']}: {abs(row['improvement']):.2f} ΔE increase ({abs(row['improvement_percent']):.1f}%)")
    else:
        print("  No modules showed worse stability")
    
    # Save comparison data
    comparison.to_csv(os.path.join(OUTPUT_DIR, 'filtering_comparison_analysis.csv'), index=False)
    print(f"\nSaved detailed comparison to {os.path.join(OUTPUT_DIR, 'filtering_comparison_analysis.csv')}")

def main():
    print("Starting improved DeltaE centroid shift analysis with angular filtering...")
    
    # Load data with angular information
    print("Loading data with angular information...")
    df = analyze_all_with_angular_data()
    df.to_csv(os.path.join(OUTPUT_DIR, 'lab_results_with_angular_data.csv'), index=False)
    print(f"Saved data with angular information to {os.path.join(OUTPUT_DIR, 'lab_results_with_angular_data.csv')}")
    
    # Run unfiltered analysis for comparison
    print("\n" + "="*60)
    print("UNFILTERED CENTROID ANALYSIS (ALL DATA)")
    print("="*60)
    unfiltered_centroid_df = calculate_centroid_shift_lab_unfiltered(df)
    unfiltered_centroid_df.to_csv(os.path.join(OUTPUT_DIR, 'lab_centroid_shifts_unfiltered.csv'), index=False)
    unfiltered_summary = summarize_color_stability(unfiltered_centroid_df, "_unfiltered")
    
    # Run filtered analysis
    print("\n" + "="*60)
    print("FILTERED CENTROID ANALYSIS (BACKSCATTER ONLY)")
    print("="*60)
    filtered_centroid_df = calculate_centroid_shift_lab_filtered(df)
    filtered_centroid_df.to_csv(os.path.join(OUTPUT_DIR, 'lab_centroid_shifts_filtered.csv'), index=False)
    filtered_summary = summarize_color_stability(filtered_centroid_df, "_filtered")
    
    # Create comparison plots
    print("\n" + "="*60)
    print("CREATING COMPARISON PLOTS")
    print("="*60)
    plot_color_stability_comparison(unfiltered_summary, filtered_summary)
    plot_improvement_analysis(unfiltered_summary, filtered_summary)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print("Best color stability (lowest max DeltaE) - Filtered analysis:")
    best_modules = filtered_summary.nsmallest(3, 'DeltaE_max')
    for _, row in best_modules.iterrows():
        print(f"  Module {row['Module']}: {row['DeltaE_max']:.2f}")
    
    print("\nWorst color stability (highest max DeltaE) - Filtered analysis:")
    worst_modules = filtered_summary.nlargest(3, 'DeltaE_max')
    for _, row in worst_modules.iterrows():
        print(f"  Module {row['Module']}: {row['DeltaE_max']:.2f}")
    
    print(f"\nAnalysis complete. All results saved in {OUTPUT_DIR}")

if __name__ == '__main__':
    main() 