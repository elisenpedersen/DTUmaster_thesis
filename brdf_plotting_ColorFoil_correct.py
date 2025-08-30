import os
import glob
import shutil
import pickle
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import colour.plotting as cplt
from colour import delta_E

# Constants
CMF_FILE = './utils/reference_data/CIE2degCMF1931forMATLAB.txt'
D65_FILE = './utils/reference_data/CIED65.txt'
XYZ_D65 = np.array([95.0489, 100, 108.8840])

# Module descriptions (for plot labels). Values are short names only.
MODULE_DESCRIPTIONS = {
    1: "REF",
    2: "B5",
    3: "B10",
    4: "B20",
    5: "Gl2.1S5",
    6: "Gl2.1S20",
    7: "Gl1.5S5",
    8: "Gl1.5S20",
    9: "Blue2.1S",
    10: "Blue2.1L",
    11: "Blue1.5S",
    12: "Blue1.5L",
    13: "BlueBa",
    14: "Gl2.1L5",
    15: "Gl2.1L20",
    16: "Gl1.5L5",
    17: "Gl1.5L20",
    18: "BrownC",
    20: "BrownBaC",
    22: "BrownGlC",
    25: "GreenC",
    27: "GreenBaC",
    28: "GreenGlC",
}

def get_module_short_name(module_name: str) -> str:
    """Extract module number and return short name label.
    
    Args:
        module_name: Module name like 'Module9' or 'Module20'
        
    Returns:
        Short name for the module
    """
    # Extract module number from 'Module{number}'
    if module_name.startswith('Module'):
        try:
            module_num = int(module_name[6:])  # Remove 'Module' prefix
            return MODULE_DESCRIPTIONS.get(module_num, module_name)
        except ValueError:
            return module_name
    return module_name

# Global variable to store unified BRDF range
UNIFIED_BRDF_RANGE = None

# Load reference data
CMF = np.genfromtxt(CMF_FILE)
D65 = np.genfromtxt(D65_FILE, delimiter='\t')

def spec2XYZ(wvl: np.ndarray, spec: np.ndarray, cmf: np.ndarray, illum: np.ndarray) -> np.ndarray:
    """Convert spectral data to XYZ color space.
    
    Args:
        wvl: Wavelength array
        spec: Spectral data
        cmf: Color matching functions
        illum: Illuminant data
        
    Returns:
        XYZ color values
    """
    Iinterp = np.interp(cmf[:,0], illum[:,0], illum[:,1])
    if len(np.shape(spec)) > 1:
        Rinterp = [np.interp(cmf[:,0], wvl, s) for s in spec]
    else:
        Rinterp = np.interp(cmf[:,0], wvl, spec)
    
    L = Iinterp * Rinterp
    N = np.trapz(Iinterp * cmf[:,2], x=cmf[:,0])
    X = np.trapz(L * cmf[:,1], x=cmf[:,0]) / N
    Y = np.trapz(L * cmf[:,2], x=cmf[:,0]) / N
    Z = np.trapz(L * cmf[:,3], x=cmf[:,0]) / N
    
    return np.array([X, Y, Z]).T

def XYZ2Lab(XYZ: np.ndarray, XYZref: np.ndarray) -> np.ndarray:
    """Convert XYZ to Lab color space.
    
    Args:
        XYZ: XYZ color values
        XYZref: Reference XYZ values
        
    Returns:
        Lab color values
    """
    ft = np.where(XYZ/XYZref > (6/29)**3,
                 np.cbrt(XYZ/XYZref),
                 XYZ/XYZref/3/(6/29)**2 + 4/29)
    L = 116 * ft[1] - 16
    a = 500 * (ft[0] - ft[1])
    b = 200 * (ft[1] - ft[2])
    return np.array([L, a, b]).T

def plot_chromaticity(results_df: pd.DataFrame, module_name: str, save_path: Optional[str] = None, use_log: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """Create a chromaticity diagram plot with BRDF values.
    
    Args:
        results_df: DataFrame containing x, y, and total_brdf values
        module_name: Name of the module being plotted
        save_path: Optional path to save the figure
        use_log: Whether to use logarithmic scaling for BRDF values
        
    Returns:
        Tuple of figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')  # Reduced figure size
    fig.subplots_adjust(left=0.14, right=0.86, top=0.88, bottom=0.12)  # Increased margins for more white space
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Create scatter plot with completely see-through points (no BRDF colors)
    scatter = ax.scatter(
        results_df.x, results_df.y, 
        c='lightgray',  # Light gray color for subtle appearance
        marker='o', edgecolor='black',
        s=100, alpha=0.4, linewidth=0.8, zorder=3  # Light gray with transparency
    )
    
    # Add chromaticity diagram
    cplt.diagrams.plot_chromaticity_diagram(axes=ax, show=False)

    # Remove axis labels and ticks
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # Add title inside the chromaticity diagram (more compact)
    theta_i = results_df['theta_i'].iloc[0]  # Get theta_i from the first row
    short_name = get_module_short_name(module_name)
    title = f'{short_name} - at incident angle {theta_i}°'  # Keep full title
    
    # Position title inside the diagram at the top center, moved higher
    ax.text(0.5, 0.9, title, fontsize=16, fontweight='normal',  # No bold, moved to 0.9
            ha='center', va='center', transform=ax.transAxes,
            color='black', bbox=dict(facecolor='white', alpha=0.8,
                                   edgecolor='black', linewidth=0.5,
                                   boxstyle='round,pad=0.3'))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   pad_inches=0.1, facecolor=fig.get_facecolor())  # Reduced padding
    
    return fig, ax

def calculate_unified_brdf_range():
    """Calculate the unified BRDF range across all modules for consistent color scaling.
    
    Returns:
        Tuple of (min_brdf, max_brdf) for unified color scaling
    """
    global UNIFIED_BRDF_RANGE
    
    print("Calculating unified BRDF range across all modules...")
    
    all_brdf_values = []
    module_folders = [f for f in glob.glob('brdf_plots/Module*') if os.path.isdir(f)]
    
    for module_folder in module_folders:
        pkl_files = glob.glob(os.path.join(module_folder, '*.pkl'))
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, 'rb') as f:
                    df = pickle.load(f)
                if 'total_brdf' in df.columns:
                    # Filter out negative or zero values for log scaling
                    positive_brdf = df['total_brdf'][df['total_brdf'] > 0]
                    if len(positive_brdf) > 0:
                        all_brdf_values.extend(positive_brdf.values)
            except Exception as e:
                print(f"Error reading {pkl_file}: {e}")
    
    if all_brdf_values:
        min_brdf = np.min(all_brdf_values)
        max_brdf = np.max(all_brdf_values)
        UNIFIED_BRDF_RANGE = (min_brdf, max_brdf)
        print(f"Unified BRDF range: {min_brdf:.2e} to {max_brdf:.2e}")
        return UNIFIED_BRDF_RANGE
    else:
        print("Warning: No BRDF data found, using default scaling")
        return None

def process_module_folder(module_folder: str) -> None:
    """Process all .pkl files in a module folder and create chromaticity plots.
    
    Args:
        module_folder: Path to the module folder containing .pkl files
    """
    module_name = os.path.basename(module_folder)
    output_dir = os.path.join('Chromaticity', module_name)
    os.makedirs(output_dir, exist_ok=True)
    
    pkl_files = glob.glob(os.path.join(module_folder, '*.pkl'))
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                df = pickle.load(f)
            
            # Calculate color values
            df.loc[:,'xyz'] = df.apply(lambda r: spec2XYZ(r.wavelength, r.spec_brdf, CMF, D65), axis=1)
            df.loc[:,'lab'] = df.apply(lambda r: XYZ2Lab(r.xyz*100, XYZ_D65), axis=1)
            df.loc[:,'x'] = df.apply(lambda r: r.xyz[0]/np.sum(r.xyz), axis=1)
            df.loc[:,'y'] = df.apply(lambda r: r.xyz[1]/np.sum(r.xyz), axis=1)
            
            # Create and save plot
            base_name = os.path.splitext(os.path.basename(pkl_file))[0]
            output_file = os.path.join(output_dir, f'{base_name}_chromaticity.png')
            fig, ax = plot_chromaticity(df, module_name, save_path=output_file)
            plt.close(fig)
            print(f"Created chromaticity plot: {output_file}")
        except Exception as e:
            print(f"Error processing {pkl_file}: {str(e)}")

def plot_blue_modules_overlay(module_numbers: list = ['Module24', 'Module25', 'Module26', 'Module27', 'Module28', 'Module29'], theta_i: float = 30):
    """Plot multiple blue pigmented PV modules in the same chromaticity diagram.
    
    Args:
        module_numbers: List of module numbers to compare
        theta_i: Incident angle in degrees
    """
    # Create figure with more compact size
    fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Adjusted margins to keep title inside
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Define line styles for different modules
    line_styles = ['-', '-', '-', '-', '-', '-']  # All solid lines
    colors = ['black', 'dimgray', 'gray', 'darkgray', 'silver', 'lightgray']
    markers = ['o', 's', '^', 'D', 'v', 'o']
    
    # Plot each module
    for module, color, style, marker in zip(module_numbers, colors, line_styles, markers):
        pkl_file = f'brdf_plots/{module}/{module}_theta{theta_i}_phi0.pkl'
        with open(pkl_file, 'rb') as f:
            df = pickle.load(f)
        
        # Calculate color values if not already present
        if 'xyz' not in df.columns:
            df.loc[:,'xyz'] = df.apply(lambda r: spec2XYZ(r.wavelength, r.spec_brdf, CMF, D65), axis=1)
            df.loc[:,'lab'] = df.apply(lambda r: XYZ2Lab(r.xyz*100, XYZ_D65), axis=1)
            df.loc[:,'x'] = df.apply(lambda r: r.xyz[0]/np.sum(r.xyz), axis=1)
            df.loc[:,'y'] = df.apply(lambda r: r.xyz[1]/np.sum(r.xyz), axis=1)
        
        # Sort data points by angle to ensure proper line connections
        df = df.sort_values('theta_r')
        
        # Create line plot with different line styles
        short_name = get_module_short_name(module)
        ax.plot(df.x, df.y, color=color, linestyle=style, linewidth=2, marker=marker, markersize=2, label=short_name, zorder=3, alpha=0.8)
    
    # Add chromaticity diagram
    cplt.diagrams.plot_chromaticity_diagram(axes=ax, show=False)
    
    # Remove axis labels and ticks
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    
    # Add title inside the diagram (more compact)
    title = f'Green Modules Comparison - at incident angle {theta_i}°'  # Keep full title
    ax.text(0.5, 0.9, title, fontsize=16, fontweight='normal',  # No bold, moved to 0.9
            ha='center', va='center', transform=ax.transAxes,
            color='black', bbox=dict(facecolor='white', alpha=0.8,
                                   edgecolor='black', linewidth=0.5,
                                   boxstyle='round,pad=0.3'))
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1),
             fontsize=12, framealpha=0.9)
    
    # Save figure
    save_path = f'Chromaticity/green_modules_overlay_theta_i_{theta_i}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                pad_inches=0.1, facecolor='white')  # Reduced padding
    plt.close(fig)
    print(f"Created overlay plot: {save_path}")

def main() -> None:
    """Main function to process specific module folders and generate chromaticity plots."""
    # Create Chromaticity folder if it doesn't exist
    os.makedirs('Chromaticity', exist_ok=True)
    
    # Calculate unified BRDF range first for consistent color scaling
    calculate_unified_brdf_range()
    
    # Define the specific modules to process (pigmented + non-pigmented)
    PIGMENTED_MODULES = [9, 10, 11, 12, 13, 18, 20, 22, 25, 27, 28]  # All pigmented modules (Blue, Brown, Green)
    NON_PIGMENTED_MODULES = [1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17]  # Glass and BaSO4 modules
    
    # Combine all modules to process
    MODULES_TO_PROCESS = PIGMENTED_MODULES + NON_PIGMENTED_MODULES
    
    print(f"\nProcessing {len(MODULES_TO_PROCESS)} specific modules...")
    for module_num in MODULES_TO_PROCESS:
        module_folder = f'brdf_plots/Module{module_num}'
        if os.path.isdir(module_folder):
            print(f"\nProcessing {module_folder}...")
            process_module_folder(module_folder)
        else:
            print(f"Warning: {module_folder} not found, skipping...")
    
    # Always create overlay plot for blue modules (do not skip if file exists)
    print("\nCreating overlay plot for blue modules...")
    plot_blue_modules_overlay()
    
    print("\nChromaticity plot generation complete! Check the 'Chromaticity' directory for the output files.")

if __name__ == "__main__":
    main() 