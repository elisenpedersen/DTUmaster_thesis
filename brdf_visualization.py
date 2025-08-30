import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import re

#script thats logarithmic and has a visible light colormap, saves plots to a folder called enhanced_plots

#run brdf_analysis_ColorFoil_correct.py first to get the pkl files and then to line 220
def load_brdf_data(pkl_file):
    """Load BRDF data from pickle file"""
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)

def create_visible_light_colormap():
    """Create a custom colormap that represents visible light wavelengths"""
    # Define colors corresponding to visible light wavelengths
    colors = [
        (0.0, 0.0, 0.0),      # Black (no light)
        (0.0, 0.0, 0.5),      # Deep blue
        (0.0, 0.5, 1.0),      # Blue
        (0.0, 1.0, 1.0),      # Cyan
        (0.5, 1.0, 0.5),      # Green
        (1.0, 1.0, 0.0),      # Yellow
        (1.0, 0.5, 0.0),      # Orange
        (1.0, 0.0, 0.0),      # Red
        (0.5, 0.0, 0.0)       # Deep red
    ]
    return LinearSegmentedColormap.from_list('visible_light', colors)

def extract_module_number(filename):
    """Extract module number from filename"""
    match = re.search(r'Module(\d+)', filename)
    return match.group(1) if match else 'Unknown'

def plot_brdf_2d_enhanced(results_df, z_limits=None, use_visible_colormap=True, 
                         parameter='total_brdf', save_path=None, module_num=None):
    """
    Create an enhanced 2D visualization of BRDF data with true logarithmic scaling
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        DataFrame containing BRDF measurements
    z_limits : tuple, optional
        (min, max) limits for the colorbar in log₁₀ space
    use_visible_colormap : bool
        Whether to use the visible light colormap
    parameter : str
        Which parameter to plot ('total_brdf', 'radiance', 'radiant_flux')
    save_path : str, optional
        If provided, save the plot to this path
    module_num : str, optional
        Module number to display in title
    """
    theta_r = np.radians(results_df['theta_r'])
    phi_r = np.radians(results_df['phi_r'])
    values = results_df[parameter].copy()
    theta_i = np.radians(results_df['theta_i'].iloc[0])
    phi_i = np.radians(results_df['phi_i'].iloc[0])

    # Convert spherical to Cartesian coordinates
    x = np.sin(theta_r) * np.cos(phi_r)
    y = np.sin(theta_r) * np.sin(phi_r)

    # Plot the incident vector
    x_i = np.sin(theta_i) * np.cos(phi_i)
    y_i = np.sin(theta_i) * np.sin(phi_i)

    # Create a 2D surface plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Set up colormap
    if use_visible_colormap:
        cmap = create_visible_light_colormap()
    else:
        cmap = 'coolwarm'
    
    # Handle non-positive values before taking log
    values = np.maximum(values, 1e-10)  # Replace zeros and negative values with small positive number
    
    # Transform values to logarithmic scale
    log_values = np.log10(values)
    
    # Set up limits
    if z_limits:
        vmin, vmax = np.log10(z_limits)  # Convert limits to log space
    else:
        vmin = log_values.min()
        vmax = log_values.max()
    
    # Create contour plot with linear spacing in log space
    contour = ax.tricontourf(x, y, log_values, cmap=cmap,
                            levels=np.linspace(vmin, vmax, 50),
                            extend='neither')
    
    # Add colorbar with appropriate label
    cbar = plt.colorbar(contour)
    cbar.set_label(f'log₁₀({parameter.replace("_", " ").title()}) [1/sr]')
    
    # Format colorbar ticks to be cleaner
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{tick:.1f}' for tick in ticks])
    
    # Plot incident light direction
    ax.quiver(x_i, y_i, -x_i, -y_i, angles='xy', scale_units='xy', scale=1, 
              color='black', label=f"$\\theta_i$={np.degrees(theta_i):.0f}°, $\\phi_i$={np.degrees(phi_i):.0f}°")

    # Add circles showing theta angles
    theta_circles = [0, 10, 30, 60, 80, 90]
    for theta in theta_circles:
        circle = plt.Circle((0, 0), np.sin(np.radians(theta)), 
                          linewidth=0.5, color='black', fill=False, linestyle='dotted')
        ax.add_artist(circle)

    # Add lines showing phi angles
    phi_lines = [0, 45, 90, 135, 180, 225, 270, 315]
    for phi in phi_lines:
        x_phi = [0, np.cos(np.radians(phi))]
        y_phi = [0, np.sin(np.radians(phi))]
        ax.plot(x_phi, y_phi, linewidth=0.5, color='black', linestyle='dotted')
    
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f"Module {module_num} - log₁₀({parameter.replace('_', ' ').title()}) at $\\theta_i$={np.degrees(theta_i):.0f}°, $\\phi_i$={np.degrees(phi_i):.0f}°")
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_brdf_3d_enhanced(results_df, z_limits=None, use_visible_colormap=True,
                         parameter='total_brdf', save_path=None, module_num=None):
    """
    Create an enhanced 3D visualization of BRDF data with true logarithmic scaling
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        DataFrame containing BRDF measurements
    z_limits : tuple, optional
        (min, max) limits for the z-axis in log₁₀ space
    use_visible_colormap : bool
        Whether to use the visible light colormap
    parameter : str
        Which parameter to plot ('total_brdf', 'radiance', 'radiant_flux')
    save_path : str, optional
        If provided, save the plot to this path
    module_num : str, optional
        Module number to display in title
    """
    theta_r = np.radians(results_df['theta_r'])
    phi_r = np.radians(results_df['phi_r'])
    values = results_df[parameter].copy()
    theta_i = np.radians(results_df['theta_i'].iloc[0])
    phi_i = np.radians(results_df['phi_i'].iloc[0])

    # Convert spherical to Cartesian coordinates
    x = np.sin(theta_r) * np.cos(phi_r)
    y = np.sin(theta_r) * np.sin(phi_r)
    
    # Handle non-positive values before taking log
    values = np.maximum(values, 1e-10)  # Replace zeros and negative values with small positive number
    
    # Transform values to logarithmic scale
    log_values = np.log10(values)
    z = log_values

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set up colormap
    if use_visible_colormap:
        cmap = create_visible_light_colormap()
    else:
        cmap = 'coolwarm'

    # Set up limits
    if z_limits:
        vmin, vmax = np.log10(z_limits)  # Convert limits to log space
    else:
        vmin = log_values.min()
        vmax = log_values.max()

    # Create a 3D surface plot
    surf = ax.plot_trisurf(x, y, z, cmap=cmap, linewidth=0.2,
                          vmin=vmin, vmax=vmax)
    
    # Add colorbar with appropriate label
    cbar = plt.colorbar(surf)
    cbar.set_label(f'log₁₀({parameter.replace("_", " ").title()}) [1/sr]')
    
    # Format colorbar ticks to be cleaner
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{tick:.1f}' for tick in ticks])

    # Set plot limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    if z_limits:
        ax.set_zlim([vmin, vmax])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('')  # Remove z-axis label

    # Adjust the aspect ratio
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.0, 1.0, 0.8, 1]))

    # Set the view angle
    ax.view_init(30, 45)

    # Customize ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    
    # Set z-axis ticks properly
    z_ticks = np.linspace(vmin, vmax, 5)
    ax.zaxis.set_major_locator(ticker.FixedLocator(z_ticks))
    ax.set_zticklabels([f'{tick:.1f}' for tick in z_ticks])

    # Customize panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')

    # Set title
    plt.title(f"Module {module_num} - log₁₀({parameter.replace('_', ' ').title()}) at $\\theta_i$={np.degrees(theta_i):.0f}°, $\\phi_i$={np.degrees(phi_i):.0f}°")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

if __name__ == "__main__":
    # Create output directory for enhanced plots
    import os
    import sys
    
    # Define modules and angles to plot
    modules = [17]  # Add more module numbers as needed
    theta_i_values = [0, 15, 30, 45, 60]  # Add more angles as needed
    
    # Create base output directory
    base_output_dir = 'enhanced_plots'
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Check if brdf_plots directory exists
    if not os.path.exists('brdf_plots'):
        print("Error: 'brdf_plots' directory not found!")
        sys.exit(1)
    
    for module in modules:
        # Create module-specific directory
        module_dir = os.path.join(base_output_dir, f'Module{module}')
        os.makedirs(module_dir, exist_ok=True)
        
        # Check if module directory exists in brdf_plots
        module_input_dir = os.path.join('brdf_plots', f'Module{module}')
        if not os.path.exists(module_input_dir):
            print(f"Error: Input directory for Module {module} not found at {module_input_dir}")
            continue
        
        for theta_i in theta_i_values:
            # Update the pkl file path with current module and theta_i
            pkl_file = os.path.join('brdf_plots', f'Module{module}', f'Module{module}_theta{theta_i}_phi0.pkl')
            
            # Check if file exists before processing
            if os.path.exists(pkl_file):
                try:
                    results_df = load_brdf_data(pkl_file)
                    
                    # Plot with custom z-limits
                    z_limits = (1e-2, 1e1)
                    plot_brdf_2d_enhanced(results_df, z_limits=z_limits, use_visible_colormap=True,
                                        module_num=module,
                                        save_path=os.path.join(module_dir, f'Module{module}_brdf_2d_zlim_theta{theta_i}.png'))
                    plot_brdf_3d_enhanced(results_df, z_limits=z_limits, use_visible_colormap=True,
                                        module_num=module,
                                        save_path=os.path.join(module_dir, f'Module{module}_brdf_3d_zlim_theta{theta_i}.png'))
                    
                except Exception as e:
                    print(f"Error processing {pkl_file}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"File not found: {pkl_file}")
    
    print("Script finished running.")