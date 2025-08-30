import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import re


# Script to plot the brdf with logarithmic scale for a single module to the log_plots folder. 
# Compares diffuseness nad specular peak

def load_brdf_data(pkl_file):
    """Load BRDF data from pickle file"""
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)

def extract_module_number(filename):
    """Extract module number from filename"""
    match = re.search(r'Module(\d+)', filename)
    return match.group(1) if match else 'Unknown'

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

def plot_brdf_log_2d(results_df, z_limits=None, use_visible_colormap=True, 
                     parameter='total_brdf', save_path=None, module_num=None,
                     n_levels=50):
    """
    Create a 2D visualization of BRDF data with logarithmic colorbar
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        DataFrame containing BRDF measurements
    z_limits : tuple, optional
        (min, max) limits for the colorbar
    use_visible_colormap : bool
        Whether to use the visible light colormap
    parameter : str
        Which parameter to plot ('total_brdf', 'radiance', 'radiant_flux')
    save_path : str, optional
        If provided, save the plot to this path
    module_num : str, optional
        Module number to display in title
    n_levels : int
        Number of levels in the contour plot
    """
    theta_r = np.radians(results_df['theta_r'])
    phi_r = np.radians(results_df['phi_r'])
    values = results_df[parameter]
    theta_i = np.radians(results_df['theta_i'].iloc[0])
    phi_i = np.radians(results_df['phi_i'].iloc[0])

    # Convert spherical to Cartesian coordinates
    x = np.sin(theta_r) * np.cos(phi_r)
    y = np.sin(theta_r) * np.sin(phi_r)

    # Plot the incident vector
    x_i = np.sin(theta_i) * np.cos(phi_i)
    y_i = np.sin(theta_i) * np.sin(phi_i)

    # Create a 2D surface plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up colormap
    if use_visible_colormap:
        cmap = create_visible_light_colormap()
    else:
        cmap = 'viridis'  # Good colormap for logarithmic data
    
    # Set up normalization
    if z_limits:
        vmin, vmax = z_limits
    else:
        vmin = max(values.min(), 1e-3)
        vmax = values.max()
    
    # Create logarithmic levels
    levels = np.logspace(np.log10(vmin), np.log10(vmax), n_levels)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    
    # Create contour plot
    contour = ax.tricontourf(x, y, values, cmap=cmap, norm=norm,
                            levels=levels, extend='both')
    
    # Add colorbar with appropriate label and format
    cbar = plt.colorbar(contour, label=f'{parameter.replace("_", " ").title()} [1/sr]')
    cbar.ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    cbar.ax.yaxis.set_minor_formatter(ticker.LogFormatterMathtext())
    
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
    ax.set_title(f"Module {module_num} - {parameter.replace('_', ' ').title()} at $\\theta_i$={np.degrees(theta_i):.0f}°, $\\phi_i$={np.degrees(phi_i):.0f}°")
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_brdf_log_3d(results_df, z_limits=None, use_visible_colormap=True,
                     parameter='total_brdf', save_path=None, module_num=None):
    """
    Create a 3D visualization of BRDF data with logarithmic colorbar
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        DataFrame containing BRDF measurements
    z_limits : tuple, optional
        (min, max) limits for the z-axis
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
    values = results_df[parameter]
    theta_i = np.radians(results_df['theta_i'].iloc[0])
    phi_i = np.radians(results_df['phi_i'].iloc[0])

    # Convert spherical to Cartesian coordinates
    x = np.sin(theta_r) * np.cos(phi_r)
    y = np.sin(theta_r) * np.sin(phi_r)
    z = values

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set up colormap
    if use_visible_colormap:
        cmap = create_visible_light_colormap()
    else:
        cmap = 'viridis'  # Good colormap for logarithmic data

    # Set up normalization
    if z_limits:
        vmin, vmax = z_limits
    else:
        vmin = max(values.min(), 1e-3)
        vmax = values.max()
    
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Create a 3D surface plot
    surf = ax.plot_trisurf(x, y, z, cmap=cmap, norm=norm, linewidth=0.2)
    
    # Add colorbar with appropriate label and format
    cbar = plt.colorbar(surf, label=f'{parameter.replace("_", " ").title()} [1/sr]')
    cbar.ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    cbar.ax.yaxis.set_minor_formatter(ticker.LogFormatterMathtext())

    # Set plot limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    if z_limits:
        ax.set_zlim(z_limits)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(f'{parameter.replace("_", " ").title()} [1/sr]', labelpad=10)

    # Adjust the aspect ratio
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.0, 1.0, 0.8, 1]))

    # Set the view angle
    ax.view_init(30, 45)

    # Customize ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.zaxis.set_major_formatter(ticker.LogFormatterMathtext())

    # Customize panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')

    # Set title
    plt.title(f"Module {module_num} - {parameter.replace('_', ' ').title()} at $\\theta_i$={np.degrees(theta_i):.0f}°, $\\phi_i$={np.degrees(phi_i):.0f}°")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

if __name__ == "__main__":
    # Example usage
    pkl_file = "brdf_plots/Module13_theta45_phi0.pkl"  # Update this path as needed
    results_df = load_brdf_data(pkl_file)
    
    # Extract module number from filename
    module_num = extract_module_number(pkl_file)
    
    # Create output directory for enhanced plots
    import os
    os.makedirs('log_plots', exist_ok=True)
    
    # Plot with visible light colormap and custom z-limits
    z_limits = (1e-3, 1e2)  # Adjust these values as needed
    
    # Create 2D plot with logarithmic scale
    plot_brdf_log_2d(results_df, z_limits=z_limits, use_visible_colormap=True,
                     module_num=module_num,
                     save_path=f'log_plots/Module{module_num}_brdf_2d_log.png')
    
    # Create 3D plot with logarithmic scale
    plot_brdf_log_3d(results_df, z_limits=z_limits, use_visible_colormap=True,
                     module_num=module_num,
                     save_path=f'log_plots/Module{module_num}_brdf_3d_log.png')
    
    # Create plots with viridis colormap (good for logarithmic data)
    plot_brdf_log_2d(results_df, z_limits=z_limits, use_visible_colormap=False,
                     module_num=module_num,
                     save_path=f'log_plots/Module{module_num}_brdf_2d_log_viridis.png')
    
    plot_brdf_log_3d(results_df, z_limits=z_limits, use_visible_colormap=False,
                     module_num=module_num,
                     save_path=f'log_plots/Module{module_num}_brdf_3d_log_viridis.png') 