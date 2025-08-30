#utility functions for converting measured photon counts into BRDF
#all input distances in meters, all input angles in degrees
import os
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import pickle

#script to plot the brdf for a single module and save the data to a pkl file

#load calibration data
utils_path = './utils/reference_data'

#load BRDF measurements
white_folder_main = './spectralon_reference'
meas_folder = './Results'
dark_folder = './dark_ref/'
ellipse_dir = './ellipse'

# Solid angle calculation and importing calibration data
#integration range chosen due to eye visibility and saturation
int_range = [320, 800]
# Solid angle as viewed by sensor
sensor_divergence = 0.72
solidangle = np.pi*np.tan(np.deg2rad(sensor_divergence))**2

calfile = os.path.join(utils_path, '20190816_calibration.txt')
caldata = np.genfromtxt(calfile, delimiter="\t", skip_header=9)

# Load spectralon reflectance data
spectralon_file = os.path.join(utils_path, 'spectralon_reflectance.csv')
spectralon = pd.read_csv(spectralon_file).to_numpy()

# Interpolate spectralon reflectance data
spectralon_interp = interp1d(spectralon[:, 0], spectralon[:, 1] / 100, kind='linear', bounds_error=False, fill_value="extrapolate")

caldata[:,1] = caldata[:,1]/1E6 #convert to Ws/(cm2*np*(cts/s))
cal_time = 0.018225 #from file; in seconds

# apparent sample area
def app_area(theta):
    sensor_diameter = 11E-3
    sensor_dist = 0.28
    return (sensor_diameter/2
            + np.tan(np.deg2rad(sensor_divergence)*sensor_dist))**2 * np.pi/np.cos(np.deg2rad(theta))

# formula at sensor angle theta_r or alpha
def spec_flux(raw, raw_time, dark, dark_time, cal, cal_time):
    sensor_aperture_cm2 = 0.95 # from file
    flux = (raw[:, 1] / raw_time
            - np.interp(raw[:, 0], dark[:, 0], dark[:, 1]) / dark_time) \
           * np.interp(raw[:, 0], cal[:, 0], cal[:, 1]) / cal_time \
           * sensor_aperture_cm2
    return np.column_stack((raw[:, 0], flux))

def wavelengt_integration(wavelengths, spectral_values, lambda_min, lambda_max):
    # integration range
    int_range_logical = (wavelengths >= lambda_min) & (wavelengths <= lambda_max)
    # numerical integration using trapezoidal rule
    total = np.trapz(spectral_values[int_range_logical], wavelengths[int_range_logical])
    return total

def filter_points_around_point(value, theta, phi, theta_spec, phi_spec, epsilon_spec):
    cos_delta_theta_dense = (
        np.sin(theta_spec) * np.sin(theta) * np.cos(phi_spec - phi) +
        np.cos(theta_spec) * np.cos(theta)
    )
    delta_theta_dense = np.arccos(cos_delta_theta_dense)
    return value[delta_theta_dense <= epsilon_spec]

# Load the single dark reference file
dark_file_path = os.path.join(dark_folder, os.listdir(dark_folder)[0])
df = pd.read_csv(dark_file_path, encoding='latin1', sep=',')

# Strip column names in case of extra spaces
df.columns = df.columns.str.strip()

# Check and extract needed columns
expected_cols = ['Wavelengths', 'Counts']
if all(col in df.columns for col in expected_cols):
    dark_mean = df[expected_cols].copy()
    dark_mean.rename(columns={'Counts': 'Mean_Counts'}, inplace=True)
else:
    raise ValueError(f"Missing expected columns in dark file: {df.columns}")

# Convert to array for later processing
dark = dark_mean.to_numpy()
dark_time = 0.1

# Calculate white reference irradiance around the sample normal
theta_normal = 0
phi_normal = 0
epsilon_normal = np.radians(10)

def process_folder(white_meas_folder, dark, dark_time, caldata, cal_time, spectralon, int_range, theta_normal, phi_normal, epsilon_normal):
    white_irradiance_ref = []
    white_spec_irradiance_ref = []
    theta_ref = []
    phi_ref = []
    
    if not os.path.exists(white_meas_folder):
        print(f"Warning: White measurement folder {white_meas_folder} does not exist")
        return None, None
        
    for white_file in os.listdir(white_meas_folder):
        if white_file.startswith('.'):  # Skip hidden files
            continue
            
        try:
            full_path = os.path.join(white_meas_folder, white_file)
            raw = pd.read_csv(full_path, encoding='latin1')
            if raw.empty:
                print(f"Warning: Empty white reference file {white_file}")
                continue
            raw = raw.to_numpy()
            
            # Split the path to get the filename
            white_file = white_file.split('/')[-1]
            # Split the filename by underscores
            white_parts = white_file.split('_')

            theta_i = int(white_parts[1])
            phi_i = int(white_parts[3])
            theta_r = int(white_parts[5])
            phi_r = int(white_parts[7])
            white_int_time = int(white_parts[9].split('.')[0])

            raw_time = white_int_time / 1E6 # in seconds, from my measurements, can be also dynamically changed based on the data

            # Adjust integration range for raw, dark, cal data and calculate spectralon reflectance values
            raw = raw[(raw[:, 0] >= int_range[0]) & (raw[:, 0] <= int_range[1])]
            dark = dark[(dark[:, 0] >= int_range[0]) & (dark[:, 0] <= int_range[1])]
            caldata = caldata[(caldata[:, 0] >= int_range[0]) & (caldata[:, 0] <= int_range[1])]
            
            if len(raw) == 0 or len(dark) == 0 or len(caldata) == 0:
                print(f"Warning: No data in integration range for white reference file {white_file}")
                continue
                
            wavelenghts = raw[:, 0]
            spectralon_interpolated_values = spectralon_interp(wavelenghts)
            
            if np.any(np.isnan(spectralon_interpolated_values)):
                print(f"Warning: NaN values in spectralon interpolation for {white_file}")
                continue
                
            spectralon = np.column_stack((wavelenghts, spectralon_interpolated_values))
            
            white_spectral_flux = spec_flux(raw, raw_time, dark, dark_time, caldata, cal_time)
            if len(white_spectral_flux) == 0:
                print(f"Warning: Empty spectral flux for white reference file {white_file}")
                continue

            white_spec_irr = (white_spectral_flux[:, 1] / (app_area(theta_i)) * (np.pi / solidangle)) # from the white measurement
            white_spec_irradiance = white_spec_irr/spectralon_interpolated_values

            if np.any(np.isnan(white_spec_irradiance)) or np.any(np.isinf(white_spec_irradiance)):
                print(f"Warning: Invalid values in white spectral irradiance for {white_file}")
                continue

            white_irradiance = wavelengt_integration(wavelenghts, white_spec_irradiance, int_range[0], int_range[1])

            # Save theta_r, phi_r, ref
            white_irradiance_ref.append(white_irradiance)
            white_spec_irradiance_ref.append(white_spec_irradiance)
            theta_ref.append(np.radians(theta_r))
            phi_ref.append(np.radians(phi_r))
            
        except Exception as e:
            print(f"Error processing white reference file {white_file}: {str(e)}")
            continue
            
    if not white_irradiance_ref:
        print("Warning: No valid white reference measurements found")
        return None, None
        
    white_irradiance_ref = np.array(white_irradiance_ref)
    white_spec_irradiance_ref = np.array(white_spec_irradiance_ref)
    theta_ref = np.array(theta_ref)
    phi_ref = np.array(phi_ref)

    # Filter out sampling directions around the sample normal with 10 degree radius
    filtered_indices = []
    for i, (theta, phi) in enumerate(zip(theta_ref, phi_ref)):
        cos_delta_theta = (
            np.sin(theta) * np.sin(theta_normal) * np.cos(phi - phi_normal) +
            np.cos(theta) * np.cos(theta_normal)
        )
        delta_theta = np.arccos(np.clip(cos_delta_theta, -1.0, 1.0))
        if delta_theta <= epsilon_normal:
            filtered_indices.append(i)
            
    if not filtered_indices:
        print("Warning: No measurements found within epsilon_normal radius")
        return None, None
        
    white_irradiance_filtered = white_irradiance_ref[filtered_indices]
    white_spec_irradiance_filtered = white_spec_irradiance_ref[filtered_indices]
    
    if len(white_irradiance_filtered) == 0:
        print("Warning: No valid filtered white reference measurements")
        return None, None
        
    white_irradiance_mean = np.nanmean(white_irradiance_filtered)
    white_spec_irradiance_mean = np.nanmean(white_spec_irradiance_filtered, axis=0)
    
    if np.isnan(white_irradiance_mean) or np.any(np.isnan(white_spec_irradiance_mean)):
        print("Warning: NaN values in mean white reference calculations")
        return None, None

    return white_irradiance_mean, white_spec_irradiance_mean

white_irradiance_reference = []
white_spec_irradiance_reference = []

# Process the main folder directly
irr, spec_irr = process_folder(white_folder_main, dark, dark_time, caldata, cal_time, spectralon, int_range, theta_normal, phi_normal, epsilon_normal)
if irr is not None and spec_irr is not None:
    white_irradiance_reference.append(irr)
    white_spec_irradiance_reference.append(spec_irr)

if not white_irradiance_reference:
    raise ValueError("No valid white reference measurements found")

white_irradiance_reference = np.array(white_irradiance_reference)
white_irradiance_reference_mean = np.nanmean(white_irradiance_reference)
white_spec_irradiance_reference = np.array(white_spec_irradiance_reference)
white_spec_irradiance_reference_mean = np.nanmean(white_spec_irradiance_reference, axis=0)

if np.isnan(white_irradiance_reference_mean) or np.any(np.isnan(white_spec_irradiance_reference_mean)):
    raise ValueError("NaN values in final white reference calculations")

def calculate_brdf(meas_subfolder, df, dark, dark_time, caldata, cal_time,
        int_range, white_irradiance_reference_mean,
        white_spec_irradiance_reference_mean):
    results = []
    for meas_file in os.listdir(meas_subfolder):
        if meas_file.startswith('.'):  # Skip hidden files
            continue
            
        try:
            full_path = os.path.join(meas_subfolder, meas_file)
            raw = pd.read_csv(full_path, encoding='latin1')
            if raw.empty:
                print(f"Warning: Empty measurement file {meas_file}")
                continue
            raw = raw.to_numpy()

            # Split the path to get the filename
            meas_file = meas_file.split('/')[-1]
            # Split the filename by underscores
            filename_parts = meas_file.split('_')

            ref1 = int(filename_parts[11].split('.')[0])
            int_time = int(filename_parts[9].split('.')[0])

            raw_time = int_time / 1E6
            # Adjust integration range for raw, dark and cal data
            raw = raw[(raw[:, 0] >= int_range[0]) & (raw[:, 0] <= int_range[1])]
            dark = dark[(dark[:, 0] >= int_range[0]) & (dark[:, 0] <= int_range[1])]
            caldata = caldata[(caldata[:, 0] >= int_range[0]) & (caldata[:, 0] <= int_range[1])]
            
            if len(raw) == 0 or len(dark) == 0 or len(caldata) == 0:
                print(f"Warning: No data in integration range for {meas_file}")
                continue
                
            wavelenghts = raw[:, 0]

            # Filter the DataFrame for matching ref
            matching_rows = df[df['ref'] == ref1]
            
            #if matching_rows.empty:
            #    print(f"Warning: No matching reference found for ref1={ref1}")
            #    continue

            # Save and print the matching rows
            for index, row in matching_rows.iterrows():
                theta_i = row['theta_i']
                phi_i = row['phi_i']
                theta_r = row['theta_r']
                phi_r = row['phi_r']
                ellipse = row['ellipse']

                spectral_flux = spec_flux(raw, raw_time, dark, dark_time, caldata, cal_time)
                if len(spectral_flux) == 0:
                    print(f"Warning: Empty spectral flux for {meas_file}")
                    continue
                    
                radiant_flux = wavelengt_integration(wavelenghts, spectral_flux[:, 1], int_range[0], int_range[1])

                spec_rad = (spectral_flux[:, 1]/solidangle/np.cos(np.deg2rad(theta_r))
                     / (app_area(theta_r)*ellipse))

                spec_radiance = np.column_stack((wavelenghts, spec_rad))
                radiance = wavelengt_integration(wavelenghts, spec_radiance[:, 1], int_range[0], int_range[1])

                spec_irr = (spectral_flux[:, 1] / (app_area(theta_i)) * (np.pi / solidangle))
                spec_irradiance = np.column_stack((wavelenghts, spec_irr))
                
                if white_spec_irradiance_reference_mean is None or np.all(white_spec_irradiance_reference_mean == 0):
                    print(f"Warning: Invalid white reference for {meas_file}")
                    continue
                    
                spec_brdf = spec_radiance[:, 1] / (white_spec_irradiance_reference_mean * np.cos(np.deg2rad(theta_i)))

                # Total BRDF
                total_brdf = radiance / (white_irradiance_reference_mean * np.cos(np.deg2rad(theta_i)))
                
                results.append([theta_i, phi_i, theta_r, phi_r, ref1, row['ref'],
                    ellipse, radiant_flux, radiance,
                    white_irradiance_reference_mean, total_brdf, wavelenghts,
                    spec_brdf])

        except Exception as e:
            print(f"Error processing measurement file {meas_file}: {str(e)}")
            continue

    results_df = pd.DataFrame(results, columns=['theta_i', 'phi_i', 'theta_r',
        'phi_r', 'ref1', 'ref2', 'ellipse', 'radiant_flux', 'radiance',
        'white_irradiance_reference_mean', 'total_brdf', 'wavelength',
        'spec_brdf'])
    
    return results_df

def plot_brdf_2d(results_df, clevels=None, save_path=None):
    """
    Create a 2D visualization of BRDF data
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        DataFrame containing BRDF measurements
    clevels : array-like, optional
        Contour levels for the plot. If None, will be automatically determined
    save_path : str, optional
        If provided, save the plot to this path
    """
    theta_r = np.radians(results_df['theta_r'])
    phi_r = np.radians(results_df['phi_r'])
    brdf = results_df['total_brdf']
    theta_i = np.radians(results_df['theta_i'].iloc[0])
    phi_i = np.radians(results_df['phi_i'].iloc[0])

    # Convert spherical to Cartesian coordinates
    x = np.sin(theta_r) * np.cos(phi_r)
    y = np.sin(theta_r) * np.sin(phi_r)

    # Plot the incident vector
    x_i = np.sin(theta_i) * np.cos(phi_i)
    y_i = np.sin(theta_i) * np.sin(phi_i)

    # Create a 2D surface plot
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if clevels is None:
        # Automatically determine levels using linear spacing
        vmin = max(brdf.min(), 1e-3)  # Avoid zero or negative values
        vmax = brdf.max()
        clevels = np.linspace(vmin, vmax, 10)
    
    contour = ax.tricontourf(x, y, brdf, cmap='coolwarm',
                            levels=clevels, extend='both')
    cbar = plt.colorbar(contour, label='BRDF [1/sr]', pad=0.05)
    cbar.ax.yaxis.labelpad = 10
    
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
    ax.set_xlabel('x', labelpad=10)
    ax.set_ylabel('y', labelpad=10)
    ax.set_title(f"BRDF at $\\theta_i$={np.degrees(theta_i):.0f}°, $\\phi_i$={np.degrees(phi_i):.0f}°")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_brdf_3d(results_df, save_path=None):
    """
    Create a 3D visualization of BRDF data
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        DataFrame containing BRDF measurements
    save_path : str, optional
        If provided, save the plot to this path
    """
    theta_r = np.radians(results_df['theta_r'])
    phi_r = np.radians(results_df['phi_r'])
    brdf = results_df['total_brdf']
    theta_i = np.radians(results_df['theta_i'].iloc[0])
    phi_i = np.radians(results_df['phi_i'].iloc[0])

    # Convert spherical to Cartesian coordinates
    x = np.sin(theta_r) * np.cos(phi_r)
    y = np.sin(theta_r) * np.sin(phi_r)
    z = brdf

    # Create a 3D plot with larger figure size and adjusted margins
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Create a 3D surface plot with linear scaling
    surf = ax.plot_trisurf(x, y, z, cmap='coolwarm', linewidth=0.2)
    cbar = plt.colorbar(surf, label='BRDF [1/sr]', pad=0.05)
    cbar.ax.yaxis.labelpad = 10

    # Set plot limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xlabel('x', labelpad=10)
    ax.set_ylabel('y', labelpad=10)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('z', labelpad=30)  # Further increased padding for z-label

    # Adjust the aspect ratio
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.0, 1.0, 0.8, 1]))

    # Set the view angle
    ax.view_init(30, 45)

    # Customize ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.zaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

    # Customize panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')

    # Set title
    plt.title(f"BRDF at $\\theta_i$={np.degrees(theta_i):.0f}°, $\\phi_i$={np.degrees(phi_i):.0f}°")
    
    # tight layout 
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

# --- USER SETTINGS ---

module_to_plot = "Module11"  # Change this to plot different modules
theta_i_to_plot = 45        # Change this to plot different incident angles (0, 15, 30, 45, or 60)
# ---------------------

# Process data
print(f"\nProcessing {module_to_plot} at θᵢ={theta_i_to_plot}°")

# Get the module folder path
module_folder = os.path.join(meas_folder, module_to_plot)
if not os.path.exists(module_folder):
    print(f"Error: {module_folder} not found")
    exit(1)

# Get all measurement files for the specified theta_i
measurement_files = [f for f in os.listdir(module_folder) 
                    if f.startswith(f'ti_{theta_i_to_plot}_') and f.endswith('.csv')]

if not measurement_files:
    print(f"No measurements found for θᵢ={theta_i_to_plot}°")
    exit(1)

print(f"Found {len(measurement_files)} measurement files")

# Create sampling directions from the measurement files
sampling_data = []
for f in measurement_files:
    parts = f.split('_')
    theta_r = float(parts[5])  # tr_2
    phi_r = float(parts[7])    # pr_103
    ref = int(parts[-1].split('.')[0])  # ref_196.csv
    sampling_data.append({
        'theta_i': theta_i_to_plot,
        'phi_i': 0.0,  # Always 0
        'theta_r': theta_r,
        'phi_r': phi_r,
        'ref': ref,
        'beta_x': 0,
        'rel_sensor_angle': 0,
        'ellipse': 1.0
    })

df = pd.DataFrame(sampling_data)

# Calculate BRDF
results_df = calculate_brdf(module_folder, df, dark, dark_time,
        caldata, cal_time, int_range,
        white_irradiance_reference_mean,
        white_spec_irradiance_reference_mean)

if results_df.empty:
    print("Error: No valid BRDF measurements found")
    exit(1)

# Create output directory for plots
os.makedirs('brdf_plots', exist_ok=True)

# Create and save plots
print(f"\nCreating plots for {module_to_plot} at θᵢ={theta_i_to_plot}°")
plot_brdf_2d(results_df, save_path=f'brdf_plots/{module_to_plot}_theta{theta_i_to_plot}_phi0_2d.png')
plot_brdf_3d(results_df, save_path=f'brdf_plots/{module_to_plot}_theta{theta_i_to_plot}_phi0_3d.png')

# Save the data
with open(f'brdf_plots/{module_to_plot}_theta{theta_i_to_plot}_phi0.pkl', 'wb') as f:
    pickle.dump(results_df, f)

print(f"Plots and data saved in brdf_plots/")
