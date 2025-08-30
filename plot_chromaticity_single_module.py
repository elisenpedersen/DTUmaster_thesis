import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import colour.plotting as cplt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from scipy.signal import find_peaks

# ========== USER SETTINGS ==========
MODULE_NUM = 9  # Change this to your module number
THETA_I = 30    # Change this to your theta_i
PHI_I = 0       # Change this to your phi_i
# ===================================

# --- Color matching and illuminant data ---
fcmf = './utils/reference_data/CIE2degCMF1931forMATLAB.txt'
cmf = np.genfromtxt(fcmf)
fD65 = './utils/reference_data/CIED65.txt'
D65 = np.genfromtxt(fD65, delimiter='\t')

# Color conversion helpers
def spec2XYZ(wvl, spec, cmf, illum):
    Iinterp = np.interp(cmf[:,0], illum[:,0], illum[:,1])
    if len(np.shape(spec)) > 1:
        Rinterp = [np.interp(cmf[:,0], wvl, s) for s in spec ]
    else:
        Rinterp = np.interp(cmf[:,0], wvl, spec)
    L = Iinterp * Rinterp
    N = np.trapz(Iinterp * cmf[:,2], x=cmf[:,0])
    X = np.trapz(L * cmf[:,1], x=cmf[:,0]) / N
    Y = np.trapz(L * cmf[:,2], x=cmf[:,0]) / N
    Z = np.trapz(L * cmf[:,3], x=cmf[:,0]) / N
    XYZ = np.array([X, Y, Z])
    return XYZ.T

# File path
pkl_path = f'brdf_plots/Module{MODULE_NUM}/Module{MODULE_NUM}_theta{THETA_I}_phi{PHI_I}.pkl'

# Load data
with open(pkl_path, 'rb') as f:
    df = pickle.load(f)

# If chromaticity columns are not present, compute them
if not (('x' in df.columns) and ('y' in df.columns)):
    # If xyz is not present, compute it from wavelength and spec_brdf
    if 'xyz' not in df.columns:
        if 'wavelength' in df.columns and 'spec_brdf' in df.columns:
            df['xyz'] = df.apply(lambda r: spec2XYZ(r['wavelength'], r['spec_brdf'], cmf, D65), axis=1)
        else:
            raise ValueError('No chromaticity or spectral columns found in data!')
    df['x'] = df['xyz'].apply(lambda xyz: xyz[0]/np.sum(xyz))
    df['y'] = df['xyz'].apply(lambda xyz: xyz[1]/np.sum(xyz))

# ========== PLOTTING =============
plt.style.use('seaborn-v0_8-white')
fig, ax = plt.subplots(figsize=(10, 9), facecolor='white')
ax.set_facecolor('white')
# Add a solid white rectangle to the figure background
# Add a larger solid white rectangle to the figure background
fig.patches.extend([
    mpatches.Rectangle(
        (-0.15, -0.05), 1.2, 1.2,  # Increased size by extending beyond figure bounds
        transform=fig.transFigure, zorder=-1000,
        facecolor='white', edgecolor='none', linewidth=0)
])

# Plot the CIE 1931 chromaticity diagram background
cplt.diagrams.plot_chromaticity_diagram_CIE1931(axes=ax, show=False)

# Overlay your BRDF data
norm = LogNorm(vmin=max(df.total_brdf.min(), 1e-6), vmax=df.total_brdf.max())
sc = ax.scatter(df['x'], df['y'], c=df['total_brdf'], cmap='plasma', norm=norm,
                edgecolor='black', s=120, alpha=0.9, linewidth=1.2, zorder=3, label='BRDF points')

# Colorbar
cbar = plt.colorbar(sc, ax=ax, pad=0.03, aspect=40)
cbar.set_label('BRDF [1/sr]', fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=15)
cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1e}"))
# Make colorbar background fully white
cbar.ax.set_facecolor('white')
cbar.ax.patch.set_facecolor('white')

# Title and labels
ax.set_title(f"Chromaticity Diagram with BRDF\nModule {MODULE_NUM}, $\\theta_i$={THETA_I}°, $\\phi_i$={PHI_I}°", fontsize=22, fontweight='bold', pad=20,
    bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', boxstyle='round,pad=0.5'))
ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('y', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=15)

# Remove axis ticks for a cleaner look
ax.set_xticks([])
ax.set_yticks([])

# Add legend
ax.legend(loc='upper right', fontsize=14, frameon=True, facecolor='white')

# Force all backgrounds to white
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
cbar.ax.set_facecolor('white')
cbar.ax.patch.set_facecolor('white')

# Save and show
out_path = f'chromaticity_module{MODULE_NUM}_theta{THETA_I}_phi{PHI_I}.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved plot to {out_path}")

# Create alternative version with black line and scatter points
plt.figure(figsize=(10, 9), facecolor='white')
ax2 = plt.gca()
ax2.set_facecolor('white')

# Add white background rectangle
fig2 = plt.gcf()
fig2.patches.extend([
    mpatches.Rectangle(
        (-0.15, -0.05), 1.2, 1.2,
        transform=fig2.transFigure, zorder=-1000,
        facecolor='white', edgecolor='none', linewidth=0)
])

# Plot the CIE 1931 chromaticity diagram background
cplt.diagrams.plot_chromaticity_diagram_CIE1931(axes=ax2, show=False)

# Sort points by x coordinate for better line connection
sorted_df = df.sort_values('x')

# Plot black line and scatter points
ax2.plot(sorted_df['x'], sorted_df['y'], 'k-', linewidth=1, zorder=2, label='BRDF path')
ax2.scatter(sorted_df['x'], sorted_df['y'], color='black', s=2, zorder=3, label='BRDF points')

# Title and labels
ax2.set_title(f"Chromaticity Diagram with BRDF\nModule {MODULE_NUM}, $\\theta_i$={THETA_I}°, $\\phi_i$={PHI_I}°", 
              fontsize=22, fontweight='bold', pad=20,
              bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', boxstyle='round,pad=0.5'))
ax2.set_xlabel('x', fontsize=18)
ax2.set_ylabel('y', fontsize=18)
ax2.tick_params(axis='both', which='major', labelsize=15)

# Remove axis ticks for a cleaner look
ax2.set_xticks([])
ax2.set_yticks([])

# Add legend
ax2.legend(loc='upper right', fontsize=14, frameon=True, facecolor='white')

# Force all backgrounds to white
fig2.patch.set_facecolor('white')
ax2.set_facecolor('white')

# Save the alternative version
out_path_alt = f'chromaticity_module{MODULE_NUM}_theta{THETA_I}_phi{PHI_I}_black.png'
plt.savefig(out_path_alt, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved alternative plot to {out_path_alt}")

# Create version without specular peak
plt.figure(figsize=(10, 9), facecolor='white')
ax3 = plt.gca()
ax3.set_facecolor('white')

# Add white background rectangle
fig3 = plt.gcf()
fig3.patches.extend([
    mpatches.Rectangle(
        (-0.15, -0.05), 1.2, 1.2,
        transform=fig3.transFigure, zorder=-1000,
        facecolor='white', edgecolor='none', linewidth=0)
])

# Plot the CIE 1931 chromaticity diagram background
cplt.diagrams.plot_chromaticity_diagram_CIE1931(axes=ax3, show=False)

# Identify non-specular points using the same method as in specular_deviation_analysis.py
peaks, _ = find_peaks(df['total_brdf'], height=np.mean(df['total_brdf']))
non_peak_mask = ~np.isin(np.arange(len(df['total_brdf'])), peaks)
mean_brdf = np.mean(df['total_brdf'][non_peak_mask])
std_brdf = np.std(df['total_brdf'][non_peak_mask])
specular_threshold = mean_brdf + 2 * std_brdf
non_specular_mask = df['total_brdf'] < specular_threshold
filtered_df = df[non_specular_mask]

# Sort points by x coordinate for better line connection
sorted_filtered_df = filtered_df.sort_values('x')

# Plot black line and scatter points
ax3.plot(sorted_filtered_df['x'], sorted_filtered_df['y'], 'k-', linewidth=1, zorder=2, label='BRDF path (no specular)')
ax3.scatter(sorted_filtered_df['x'], sorted_filtered_df['y'], color='black', s=2, zorder=3, label='BRDF points')

# Title and labels
ax3.set_title(f"Chromaticity Diagram with BRDF (No Specular)\nModule {MODULE_NUM}, $\\theta_i$={THETA_I}°, $\\phi_i$={PHI_I}°", 
              fontsize=22, fontweight='bold', pad=20,
              bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', boxstyle='round,pad=0.5'))
ax3.set_xlabel('x', fontsize=18)
ax3.set_ylabel('y', fontsize=18)
ax3.tick_params(axis='both', which='major', labelsize=15)

# Remove axis ticks for a cleaner look
ax3.set_xticks([])
ax3.set_yticks([])

# Add legend
ax3.legend(loc='upper right', fontsize=14, frameon=True, facecolor='white')

# Force all backgrounds to white
fig3.patch.set_facecolor('white')
ax3.set_facecolor('white')

# Save the no-specular version
out_path_no_specular = f'chromaticity_module{MODULE_NUM}_theta{THETA_I}_phi{PHI_I}_no_specular.png'
plt.savefig(out_path_no_specular, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved no-specular plot to {out_path_no_specular}")
# plt.show()  # Do not open automatically 