#!/usr/bin/env python3
"""
Comprehensive BRDF analysis framework with proper solid angle normalization.
Implements the correct formula: R_z = (sum of weighted BRDF in zone) / (total weighted BRDF)
where each point is weighted by its solid angle contribution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from colour import SpectralDistribution, sd_to_XYZ, XYZ_to_Lab, MSDS_CMFS, SDS_ILLUMINANTS

# --- Configuration ---
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define angular zones for classification
ANGULAR_ZONES = {
    'specular': {'deviation_range': (0, 15), 'color': 'white'},
    'forward_scatter': {'deviation_range': (15, 45), 'color': 'gray'},
    'backscatter': {'deviation_range': (45, 90), 'color': 'black'},
    'residual': {'deviation_range': (90, 180), 'color': 'red'}
}

# Define module groups for analysis (matching your comprehensive module database exactly)
MODULE_GROUPS = {
    # Reference module
    'Reference': {
        'modules': [1],
        'short_names': ['REF'],
        'pigment': 'None',
        'filler_type': 'None',
        'filler_content': [0],
        'category': 'Reference'
    },
    
    # Non-pigmented filler groups
    'BaSO4_Fillers': {
        'modules': [2, 3, 4],
        'short_names': ['B5', 'B10', 'B20'],
        'pigment': 'None',
        'filler_type': 'BaSO4',
        'filler_content': [5, 10, 20],
        'category': 'BaSO4_Fillers'
    },
    
    'Glass_Fillers_Small': {
        'modules': [5, 6, 7, 8],
        'short_names': ['Gl2.1S5', 'Gl2.1S20', 'Gl1.5S5', 'Gl1.5S20'],
        'pigment': 'None',
        'filler_type': 'Glass',
        'filler_content': [5, 20, 5, 20],
        'category': 'Glass_Fillers_Small'
    },
    
    'Glass_Fillers_Large': {
        'modules': [14, 15, 16, 17],
        'short_names': ['Gl2.1L5', 'Gl2.1L20', 'Gl1.5L5', 'Gl1.5L20'],
        'pigment': 'None',
        'filler_type': 'Glass',
        'filler_content': [5, 20, 5, 20],
        'category': 'Glass_Fillers_Large'
    },
    
    # Pigmented groups
    'Blue_Pigment': {
        'modules': [9, 10, 11, 12, 13],
        'short_names': ['Blue2.1S', 'Blue2.1L', 'Blue1.5S', 'Blue1.5L', 'BlueBa'],
        'pigment': 'MB Blue',
        'filler_type': 'Mixed',
        'filler_content': [9, 9, 9, 9, 9],
        'category': 'Blue_Pigment'
    },
    
    'Brown_Pigment': {
        'modules': [18, 20, 22],
        'short_names': ['BrownC', 'BrownBaC', 'BrownGlC'],
        'pigment': 'MB Brown',
        'filler_type': 'Mixed',
        'filler_content': [5, 20, 20],
        'category': 'Brown_Pigment'
    },
    
    'Green_Pigment': {
        'modules': [25, 27, 28],
        'short_names': ['GreenC', 'GreenBaC', 'GreenGlC'],
        'pigment': 'MB Green',
        'filler_type': 'Mixed',
        'filler_content': [7, 20, 20],
        'category': 'Green_Pigment'
    }
}

# Helper function to get module information
def get_module_info(module_id):
    for group_name, group_data in MODULE_GROUPS.items():
        if module_id in group_data['modules']:
            idx = group_data['modules'].index(module_id)
            return {
                'module_id': module_id,
                'group_name': group_name,
                'short_name': group_data['short_names'][idx],
                'pigment': group_data['pigment'],
                'filler_type': group_data['filler_type'],
                'filler_content': group_data['filler_content'][idx] if isinstance(group_data['filler_content'], list) else group_data['filler_content'],
                'category': group_data['category']
            }
    return None

class SolidAngleBRDFAnalyzer:
    """Comprehensive BRDF analyzer with proper solid angle normalization."""
    
    def __init__(self):
        self.angular_zones = ANGULAR_ZONES
        self.angles = [0, 15, 30, 45, 60]  # Incident angles to analyze

    def calculate_angular_deviation_from_specular(self, theta_i, theta_r, phi_r):
        """Calculate angular deviation from specular direction."""
        theta_i_rad = np.radians(theta_i)
        theta_r_rad = np.radians(theta_r)
        phi_r_rad = np.radians(phi_r)
        
        # Correct specular direction vector calculation
        R_specular = np.array([
            -np.sin(theta_i_rad),
            0.0,
            np.cos(theta_i_rad)
        ])
        
        # Measured reflected direction vector
        R_measured = np.array([
            np.sin(theta_r_rad) * np.cos(phi_r_rad),
            np.sin(theta_r_rad) * np.sin(phi_r_rad),
            np.cos(theta_r_rad)
        ])
        
        # Calculate the dot product
        dot_product = np.dot(R_measured, R_specular)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Angle between the two vectors (deviation)
        deviation_rad = np.arccos(dot_product)
        deviation_deg = np.degrees(deviation_rad)
        
        return deviation_deg

    def classify_point_to_zone(self, deviation_deg):
        """Classify a measurement point into a zone based on angular deviation."""
        for zone, config in self.angular_zones.items():
            min_dev, max_dev = config['deviation_range']
            if min_dev <= deviation_deg < max_dev:
                return zone
        return None

    def analyze_single_module(self, module_id):
        """Analyze a single module with proper solid angle normalization."""
        print(f"Analyzing Module {module_id}...")
        
        all_data = []
        
        for angle in self.angles:
            # Load BRDF data from the same path structure as your original framework
            pkl_path = os.path.join('brdf_plots', f'Module{module_id}', f'Module{module_id}_theta{angle}_phi0.pkl')
            if not os.path.exists(pkl_path):
                pkl_path = os.path.join('..', 'brdf_plots', f'Module{module_id}', f'Module{module_id}_theta{angle}_phi0.pkl')
                if not os.path.exists(pkl_path):
                    print(f"Missing: {pkl_path}")
                    continue
            
            try:
                with open(pkl_path, 'rb') as f:
                    df = pickle.load(f)
            except:
                print(f"Failed to load: {pkl_path}")
                continue
            
            wavelengths = np.linspace(400, 700, len(df['spec_brdf'].iloc[0]))
            
            for idx, row in df.iterrows():
                theta_r = row['theta_r']
                phi_r = row.get('phi_r', 0)
                spectrum = np.array(row['spec_brdf'])
                
                # Filter out abnormal BRDF values
                if np.any(spectrum > 2.0):
                    continue
                
                # Convert to Lab color space
                try:
                    std_wl = np.arange(360, 781, 1)
                    spectrum_interp = np.interp(std_wl, wavelengths, spectrum)
                    sd = SpectralDistribution(dict(zip(std_wl, spectrum_interp)))
                    cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
                    illuminant = SDS_ILLUMINANTS['D65']
                    XYZ = sd_to_XYZ(sd, cmfs, illuminant)
                    Lab = XYZ_to_Lab(XYZ / 100)
                except:
                    continue
                
                angular_deviation = self.calculate_angular_deviation_from_specular(angle, theta_r, phi_r)
                zone = self.classify_point_to_zone(angular_deviation)
                
                hue = (np.degrees(np.arctan2(Lab[2], Lab[1])) + 360) % 360
                chroma = np.sqrt(Lab[1]**2 + Lab[2]**2)
                
                data_point = {
                    'Module': module_id,
                    'Theta_i': angle,
                    'Theta_r': theta_r,
                    'Phi_r': phi_r,
                    'Angular_Deviation': angular_deviation,
                    'Zone': zone,
                    'L': Lab[0],
                    'a': Lab[1],
                    'b': Lab[2],
                    'Hue': hue,
                    'Chroma': chroma,
                    'total_brdf': row.get('total_brdf', np.nan)
                }
                
                all_data.append(data_point)
        
        df_result = pd.DataFrame(all_data)
        
        # Calculate properly normalized zone ratios
        zone_ratios = self.calculate_normalized_zone_ratios(df_result)
        
        return {
            'module_id': module_id,
            'module_info': get_module_info(module_id),
            'data': df_result,
            'zone_ratios': zone_ratios
        }

    def calculate_normalized_zone_ratios(self, df):
        """Calculate zone ratios with proper solid angle weighting."""
        zone_ratios = {}
        
        # Group by incident angle for proper solid angle calculation
        for theta_i in df['Theta_i'].unique():
            df_angle = df[df['Theta_i'] == theta_i].copy()
            
            # Calculate solid angle weight for each measurement point
            # w_k = cos(theta_r) * sin(theta_r)
            solid_angle_weights = []
            for _, row in df_angle.iterrows():
                theta_r = row['Theta_r']
                theta_r_rad = np.radians(theta_r)
                
                if theta_r > 90:
                    # Outside reflection hemisphere
                    solid_angle_weight = 0.0
                else:
                    # Inside reflection hemisphere
                    solid_angle_weight = np.sin(theta_r_rad) * np.cos(theta_r_rad)
                
                solid_angle_weights.append(solid_angle_weight)
            
            df_angle['solid_angle_weight'] = solid_angle_weights
            
            # Calculate total weighted BRDF for this incident angle
            total_weighted_brdf_angle = (df_angle['total_brdf'] * df_angle['solid_angle_weight']).sum()
            
            # Aggregate weighted BRDF per zone for this incident angle
            for zone_name in self.angular_zones.keys():
                zone_data = df_angle[df_angle['Zone'] == zone_name]
                
                if len(zone_data) == 0:
                    continue
                
                zone_weighted_brdf = (zone_data['total_brdf'] * zone_data['solid_angle_weight']).sum()
                
                # Initialize zone_ratios entry if it doesn't exist
                if zone_name not in zone_ratios:
                    zone_ratios[zone_name] = {
                        'total_weighted_brdf': 0.0,
                        'total_brdf_sum': 0.0,
                        'point_count': 0,
                        'sum_solid_angle_weights': 0.0
                    }
                
                zone_ratios[zone_name]['total_weighted_brdf'] += zone_weighted_brdf
                zone_ratios[zone_name]['total_brdf_sum'] += zone_data['total_brdf'].sum()
                zone_ratios[zone_name]['point_count'] += len(zone_data)
                zone_ratios[zone_name]['sum_solid_angle_weights'] += zone_data['solid_angle_weight'].sum()

        # Calculate final ratios and average weights across all incident angles
        final_total_weighted_brdf = sum(stats['total_weighted_brdf'] for stats in zone_ratios.values())

        for zone_name, stats in zone_ratios.items():
            ratio = (stats['total_weighted_brdf'] / final_total_weighted_brdf) * 100 if final_total_weighted_brdf > 0 else 0.0
            avg_solid_angle_weight = stats['sum_solid_angle_weights'] / stats['point_count'] if stats['point_count'] > 0 else 0.0
            
            zone_ratios[zone_name]['ratio_percent'] = ratio
            zone_ratios[zone_name]['avg_solid_angle_weight'] = avg_solid_angle_weight
            zone_ratios[zone_name]['raw_brdf_sum'] = stats['total_brdf_sum']

        return zone_ratios

    def analyze_module_group(self, module_ids, group_name):
        """Analyze a group of modules."""
        print(f"Analyzing group: {group_name}")
        
        group_results = {}
        all_group_data = []
        
        for module_id in module_ids:
            module_result = self.analyze_single_module(module_id)
            if module_result:
                group_results[module_id] = module_result
                all_group_data.append(module_result['data'])
        
        # Combine all data
        if all_group_data:
            combined_data = pd.concat(all_group_data, ignore_index=True)
        else:
            combined_data = pd.DataFrame()
        
        return {
            'group_name': group_name,
            'module_ids': module_ids,
            'individual_results': group_results,
            'combined_data': combined_data
        }

    def create_comparison_plots(self, group_results):
        """Create comparison plots for a group, similar to BaSO4_Fillers_comparison.png."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Clean up group name for title (remove underscores and improve readability)
        group_name_clean = group_results["group_name"].replace('_', ' ')
        if group_name_clean == 'BaSO4 Fillers':
            group_name_clean = 'BaSO4 Module Group'
        elif group_name_clean == 'Glass Fillers Small':
            group_name_clean = 'Glass Module Group (Small Particles)'
        elif group_name_clean == 'Glass Fillers Large':
            group_name_clean = 'Glass Module Group (Large Particles)'
        elif group_name_clean == 'Blue Pigment':
            group_name_clean = 'Blue Module Group'
        elif group_name_clean == 'Green Pigment':
            group_name_clean = 'Green Module Group'
        elif group_name_clean == 'Brown Pigment':
            group_name_clean = 'Brown Module Group'
        elif group_name_clean == 'Reference':
            group_name_clean = 'Reference Module Group'
        
        fig.suptitle(f'{group_name_clean} - BRDF Analysis', 
                    fontsize=22, y=0.92)

        zones = list(self.angular_zones.keys())
        module_names = []
        module_ids = group_results['module_ids']

        # Data for plotting
        overall_zone_distribution = {zone: [] for zone in zones}
        specular_values = []
        diffuse_values = []

        for module_id in module_ids:
            result = group_results['individual_results'].get(module_id)
            if result and result['module_info']:
                module_names.append(result['module_info']['short_name'])
                zone_ratios = result['zone_ratios']
                
                # Overall Zone Distribution
                for zone in zones:
                    overall_zone_distribution[zone].append(zone_ratios.get(zone, {}).get('ratio_percent', 0))
                
                # Data for Specular vs Diffuse Scattering plot
                specular_values.append(zone_ratios.get('specular', {}).get('ratio_percent', 0))
                
                # Diffuse Scattering (sum of forward, backscatter, residual)
                diffuse_sum = (
                    zone_ratios.get('forward_scatter', {}).get('ratio_percent', 0) +
                    zone_ratios.get('backscatter', {}).get('ratio_percent', 0) +
                    zone_ratios.get('residual', {}).get('ratio_percent', 0)
                )
                diffuse_values.append(diffuse_sum)

        if not module_names:
            print("No valid module results to plot")
            return

        # Plot 1: Overall Zone Distribution (Stacked Bar Chart)
        bottom = np.zeros(len(module_names))
        for zone_idx, zone in enumerate(zones):
            # Use better colors with dark grey for backscatter
            if zone == 'backscatter':
                color = '#404040'  # Dark grey for better text visibility
            elif zone == 'specular':
                color = 'white'  # White for specular
            elif zone == 'forward_scatter':
                color = '#A0A0A0'  # Medium grey
            elif zone == 'residual':
                color = 'red'  # Red like before
            else:
                color = self.angular_zones[zone]['color']
                
            bars = axes[0].bar(module_names, overall_zone_distribution[zone], 
                               bottom=bottom, label=zone.replace('_', ' ').title(), 
                               color=color, edgecolor='black')
            bottom += np.array(overall_zone_distribution[zone])
            
            # Add value labels on bars with larger font
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0.5:  # Only label if segment is large enough
                    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_y() + height/2.,
                                f'{height:.1f}', ha='center', va='center', color='black', fontsize=10, fontweight='bold')

        axes[0].set_title('Overall Zone Distribution', fontsize=16)
        axes[0].set_ylabel('Percentage of Total Weighted Reflected Light (%)', fontsize=14)
        axes[0].set_xlabel('Module', fontsize=14)
        axes[0].legend(title='Zone', fontsize=12)
        axes[0].tick_params(axis='both', which='major', labelsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 100)

        # Plot 2: Specular vs Diffuse Scattering (Scatter Plot)
        axes[1].scatter(specular_values, diffuse_values, s=120, alpha=0.8, edgecolors='black', linewidth=1.5)
        for i, txt in enumerate(module_names):
            axes[1].annotate(txt, (specular_values[i], diffuse_values[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
        axes[1].set_title('Specular vs Diffuse Scattering', fontsize=16)
        axes[1].set_xlabel('Specular Light Percentage (%)', fontsize=14)
        axes[1].set_ylabel('Diffuse Light Percentage (%)', fontsize=14)
        axes[1].tick_params(axis='both', which='major', labelsize=12)
        axes[1].grid(True, alpha=0.3)
        
        if specular_values and diffuse_values:
            axes[1].set_xlim(min(specular_values)*0.9, max(specular_values)*1.1)
            axes[1].set_ylim(min(diffuse_values)*0.9, max(diffuse_values)*1.1)

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        plt.savefig(os.path.join(OUTPUT_DIR, f'{group_results["group_name"]}_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plots saved for {group_results['group_name']}")

    def run_comprehensive_analysis(self):
        """Run comprehensive analysis for all defined module groups."""
        print("Starting Comprehensive Solid Angle BRDF Analysis...")
        
        # Define all analysis groups to process
        analysis_groups = [
            'Reference',
            'BaSO4_Fillers',
            'Glass_Fillers_Small',
            'Glass_Fillers_Large',
            'Blue_Pigment',
            'Brown_Pigment',
            'Green_Pigment'
        ]
        
        for group_name in analysis_groups:
            if group_name in MODULE_GROUPS:
                print(f"\n{'='*50}")
                print(f"Analyzing {group_name}")
                print(f"{'='*50}")
                
                group_results = self.analyze_module_group(MODULE_GROUPS[group_name]['modules'], group_name)
                if group_results:
                    self.create_comparison_plots(group_results)
                    self.save_group_results(group_results, group_name)
        
        print("\nComprehensive analysis complete!")

    def save_group_results(self, group_results, category):
        """Save detailed results for a group."""
        # Save combined data
        if not group_results['combined_data'].empty:
            combined_data_path = os.path.join(OUTPUT_DIR, f'{category}_combined_data.csv')
            group_results['combined_data'].to_csv(combined_data_path, index=False)
        
        # Save summary statistics
        summary_data = []
        for module_id in group_results['module_ids']:
            module_result = group_results['individual_results'].get(module_id)
            if module_result and module_result['module_info']:
                module_info = module_result['module_info']
                zone_ratios = module_result['zone_ratios']
                
                for zone, stats in zone_ratios.items():
                    summary_data.append({
                        'Module_ID': module_id,
                        'Short_Name': module_info['short_name'],
                        'Pigment': module_info['pigment'],
                        'Filler_Type': module_info['filler_type'],
                        'Filler_Content': module_info['filler_content'],
                        'Category': module_info['category'],
                        'Zone': zone,
                        'Ratio_Percent': stats['ratio_percent'],
                        'Total_Weighted_BRDF': stats['total_weighted_brdf'],
                        'Point_Count': stats['point_count'],
                        'Avg_Solid_Angle_Weight': stats['avg_solid_angle_weight']
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(OUTPUT_DIR, f'{category}_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"Results saved for {category}")

def main():
    """Main function to run the comprehensive analysis."""
    analyzer = SolidAngleBRDFAnalyzer()
    analyzer.run_comprehensive_analysis()

if __name__ == "__main__":
    main()
