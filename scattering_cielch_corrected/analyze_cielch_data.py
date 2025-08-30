import pandas as pd
import numpy as np
import os

def analyze_module_data(module_id, data_file):
    """Analyze CIE LCh data for a specific module."""
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return None
    
    df = pd.read_csv(data_file)
    
    # Filter by module
    module_data = df[df['Module'] == module_id]
    
    if len(module_data) == 0:
        print(f"No data found for module {module_id}")
        return None
    
    results = {}
    
    # Overall statistics
    results['total_points'] = len(module_data)
    results['mean_L'] = module_data['L'].mean()
    results['std_L'] = module_data['L'].std()
    results['mean_a'] = module_data['a'].mean()
    results['std_a'] = module_data['a'].std()
    results['mean_b'] = module_data['b'].mean()
    results['std_b'] = module_data['b'].std()
    results['mean_hue'] = module_data['Hue'].mean()
    results['std_hue'] = module_data['Hue'].std()
    results['mean_chroma'] = module_data['Chroma'].mean()
    results['std_chroma'] = module_data['Chroma'].std()
    
    # Angular zone analysis
    zone_columns = ['zone_specular', 'zone_forward_scatter', 'zone_backscatter', 'zone_residual']
    for zone in zone_columns:
        if zone in module_data.columns:
            zone_data = module_data[module_data[zone] == True]
            if len(zone_data) > 0:
                results[f'{zone}_count'] = len(zone_data)
                results[f'{zone}_mean_L'] = zone_data['L'].mean()
                results[f'{zone}_mean_hue'] = zone_data['Hue'].mean()
                results[f'{zone}_mean_chroma'] = zone_data['Chroma'].mean()
                results[f'{zone}_mean_brdf'] = zone_data['total_brdf'].mean()
    
    # Per-incident-angle analysis
    angles = [0, 15, 30, 45, 60]
    angle_results = {}
    
    for angle in angles:
        angle_data = module_data[module_data['Theta_i'] == angle]
        if len(angle_data) > 0:
            angle_results[angle] = {
                'count': len(angle_data),
                'mean_hue': angle_data['Hue'].mean(),
                'std_hue': angle_data['Hue'].std(),
                'mean_chroma': angle_data['Chroma'].mean(),
                'std_chroma': angle_data['Chroma'].std(),
                'mean_L': angle_data['L'].mean(),
                'std_L': angle_data['L'].std()
            }
    
    results['angle_analysis'] = angle_results
    
    return results

def analyze_weighted_zones(module_id, zone_file):
    """Analyze weighted zone distribution for a module."""
    if not os.path.exists(zone_file):
        print(f"Zone file not found: {zone_file}")
        return None
    
    df = pd.read_csv(zone_file)
    module_data = df[df['Module'] == module_id]
    
    if len(module_data) == 0:
        return None
    
    results = {}
    
    # Overall zone distribution
    overall = module_data[module_data['Theta_i'] == 'All']
    if len(overall) > 0:
        for _, row in overall.iterrows():
            zone = row['Zone']
            results[f'{zone}_ratio'] = row['Ratio']
            results[f'{zone}_percent'] = row['Ratio_Percent']
            results[f'{zone}_points'] = row['Point_Count']
    
    # Per-angle zone distribution
    angles = [0, 15, 30, 45, 60]
    angle_zones = {}
    
    for angle in angles:
        angle_data = module_data[module_data['Theta_i'] == angle]
        if len(angle_data) > 0:
            angle_zones[angle] = {}
            for _, row in angle_data.iterrows():
                zone = row['Zone']
                angle_zones[angle][zone] = {
                    'ratio': row['Ratio'],
                    'percent': row['Ratio_Percent'],
                    'points': row['Point_Count']
                }
    
    results['angle_zones'] = angle_zones
    
    return results

def main():
    """Main analysis function."""
    base_dir = "."
    
    # Define modules to analyze
    modules = {
        18: "BrownC (uniform)",
        20: "BrownBaC (BaSO4 filler)", 
        25: "GreenC (uniform)",
        27: "GreenBaC (BaSO4 filler)"
    }
    
    print("=== CIE LCh° Polar Plots Data Analysis ===\n")
    
    for module_id, description in modules.items():
        print(f"--- {description} (Module {module_id}) ---")
        
        # Analyze CIE LCh data
        data_file = os.path.join(base_dir, f"m{module_id}_cielch_data_corrected.csv")
        cielch_results = analyze_module_data(module_id, data_file)
        
        if cielch_results:
            print(f"Total data points: {cielch_results['total_points']}")
            print(f"Overall L*: {cielch_results['mean_L']:.1f} ± {cielch_results['std_L']:.1f}")
            print(f"Overall a*: {cielch_results['mean_a']:.1f} ± {cielch_results['std_a']:.1f}")
            print(f"Overall b*: {cielch_results['mean_b']:.1f} ± {cielch_results['std_b']:.1f}")
            print(f"Overall Hue: {cielch_results['mean_hue']:.1f}° ± {cielch_results['std_hue']:.1f}°")
            print(f"Overall Chroma: {cielch_results['mean_chroma']:.1f} ± {cielch_results['std_chroma']:.1f}")
            
            # Angular zone analysis
            if 'zone_specular_count' in cielch_results:
                print(f"Specular points: {cielch_results['zone_specular_count']}")
                print(f"  - Mean L*: {cielch_results['zone_specular_mean_L']:.1f}")
                print(f"  - Mean Hue: {cielch_results['zone_specular_mean_hue']:.1f}°")
                print(f"  - Mean Chroma: {cielch_results['zone_specular_mean_chroma']:.1f}")
            
            if 'zone_forward_scatter_count' in cielch_results:
                print(f"Forward scatter points: {cielch_results['zone_forward_scatter_count']}")
                print(f"  - Mean L*: {cielch_results['zone_forward_scatter_mean_L']:.1f}")
                print(f"  - Mean Hue: {cielch_results['zone_forward_scatter_mean_hue']:.1f}°")
                print(f"  - Mean Chroma: {cielch_results['zone_forward_scatter_mean_chroma']:.1f}")
            
            if 'zone_backscatter_count' in cielch_results:
                print(f"Backscatter points: {cielch_results['zone_backscatter_count']}")
                print(f"  - Mean L*: {cielch_results['zone_backscatter_mean_L']:.1f}")
                print(f"  - Mean Hue: {cielch_results['zone_backscatter_mean_hue']:.1f}°")
                print(f"  - Mean Chroma: {cielch_results['zone_backscatter_mean_chroma']:.1f}")
            
            # Per-angle analysis
            print("\nPer-incident-angle analysis:")
            for angle, angle_data in cielch_results['angle_analysis'].items():
                print(f"  θᵢ = {angle}°: {angle_data['count']} points")
                print(f"    Hue: {angle_data['mean_hue']:.1f}° ± {angle_data['std_hue']:.1f}°")
                print(f"    Chroma: {angle_data['mean_chroma']:.1f} ± {angle_data['std_chroma']:.1f}")
                print(f"    L*: {angle_data['mean_L']:.1f} ± {angle_data['std_L']:.1f}")
        
        # Analyze weighted zones
        zone_file = os.path.join(base_dir, f"m{module_id}_weighted_zone_analysis.csv")
        zone_results = analyze_weighted_zones(module_id, zone_file)
        
        if zone_results:
            print(f"\nWeighted zone distribution:")
            if 'specular_ratio' in zone_results:
                print(f"  Specular: {zone_results['specular_percent']:.1f}% ({zone_results['specular_points']} points)")
            if 'forward_scatter_ratio' in zone_results:
                print(f"  Forward scatter: {zone_results['forward_scatter_percent']:.1f}% ({zone_results['forward_scatter_points']} points)")
            if 'backscatter_ratio' in zone_results:
                print(f"  Backscatter: {zone_results['backscatter_percent']:.1f}% ({zone_results['backscatter_points']} points)")
            if 'residual_ratio' in zone_results:
                print(f"  Residual: {zone_results['residual_percent']:.1f}% ({zone_results['residual_points']} points)")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
