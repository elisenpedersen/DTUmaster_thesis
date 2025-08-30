import pandas as pd
import numpy as np

def analyze_blue_modules():
    """Analyze blue modules from scattering lobe data."""
    df = pd.read_csv('scattering_lobe_color_data_corrected.csv')
    
    blue_modules = [1, 9]
    
    for module_id in blue_modules:
        print(f"=== Module {module_id} (Blue) Analysis ===")
        
        module_data = df[df['Module'] == module_id]
        if len(module_data) == 0:
            print(f"No data found for module {module_id}")
            continue
        
        print(f"Total data points: {len(module_data)}")
        
        # Overall statistics
        print(f"Overall L*: {module_data['L'].mean():.1f} ± {module_data['L'].std():.1f}")
        print(f"Overall a*: {module_data['a'].mean():.1f} ± {module_data['a'].std():.1f}")
        print(f"Overall b*: {module_data['b'].mean():.1f} ± {module_data['b'].std():.1f}")
        print(f"Overall Hue: {module_data['Hue'].mean():.1f}° ± {module_data['Hue'].std():.1f}°")
        print(f"Overall Chroma: {module_data['Chroma'].mean():.1f} ± {module_data['Chroma'].std():.1f}")
        
        # Angular zone analysis
        zone_columns = ['lobe_specular', 'lobe_forward_scatter', 'lobe_backscatter', 'lobe_residual']
        for zone in zone_columns:
            if zone in module_data.columns:
                zone_data = module_data[module_data[zone] == True]
                if len(zone_data) > 0:
                    zone_name = zone.replace('lobe_', '')
                    print(f"{zone_name.capitalize()} points: {len(zone_data)}")
                    print(f"  - Mean L*: {zone_data['L'].mean():.1f}")
                    print(f"  - Mean Hue: {zone_data['Hue'].mean():.1f}°")
                    print(f"  - Mean Chroma: {zone_data['Chroma'].mean():.1f}")
        
        # Per-incident-angle analysis
        angles = [0, 15, 30, 45, 60]
        print("\nPer-incident-angle analysis:")
        for angle in angles:
            angle_data = module_data[module_data['Theta_i'] == angle]
            if len(angle_data) > 0:
                print(f"  θᵢ = {angle}°: {len(angle_data)} points")
                print(f"    Hue: {angle_data['Hue'].mean():.1f}° ± {angle_data['Hue'].std():.1f}°")
                print(f"    Chroma: {angle_data['Chroma'].mean():.1f} ± {angle_data['Chroma'].std():.1f}")
                print(f"    L*: {angle_data['L'].mean():.1f} ± {angle_data['L'].std():.1f}")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    analyze_blue_modules()


