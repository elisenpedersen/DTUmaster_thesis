import pandas as pd
import numpy as np
import os

def analyze_module_comprehensive(module_id, data_file):
    """Analyze a module comprehensively for the comparison table."""
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return None
    
    df = pd.read_csv(data_file)
    module_data = df[df['Module'] == module_id]
    
    if len(module_data) == 0:
        print(f"No data found for module {module_id}")
        return None
    
    # Calculate correct mean hue from vector components
    mean_a = module_data['a'].mean()
    mean_b = module_data['b'].mean()
    correct_mean_hue = np.degrees(np.arctan2(mean_b, mean_a))
    if correct_mean_hue < 0:
        correct_mean_hue += 360
    
    # Calculate circular standard deviation
    n = len(module_data)
    if n > 1:
        cos_hues = np.cos(np.radians(module_data['Hue']))
        sin_hues = np.sin(np.radians(module_data['Hue']))
        R = np.sqrt(np.sum(cos_hues)**2 + np.sum(sin_hues)**2) / n
        circular_std = np.sqrt(-2 * np.log(R))
        circular_std_deg = np.degrees(circular_std)
    else:
        circular_std_deg = 0
    
    # Calculate chroma statistics
    mean_chroma = module_data['Chroma'].mean()
    std_chroma = module_data['Chroma'].std()
    min_chroma = module_data['Chroma'].min()
    max_chroma = module_data['Chroma'].max()
    
    return {
        'module_id': module_id,
        'correct_mean_hue': correct_mean_hue,
        'circular_std': circular_std_deg,
        'mean_chroma': mean_chroma,
        'std_chroma': std_chroma,
        'chroma_range': f"{min_chroma:.1f} - {max_chroma:.1f}",
        'total_points': n
    }

def main():
    """Analyze all modules for comprehensive comparison."""
    print("=== Comprehensive Module Analysis for Comparison Table ===\n")
    
    # Define modules to analyze with their descriptions
    modules = {
        1: "BlueBa (BaSO4 filler)",
        9: "Blue2.1S (uniform)",
        18: "BrownC (uniform)", 
        20: "BrownBaC (BaSO4 filler)",
        25: "GreenC (uniform)",
        27: "GreenBaC (BaSO4 filler)"
    }
    
    results = {}
    
    # Analyze modules from scattering lobe data (modules 1, 9, 18, 25)
    for module_id in [1, 9, 18, 25]:
        data_file = "scattering_lobe_color_data_corrected.csv"
        result = analyze_module_comprehensive(module_id, data_file)
        if result:
            results[module_id] = result
            print(f"Module {module_id} ({modules[module_id]}):")
            print(f"  Correct mean hue: {result['correct_mean_hue']:.1f}°")
            print(f"  Circular std dev: {result['circular_std']:.1f}°")
            print(f"  Mean chroma: {result['mean_chroma']:.1f} ± {result['std_chroma']:.1f}")
            print(f"  Chroma range: {result['chroma_range']}")
            print()
    
    # Analyze modules from individual CSV files (modules 20, 27)
    for module_id in [20, 27]:
        data_file = f"m{module_id}_cielch_data_corrected.csv"
        result = analyze_module_comprehensive(module_id, data_file)
        if result:
            results[module_id] = result
            print(f"Module {module_id} ({modules[module_id]}):")
            print(f"  Correct mean hue: {result['correct_mean_hue']:.1f}°")
            print(f"  Circular std dev: {result['circular_std']:.1f}°")
            print(f"  Mean chroma: {result['mean_chroma']:.1f} ± {result['std_chroma']:.1f}")
            print(f"  Chroma range: {result['chroma_range']}")
            print()
    
    # Create comprehensive comparison table
    print("=== COMPREHENSIVE COMPARISON TABLE ===")
    print("Module Group | Module | Correct Mean Hue | Circular Std Dev | Mean Chroma | Chroma Range")
    print("-------------|---------|------------------|------------------|-------------|-------------")
    
    # Group by type
    blue_modules = [(1, "BlueBa"), (9, "Blue2.1S")]
    brown_modules = [(18, "BrownC"), (20, "BrownBaC")]
    green_modules = [(25, "GreenC"), (27, "GreenBaC")]
    
    for module_id, name in blue_modules:
        if module_id in results:
            r = results[module_id]
            print(f"Blue         | {name:7} | {r['correct_mean_hue']:15.1f}° | {r['circular_std']:16.1f}° | {r['mean_chroma']:11.1f} | {r['chroma_range']}")
    
    for module_id, name in brown_modules:
        if module_id in results:
            r = results[module_id]
            print(f"Brown        | {name:7} | {r['correct_mean_hue']:15.1f}° | {r['circular_std']:16.1f}° | {r['mean_chroma']:11.1f} | {r['chroma_range']}")
    
    for module_id, name in green_modules:
        if module_id in results:
            r = results[module_id]
            print(f"Green        | {name:7} | {r['correct_mean_hue']:15.1f}° | {r['circular_std']:16.1f}° | {r['mean_chroma']:11.1f} | {r['chroma_range']}")
    
    # Summary statistics by group
    print("\n=== GROUP SUMMARY STATISTICS ===")
    
    for group_name, group_modules in [("Blue", blue_modules), ("Brown", brown_modules), ("Green", green_modules)]:
        group_results = [results[mid] for mid, _ in group_modules if mid in results]
        if group_results:
            avg_std = np.mean([r['circular_std'] for r in group_results])
            avg_chroma = np.mean([r['mean_chroma'] for r in group_results])
            print(f"{group_name} modules: Average circular std dev = {avg_std:.1f}°, Average chroma = {avg_chroma:.1f}")

if __name__ == "__main__":
    main()


