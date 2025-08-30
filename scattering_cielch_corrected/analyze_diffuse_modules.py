import pandas as pd
import numpy as np

def analyze_diffuse_modules():
    """Analyze available diffuse modules for correct hue statistics."""
    # Check what modules are available
    df = pd.read_csv('scattering_lobe_color_data_corrected.csv')
    available_modules = df['Module'].unique()
    print(f"Available modules: {available_modules}")
    
    # Analyze module 1 (BlueBa) which appears to be a diffuse module
    module_1_data = df[df['Module'] == 1]
    if len(module_1_data) > 0:
        print(f"\n=== Module 1 (BlueBa - Diffuse) Analysis ===")
        print(f"Total data points: {len(module_1_data)}")
        
        # Calculate correct mean hue from vector components
        mean_a = module_1_data['a'].mean()
        mean_b = module_1_data['b'].mean()
        correct_mean_hue = np.degrees(np.arctan2(mean_b, mean_a))
        if correct_mean_hue < 0:
            correct_mean_hue += 360
        
        # Calculate circular standard deviation
        n = len(module_1_data)
        if n > 1:
            cos_hues = np.cos(np.radians(module_1_data['Hue']))
            sin_hues = np.sin(np.radians(module_1_data['Hue']))
            R = np.sqrt(np.sum(cos_hues)**2 + np.sum(sin_hues)**2) / n
            circular_std = np.sqrt(-2 * np.log(R))
            circular_std_deg = np.degrees(circular_std)
        else:
            circular_std_deg = 0
        
        # Calculate chroma statistics
        mean_chroma = module_1_data['Chroma'].mean()
        std_chroma = module_1_data['Chroma'].std()
        min_chroma = module_1_data['Chroma'].min()
        max_chroma = module_1_data['Chroma'].max()
        
        print(f"Mean a*: {mean_a:.2f}")
        print(f"Mean b*: {mean_b:.2f}")
        print(f"Correct mean hue: {correct_mean_hue:.1f}°")
        print(f"Circular standard deviation: {circular_std_deg:.1f}°")
        print(f"Mean chroma: {mean_chroma:.1f} ± {std_chroma:.1f}")
        print(f"Chroma range: {min_chroma:.1f} - {max_chroma:.1f}")
        
        # Compare with naive approach
        naive_mean_hue = module_1_data['Hue'].mean()
        naive_std_hue = module_1_data['Hue'].std()
        print(f"\nComparison with naive approach:")
        print(f"Naive mean hue: {naive_mean_hue:.1f}°")
        print(f"Naive std hue: {naive_std_hue:.1f}°")
        print(f"Difference: {abs(correct_mean_hue - naive_mean_hue):.1f}°")
        
        return {
            'correct_mean_hue': correct_mean_hue,
            'circular_std': circular_std_deg,
            'mean_chroma': mean_chroma,
            'std_chroma': std_chroma,
            'chroma_range': f"{min_chroma:.1f} - {max_chroma:.1f}"
        }
    
    return None

if __name__ == "__main__":
    result = analyze_diffuse_modules()
    if result:
        print(f"\n=== SUMMARY ===")
        print(f"Diffuse module (BlueBa):")
        print(f"  Correct mean hue: {result['correct_mean_hue']:.1f}°")
        print(f"  Circular std dev: {result['circular_std']:.1f}°")
        print(f"  Mean chroma: {result['mean_chroma']:.1f} ± {result['std_chroma']:.1f}")
        print(f"  Chroma range: {result['chroma_range']}")


