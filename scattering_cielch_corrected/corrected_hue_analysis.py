import pandas as pd
import numpy as np

def correct_hue_statistics(data_file, module_id):
    """Calculate correct hue statistics using vector-based approach."""
    df = pd.read_csv(data_file)
    module_data = df[df['Module'] == module_id]
    
    if len(module_data) == 0:
        print(f"No data found for module {module_id}")
        return None
    
    print(f"=== Module {module_id} - Corrected Hue Analysis ===")
    print(f"Total data points: {len(module_data)}")
    
    # Calculate mean a* and b* vectors
    mean_a = module_data['a'].mean()
    mean_b = module_data['b'].mean()
    
    # Calculate correct mean hue from vector components
    correct_mean_hue = np.degrees(np.arctan2(mean_b, mean_a))
    if correct_mean_hue < 0:
        correct_mean_hue += 360
    
    # Calculate hue dispersion using vector components
    # Convert to radians for vector calculations
    a_vectors = module_data['a'].values
    b_vectors = module_data['b'].values
    
    # Calculate the magnitude of the mean vector
    mean_vector_magnitude = np.sqrt(mean_a**2 + mean_b**2)
    
    # Calculate the circular standard deviation
    # This is a measure of how spread out the hue angles are
    n = len(module_data)
    if n > 1:
        # Calculate the mean resultant vector length
        cos_hues = np.cos(np.radians(module_data['Hue']))
        sin_hues = np.sin(np.radians(module_data['Hue']))
        
        R = np.sqrt(np.sum(cos_hues)**2 + np.sum(sin_hues)**2) / n
        
        # Circular standard deviation
        circular_std = np.sqrt(-2 * np.log(R))
        circular_std_deg = np.degrees(circular_std)
    else:
        circular_std_deg = 0
    
    print(f"Mean a*: {mean_a:.2f}")
    print(f"Mean b*: {mean_b:.2f}")
    print(f"Correct mean hue: {correct_mean_hue:.1f}°")
    print(f"Mean vector magnitude: {mean_vector_magnitude:.2f}")
    print(f"Circular standard deviation: {circular_std_deg:.1f}°")
    
    # Compare with naive averaging
    naive_mean_hue = module_data['Hue'].mean()
    naive_std_hue = module_data['Hue'].std()
    
    print(f"\nComparison with naive approach:")
    print(f"Naive mean hue: {naive_mean_hue:.1f}°")
    print(f"Naive std hue: {naive_std_hue:.1f}°")
    print(f"Difference: {abs(correct_mean_hue - naive_mean_hue):.1f}°")
    
    # Per-incident-angle analysis with correct hue calculation
    angles = [0, 15, 30, 45, 60]
    print(f"\nPer-incident-angle analysis (corrected):")
    
    for angle in angles:
        angle_data = module_data[module_data['Theta_i'] == angle]
        if len(angle_data) > 0:
            angle_mean_a = angle_data['a'].mean()
            angle_mean_b = angle_data['b'].mean()
            angle_correct_hue = np.degrees(np.arctan2(angle_mean_b, angle_mean_a))
            if angle_correct_hue < 0:
                angle_correct_hue += 360
            
            print(f"  θᵢ = {angle}°: {len(angle_data)} points")
            print(f"    Correct mean hue: {angle_correct_hue:.1f}°")
            print(f"    Mean a*: {angle_mean_a:.2f}")
            print(f"    Mean b*: {angle_mean_b:.2f}")
    
    return {
        'correct_mean_hue': correct_mean_hue,
        'circular_std': circular_std_deg,
        'mean_a': mean_a,
        'mean_b': mean_b,
        'mean_vector_magnitude': mean_vector_magnitude
    }

def main():
    """Analyze all modules with correct hue statistics."""
    modules = [18, 20, 25, 27]  # Brown and Green modules
    
    results = {}
    
    for module_id in modules:
        data_file = f"m{module_id}_cielch_data_corrected.csv"
        if os.path.exists(data_file):
            result = correct_hue_statistics(data_file, module_id)
            if result:
                results[module_id] = result
            print("\n" + "="*50 + "\n")
    
    # Summary comparison
    print("=== SUMMARY COMPARISON ===")
    print("Module | Correct Mean Hue | Circular Std Dev | Mean Vector Magnitude")
    print("-------|------------------|------------------|----------------------")
    for module_id, result in results.items():
        print(f"{module_id:6d} | {result['correct_mean_hue']:15.1f}° | {result['circular_std']:16.1f}° | {result['mean_vector_magnitude']:22.2f}")

if __name__ == "__main__":
    import os
    main()


