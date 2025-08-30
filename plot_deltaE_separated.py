import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CONFIGURATION
# Module definitions (23 modules)
NON_PIGMENTED = [1,2,3,4,5,6,7,8,14,15,16,17]
BLUE_MODULES = [9,10,11,12,13]
BROWN_MODULES = [18,20,22]
GREEN_MODULES = [25,27,28]

MODULES = NON_PIGMENTED + BLUE_MODULES + BROWN_MODULES + GREEN_MODULES
ANGLES = [0, 15, 30, 45, 60]
MODULE_DESCRIPTIONS = {
    1: "REF", 2: "B5", 3: "B10", 4: "B20",
    5: "G2.1S5", 6: "G2.1S20", 7: "G1.5S5", 8: "G1.5S20",
    9: "Blue2.1S", 10: "Blue2.1L", 11: "Blue1.5S", 12: "Blue1.5L", 13: "BlueBa",
    14: "G2.1L5", 15: "G2.1L20", 16: "G1.5L5", 17: "G1.5L20",
    18: "BrownC", 20: "BrownBaC", 22: "BrownGlC",
    25: "GreenC", 27: "GreenBaC", 28: "GreenGlC",
}

def calculate_deltaE_from_csv(csv_path):
    """
    Calculate Delta E over incident angles using the CSV dataset.
    Delta E represents the color difference between consecutive angles for each module.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        DataFrame with Delta E values for each module-angle combination
    """
    # Load the CSV data
    df = pd.read_csv(csv_path)
    
    # Filter to only include modules in our MODULES list
    df = df[df['Module'].isin(MODULES)]
    # Convert to string for consistency
    df['Module'] = df['Module'].astype(str)
    
    # Get unique modules and angles
    modules = df['Module'].unique()
    angles = sorted(df['Theta_i'].unique())
    
    results = []
    
    # Calculate Delta E for each module between consecutive angles
    for module in modules:
        # Get all data for this module
        module_data = df[df['Module'] == module]
        
        # Calculate mean LAB values for each angle
        angle_means = {}
        for angle in angles:
            angle_data = module_data[module_data['Theta_i'] == angle]
            if not angle_data.empty:
                angle_means[angle] = {
                    'L': angle_data['L'].mean(),
                    'a': angle_data['a'].mean(),
                    'b': angle_data['b'].mean()
                }
        
        # Calculate Delta E between consecutive angles
        for i in range(len(angles) - 1):
            current_angle = angles[i]
            next_angle = angles[i + 1]
            
            if current_angle in angle_means and next_angle in angle_means:
                # Calculate Delta E between current and next angle
                deltaE = np.sqrt(
                    (angle_means[next_angle]['L'] - angle_means[current_angle]['L'])**2 + 
                    (angle_means[next_angle]['a'] - angle_means[current_angle]['a'])**2 + 
                    (angle_means[next_angle]['b'] - angle_means[current_angle]['b'])**2
                )
                
                # Convert module number to short name
                module_int = int(module)
                module_name = MODULE_DESCRIPTIONS.get(module_int, f"Module_{module}")
                
                # Store Delta E at the current angle (representing the change to the next angle)
                results.append({
                    'Module': module_name,
                    'Theta_i': current_angle,
                    'DeltaE': deltaE
                })
    
    return pd.DataFrame(results)

def plot_deltaE_separated(deltaE_df, output_dir):
    """
    Plot Delta E over incident angles, separated by colored and non-pigmented modules.
    
    Args:
        deltaE_df: DataFrame with Delta E results
        output_dir: Directory to save plots
    """
    # Get unique modules (including reference module)
    all_modules = deltaE_df['Module'].unique()
    
    # Separate modules into colored and non-pigmented
    colored_modules = []
    non_pigmented_modules = []
    
    for module in all_modules:
        if any(color in module for color in ['Blue', 'Brown', 'Green']):
            colored_modules.append(module)
        else:
            non_pigmented_modules.append(module)
    
    # Plot 1: Non-pigmented modules
    plt.figure(figsize=(12, 8))
    
    # Define colors for non-pigmented modules (REF will be black)
    non_pigmented_colors = {}
    for i, module in enumerate(non_pigmented_modules):
        if module == 'REF':
            non_pigmented_colors[module] = 'black'
        else:
            # Use default matplotlib colors for other modules
            non_pigmented_colors[module] = f'C{i}'
    
    for module in non_pigmented_modules:
        module_data = deltaE_df[deltaE_df['Module'] == module].sort_values('Theta_i')
        color = non_pigmented_colors[module]
        
        # Plot dashed line with see-through alpha
        plt.plot(module_data['Theta_i'], module_data['DeltaE'], 'o--', 
                label=f'{module}', linewidth=1, markersize=8, alpha=0.4, 
                color=color, markeredgewidth=1.5)
        # Plot solid markers with full opacity
        plt.plot(module_data['Theta_i'], module_data['DeltaE'], 'o', 
                markersize=8, alpha=1.0, color=color, markeredgewidth=1.5)
    
    # Set custom x-axis labels to show angle transitions
    plt.xticks([0, 15, 30, 45], ['0°→15°', '15°→30°', '30°→45°', '45°→60°'])
    plt.xlabel('Angle Transition', fontsize=14)
    plt.ylabel('ΔE', fontsize=14)
    plt.title('Centroid Color Shift ΔE - Non-Pigmented Modules\n(With Lightness Filtering and Non-Specular Peaks)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deltaE_over_angles_non_pigmented.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Colored modules
    plt.figure(figsize=(12, 8))
    
    # Define color schemes for different module types
    green_colors = ['#228B22', '#32CD32', '#00FF00']  # Forest Green, Lime Green, Pure Green
    blue_colors = ['#000080', '#0000CD', '#4169E1', '#6495ED', '#87CEFA']  # Navy, Medium Blue, Royal Blue, Cornflower Blue, Light Sky Blue
    brown_colors = ['#8B4513', '#A0522D', '#CD853F']  # Saddle Brown, Sienna, Peru
    
    # Assign colors to modules
    module_colors = {}
    green_count = 0
    blue_count = 0
    brown_count = 0
    
    for module in colored_modules:
        if 'Green' in module:
            module_colors[module] = green_colors[green_count % len(green_colors)]
            green_count += 1
        elif 'Blue' in module:
            module_colors[module] = blue_colors[blue_count % len(blue_colors)]
            blue_count += 1
        elif 'Brown' in module:
            module_colors[module] = brown_colors[brown_count % len(brown_colors)]
            brown_count += 1
    
    for module in colored_modules:
        module_data = deltaE_df[deltaE_df['Module'] == module].sort_values('Theta_i')
        plt.plot(module_data['Theta_i'], module_data['DeltaE'], 'o--', 
                label=f'{module}', linewidth=1, markersize=8, alpha=0.4,
                color=module_colors[module], markeredgewidth=1.5)
        # Make markers fully opaque by plotting them again
        plt.plot(module_data['Theta_i'], module_data['DeltaE'], 'o', 
                markersize=8, alpha=1.0, color=module_colors[module], markeredgewidth=1.5)
    
    # Set custom x-axis labels to show angle transitions
    plt.xticks([0, 15, 30, 45], ['0°→15°', '15°→30°', '30°→45°', '45°→60°'])
    plt.xlabel('Angle Transition', fontsize=14)
    plt.ylabel('ΔE', fontsize=14)
    plt.title('Centroid Color Shift ΔE - Colored Modules\n(With Lightness Filtering and Non-Specular Peaks)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deltaE_over_angles_colored.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Delta E separated plots saved to {output_dir}")

def main():
    # Set up output directory
    output_dir = "LAB/deltaE_separated_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to CSV data
    csv_path = 'LAB/all_modules_lab_analysis/HUE_chroma/cielab/nonspecular/lightness_correction/hue_chroma_lab_results_nonspecular_lightness_correction.csv'
    
    if os.path.exists(csv_path):
        print(f"Loading CSV data from: {csv_path}")
        
        # Calculate Delta E
        deltaE_df = calculate_deltaE_from_csv(csv_path)
        
        # Create separated plots
        plot_deltaE_separated(deltaE_df, output_dir)
        
        # Save the data
        deltaE_df.to_csv(os.path.join(output_dir, 'deltaE_separated_data.csv'), index=False)
        
        print(f"Analysis complete. Results saved to {output_dir}")
        print(f"Generated {len(deltaE_df)} Delta E calculations")
        
    else:
        print(f"Error: CSV file not found at {csv_path}")

if __name__ == '__main__':
    main()
