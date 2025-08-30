"""
Hue Calculation Comparison
This script demonstrates the difference between the old and new hue calculation methods.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import BRDFDataLoader, FilteringConfig
from color_statistics import AdvancedColorAnalyzer

def compare_hue_calculations():
    """Compare old vs new hue calculation methods."""
    
    print("HUE CALCULATION COMPARISON")
    print("=" * 50)
    
    # Initialize components
    loader = BRDFDataLoader('LAB/all_modules_lab_analysis/all_modules_lab_results.csv')
    analyzer = AdvancedColorAnalyzer()
    config = FilteringConfig(min_lightness=15.0, max_lightness=95.0, exclude_specular=True, min_chroma=1.0)
    
    # Test with a few modules
    test_modules = [18, 20, 22, 24, 26, 29]  # Brown and green modules
    theta_i = 0
    
    comparison_data = []
    
    for module_num in test_modules:
        print(f"\nAnalyzing Module {module_num}...")
        
        # Load and filter data
        df = loader.get_filtered_dataset(module_num, theta_i, 0, config)
        if df is None or len(df) == 0:
            print(f"  No data available for Module {module_num}")
            continue
        
        # Calculate statistics
        stats = analyzer.calculate_comprehensive_statistics(df)
        
        # Old method: arithmetic mean of hue angles
        old_mean_h = df['H_star'].mean()
        
        # New method: angle of mean a* and mean b*
        mean_a = df['a_star'].mean()
        mean_b = df['b_star'].mean()
        new_mean_h = (np.degrees(np.arctan2(mean_b, mean_a)) + 360) % 360
        
        # Calculate difference
        hue_diff = abs(new_mean_h - old_mean_h)
        if hue_diff > 180:  # Handle circular difference
            hue_diff = 360 - hue_diff
        
        comparison_data.append({
            'module': module_num,
            'n_points': len(df),
            'old_mean_h': old_mean_h,
            'new_mean_h': new_mean_h,
            'hue_difference': hue_diff,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'mean_l': df['L_star'].mean(),
            'mean_c': df['C_star'].mean()
        })
        
        print(f"  Points: {len(df)}")
        print(f"  Old method (arithmetic mean): {old_mean_h:.1f}°")
        print(f"  New method (angle of mean a*b*): {new_mean_h:.1f}°")
        print(f"  Difference: {hue_diff:.1f}°")
        print(f"  Mean a*: {mean_a:.1f}, Mean b*: {mean_b:.1f}")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save results
    output_dir = 'improved_analysis_results/hue_calculation_comparison'
    os.makedirs(output_dir, exist_ok=True)
    comparison_df.to_csv(f'{output_dir}/hue_calculation_comparison.csv', index=False)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Old vs New hue values
    ax1.scatter(comparison_df['old_mean_h'], comparison_df['new_mean_h'], 
               s=100, alpha=0.7, c=['brown' if m in [18, 20, 22] else 'green' for m in comparison_df['module']])
    
    # Add module labels
    for _, row in comparison_df.iterrows():
        ax1.annotate(f"M{row['module']}", (row['old_mean_h'], row['new_mean_h']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add diagonal line
    min_val = min(comparison_df['old_mean_h'].min(), comparison_df['new_mean_h'].min())
    max_val = max(comparison_df['old_mean_h'].max(), comparison_df['new_mean_h'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect agreement')
    
    ax1.set_xlabel('Old Method: Arithmetic Mean of Hue Angles (°)')
    ax1.set_ylabel('New Method: Angle of Mean a* and Mean b* (°)')
    ax1.set_title('Hue Calculation Method Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Hue differences
    colors = ['brown' if m in [18, 20, 22] else 'green' for m in comparison_df['module']]
    bars = ax2.bar(range(len(comparison_df)), comparison_df['hue_difference'], 
                   color=colors, alpha=0.7)
    
    ax2.set_xlabel('Module')
    ax2.set_ylabel('Absolute Hue Difference (°)')
    ax2.set_title('Difference Between Old and New Hue Calculations')
    ax2.set_xticks(range(len(comparison_df)))
    ax2.set_xticklabels([f"M{m}" for m in comparison_df['module']])
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}°', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hue_calculation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"\nSUMMARY")
    print("-" * 20)
    print(f"Average hue difference: {comparison_df['hue_difference'].mean():.1f}°")
    print(f"Maximum hue difference: {comparison_df['hue_difference'].max():.1f}°")
    print(f"Minimum hue difference: {comparison_df['hue_difference'].min():.1f}°")
    
    # Show which method is more appropriate
    print(f"\nCONCLUSION")
    print("-" * 20)
    print("The new method (angle of mean a* and mean b*) is mathematically correct")
    print("for averaging hue angles, especially when hue values span across the 0°/360° boundary.")
    print("This is the same method used in hue_chroma_analysis.py.")
    
    return comparison_df

def main():
    """Main function to run hue calculation comparison."""
    comparison_df = compare_hue_calculations()
    
    print(f"\nResults saved to 'improved_analysis_results/hue_calculation_comparison/'")
    print(f"Comparison CSV: 'hue_calculation_comparison.csv'")
    print(f"Comparison plot: 'hue_calculation_comparison.png'")

if __name__ == "__main__":
    main() 