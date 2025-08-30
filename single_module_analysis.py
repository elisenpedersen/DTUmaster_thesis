"""
Single Module Analysis
Detailed analysis of individual modules across incident angles.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bipv_analysis_framework import BIPVAnalyzer
from module_database import get_module_info, print_module_summary

def analyze_single_module_detailed(module_id):
    """Perform detailed analysis of a single module."""
    
    analyzer = BIPVAnalyzer()
    
    # Get module information
    module_info = get_module_info(module_id)
    if not module_info:
        print(f"Module {module_id} not found in database")
        return
    
    print(f"Analyzing Module {module_id}: {module_info['short_name']}")
    print(f"Pigment: {module_info['pigment']}")
    print(f"Filler Type: {module_info['filler_type']}")
    print(f"Filler Content: {module_info['filler_content']}%")
    print(f"Notes: {module_info['notes']}")
    
    # Perform analysis
    result = analyzer.analyze_single_module(module_id)
    
    # Create detailed plots
    create_single_module_plots(result)
    
    # Print detailed statistics
    print_detailed_statistics(result)
    
    return result

def create_single_module_plots(result):
    """Create detailed plots for a single module."""
    
    analyzer = BIPVAnalyzer()
    module_id = result['module_id']
    module_info = result['module_info']
    data = result['data']
    zone_ratios = result['zone_ratios']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Overall zone distribution
    ax1 = plt.subplot(2, 3, 1)
    zones = list(zone_ratios.keys())
    ratios = [zone_ratios[zone]['ratio'] * 100 for zone in zones]
    colors = ['white', 'grey', 'black', 'red']
    
    bars = ax1.bar(zones, ratios, color=colors, edgecolor='black', alpha=0.7)
    ax1.set_title(f'Module {module_id} ({module_info["short_name"]}) - Overall Zone Distribution')
    ax1.set_ylabel('Percentage of Total Reflected Light (%)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{ratio:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Specular reflection vs incident angle
    ax2 = plt.subplot(2, 3, 2)
    angles = [0, 15, 30, 45, 60]
    specular_by_angle = []
    
    for angle in angles:
        angle_data = data[data['Theta_i'] == angle]
        if len(angle_data) > 0:
            zone_ratios_angle, _ = analyzer.calculate_weighted_zone_ratios(angle_data)
            specular_by_angle.append(zone_ratios_angle['specular']['ratio'] * 100)
        else:
            specular_by_angle.append(0)
    
    ax2.plot(angles, specular_by_angle, 'o-', linewidth=2, markersize=8, color='blue')
    ax2.set_title('Specular Reflection vs Incident Angle')
    ax2.set_xlabel('Incident Angle θᵢ (°)')
    ax2.set_ylabel('Specular Light Percentage (%)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Plot 3: Diffuse scattering vs incident angle
    ax3 = plt.subplot(2, 3, 3)
    diffuse_by_angle = []
    
    for angle in angles:
        angle_data = data[data['Theta_i'] == angle]
        if len(angle_data) > 0:
            zone_ratios_angle, _ = analyzer.calculate_weighted_zone_ratios(angle_data)
            forward = zone_ratios_angle['forward_scatter']['ratio'] * 100
            backscatter = zone_ratios_angle['backscatter']['ratio'] * 100
            diffuse_by_angle.append(forward + backscatter)
        else:
            diffuse_by_angle.append(0)
    
    ax3.plot(angles, diffuse_by_angle, 's-', linewidth=2, markersize=8, color='green')
    ax3.set_title('Diffuse Scattering vs Incident Angle')
    ax3.set_xlabel('Incident Angle θᵢ (°)')
    ax3.set_ylabel('Diffuse Light Percentage (%)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 50)
    
    # Plot 4: Angular deviation histogram
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(data['Angular_Deviation'], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    ax4.set_title('Angular Deviation Distribution')
    ax4.set_xlabel('Angular Deviation from Specular (°)')
    ax4.set_ylabel('Number of Measurements')
    ax4.grid(True, alpha=0.3)
    
    # Add zone boundaries
    ANGULAR_ZONES = {
        'specular': {'deviation_range': (0, 15)},
        'forward_scatter': {'deviation_range': (15, 45)},
        'backscatter': {'deviation_range': (45, 90)},
        'residual': {'deviation_range': (90, 180)}
    }
    for zone, config in ANGULAR_ZONES.items():
        min_dev, max_dev = config['deviation_range']
        ax4.axvline(x=min_dev, color='red', linestyle='--', alpha=0.7)
        ax4.axvline(x=max_dev, color='red', linestyle='--', alpha=0.7)
    
    # Plot 5: L* vs Angular Deviation
    ax5 = plt.subplot(2, 3, 5)
    
    # Debug: Check actual L* values being plotted
    print(f"DEBUG: L* values range: {data['L'].min():.1f} to {data['L'].max():.1f}")
    print(f"DEBUG: First 5 L* values: {data['L'].head().tolist()}")
    
    scatter = ax5.scatter(data['Angular_Deviation'], data['L'], 
                         c=data['total_brdf'], cmap='viridis', alpha=0.6, s=20)
    ax5.set_title('L* vs Angular Deviation')
    ax5.set_xlabel('Angular Deviation from Specular (°)')
    ax5.set_ylabel('L* (Lightness)')
    
    # Adaptive L* axis limits - show full range but ensure 0-100 scale if possible
    l_min, l_max = data['L'].min(), data['L'].max()
    if l_max <= 100:
        ax5.set_ylim(0, 100)  # Standard CIE L*a*b* range
    else:
        # If data exceeds 100, show full range but add note
        ax5.set_ylim(0, l_max * 1.1)
        ax5.text(0.02, 0.98, f'L* range: {l_min:.1f}-{l_max:.1f}', 
                transform=ax5.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Total BRDF')
    
    # Plot 6: Chroma vs Angular Deviation
    ax6 = plt.subplot(2, 3, 6)
    scatter = ax6.scatter(data['Angular_Deviation'], data['Chroma'], 
                         c=data['total_brdf'], cmap='viridis', alpha=0.6, s=20)
    ax6.set_title('Chroma vs Angular Deviation')
    ax6.set_xlabel('Angular Deviation from Specular (°)')
    ax6.set_ylabel('Chroma')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Total BRDF')
    
    plt.tight_layout()
    plt.savefig(f'results/Module_{module_id}_detailed_analysis.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed plots saved for Module {module_id}")

def print_detailed_statistics(result):
    """Print detailed statistics for a single module."""
    
    analyzer = BIPVAnalyzer()
    module_id = result['module_id']
    module_info = result['module_info']
    data = result['data']
    zone_ratios = result['zone_ratios']
    
    print(f"\n{'='*60}")
    print(f"DETAILED STATISTICS - Module {module_id} ({module_info['short_name']})")
    print(f"{'='*60}")
    
    # Overall statistics
    print(f"\nOverall Light Distribution:")
    for zone, stats in zone_ratios.items():
        print(f"  {zone:15s}: {stats['ratio']:.3f} ({stats['ratio']*100:.1f}%) - {stats['point_count']} points")
    
    # Per-incident-angle statistics
    print(f"\nPer-Incident-Angle Analysis:")
    angles = [0, 15, 30, 45, 60]
    
    for angle in angles:
        angle_data = data[data['Theta_i'] == angle]
        if len(angle_data) > 0:
            zone_ratios_angle, _ = analyzer.calculate_weighted_zone_ratios(angle_data)
            print(f"\n  θᵢ = {angle}°:")
            for zone, stats in zone_ratios_angle.items():
                print(f"    {zone:15s}: {stats['ratio']:.3f} ({stats['ratio']*100:.1f}%) - {stats['point_count']} points")
    
    # Angular deviation statistics
    print(f"\nAngular Deviation Statistics:")
    print(f"  Mean: {data['Angular_Deviation'].mean():.2f}°")
    print(f"  Std:  {data['Angular_Deviation'].std():.2f}°")
    print(f"  Min:  {data['Angular_Deviation'].min():.2f}°")
    print(f"  Max:  {data['Angular_Deviation'].max():.2f}°")
    
    # Color statistics
    print(f"\nColor Statistics:")
    print(f"  L* - Mean: {data['L'].mean():.2f}, Std: {data['L'].std():.2f}")
    print(f"  a* - Mean: {data['a'].mean():.2f}, Std: {data['a'].std():.2f}")
    print(f"  b* - Mean: {data['b'].mean():.2f}, Std: {data['b'].std():.2f}")
    print(f"  Chroma - Mean: {data['Chroma'].mean():.2f}, Std: {data['Chroma'].std():.2f}")
    
    # BRDF statistics
    print(f"\nBRDF Statistics:")
    print(f"  Total BRDF - Mean: {data['total_brdf'].mean():.6f}")
    print(f"  Total BRDF - Std:  {data['total_brdf'].std():.6f}")
    print(f"  Total BRDF - Min:  {data['total_brdf'].min():.6f}")
    print(f"  Total BRDF - Max:  {data['total_brdf'].max():.6f}")

def main():
    """Main function for single module analysis."""
    
    if len(sys.argv) > 1:
        module_id = int(sys.argv[1])
    else:
        print("Available modules:")
        print_module_summary()
        module_id = int(input("\nEnter module ID to analyze: "))
    
    # Perform analysis
    result = analyze_single_module_detailed(module_id)
    
    # Save results
    if result:
        output_file = f'results/Module_{module_id}_analysis.csv'
        result['data'].to_csv(output_file, index=False)
        print(f"\nDetailed data saved to: {output_file}")

if __name__ == '__main__':
    main() 