#!/usr/bin/env python3
"""
Create a bar plot ranking all modules from most diffuse to least diffuse.
This provides a comprehensive overview of diffuse scattering across all modules.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from bipv_analysis_framework import BIPVAnalyzer
from module_database import MODULE_CLASSIFICATIONS, get_module_info

def create_diffuse_ranking_plot():
    """Create a bar plot ranking all modules by diffuse scattering percentage."""
    
    # Initialize analyzer
    analyzer = BIPVAnalyzer()
    
    # Collect data for all modules
    module_data = []
    
    print("Collecting diffuse scattering data for all modules...")
    
    # Process each module group
    for group_name, module_ids in MODULE_CLASSIFICATIONS.items():
        print(f"  Processing {group_name}...")
        
        # Analyze the group
        group_results = analyzer.analyze_module_group(module_ids, group_name)
        
        if group_results:
            # Extract data for each module in the group
            for module_id in module_ids:
                result = group_results['individual_results'].get(module_id)
                if result and result['module_info']:
                    module_info = result['module_info']
                    
                    # Calculate diffuse scattering (sum of forward, backscatter, residual)
                    diffuse_percentage = 0
                    specular_percentage = 0
                    
                    # Get the zone ratios from the summary data
                    if 'zone_ratios' in result and result['zone_ratios']:
                        # Try to access zone ratios directly
                        for zone_name, zone_data in result['zone_ratios'].items():
                            if isinstance(zone_data, dict):
                                if zone_name == 'specular':
                                    specular_percentage = zone_data.get('ratio_percent', 0)
                                elif zone_name in ['forward_scatter', 'backscatter', 'residual']:
                                    diffuse_percentage += zone_data.get('ratio_percent', 0)
                    else:
                        # Try to access from the data structure that was actually generated
                        # Look for the summary data in the results
                        summary_file = os.path.join('results', f'{group_name}_summary.csv')
                        if os.path.exists(summary_file):
                            try:
                                summary_df = pd.read_csv(summary_file)
                                module_summary = summary_df[summary_df['Module_ID'] == module_id]
                                
                                if not module_summary.empty:
                                    # Calculate diffuse from the summary data
                                    for _, row in module_summary.iterrows():
                                        zone = row['Zone']
                                        ratio_percent = row['Ratio_Percent']
                                        
                                        if zone == 'specular':
                                            specular_percentage = ratio_percent
                                        elif zone in ['forward_scatter', 'backscatter', 'residual']:
                                            diffuse_percentage += ratio_percent
                            except Exception as e:
                                print(f"    Warning: Could not read summary for Module {module_id}: {e}")
                    
                    module_data.append({
                        'Module_ID': module_id,
                        'Short_Name': module_info['short_name'],
                        'Group': group_name,
                        'Pigment': module_info['pigment'],
                        'Filler_Type': module_info['filler_type'],
                        'Filler_Content': module_info['filler_content'],
                        'Diffuse_Percentage': diffuse_percentage,
                        'Specular_Percentage': specular_percentage,
                        'Category': module_info['category']
                    })
    
    if not module_data:
        print("No module data collected!")
        return
    
    # Convert to DataFrame and sort by diffuse percentage (most diffuse first)
    df = pd.DataFrame(module_data)
    df = df.sort_values('Diffuse_Percentage', ascending=False)
    
    print(f"\nCollected data for {len(df)} modules")
    print(f"Diffuse scattering range: {df['Diffuse_Percentage'].min():.1f}% to {df['Diffuse_Percentage'].max():.1f}%")
    
    # Create the ranking plot - separate plots instead of subplots
    fig1, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    fig1.suptitle('Module Ranking by Diffuse Reflectance Percentage\n(Most Diffuse → Least Diffuse)', 
                fontsize=18, fontweight='bold')
    
    # Color coding for different groups - intuitive colors
    group_colors = {
        'Reference': '#000000',           # Black
        'BaSO4_Fillers': '#404040',      # Dark gray
        'Glass_Fillers_Small': '#808080', # Gray
        'Glass_Fillers_Large': '#808080', # Gray (same as small)
        'Blue_Pigment': '#0066CC',        # Blue
        'Brown_Pigment': '#8B4513',       # Brown
        'Green_Pigment': '#228B22'        # Green
    }
    
    # Plot 1: Diffuse Scattering Ranking
    bars = ax1.bar(range(len(df)), df['Diffuse_Percentage'], 
                  color=[group_colors.get(group, '#7f7f7f') for group in df['Group']],
                  edgecolor='black', alpha=0.8)
    
    ax1.set_title('Diffuse Reflectance Percentage by Module', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Diffuse Reflectance (%)', fontsize=12)
    ax1.set_xlabel('Module Rank', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, df['Diffuse_Percentage'].max() * 1.1)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df['Diffuse_Percentage'])):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=8, rotation=45)
    
    # Set x-axis labels
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels([f"{row['Short_Name']}\n({row['Group']})" for _, row in df.iterrows()], 
                       rotation=45, ha='right', fontsize=8)
    
    # Add legend for group colors
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='black', alpha=0.8, label=group) 
                      for group, color in group_colors.items()]
    ax1.legend(handles=legend_elements, loc='upper right', title='Module Groups')
    
    plt.tight_layout()
    
    # Create second plot: Specular vs Diffuse Comparison
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    fig2.suptitle('Specular vs Diffuse Reflectance Relationship', 
                fontsize=18, fontweight='bold')
    
    # Plot 2: Specular vs Diffuse Comparison
    scatter = ax2.scatter(df['Specular_Percentage'], df['Diffuse_Percentage'], 
               c=[group_colors.get(group, '#7f7f7f') for group in df['Group']], 
               s=100, alpha=0.8, edgecolors='black')
    
    # Add module labels
    for _, row in df.iterrows():
        ax2.annotate(row['Short_Name'], 
                    (row['Specular_Percentage'], row['Diffuse_Percentage']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_title('Specular vs Diffuse Reflectance Relationship', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Specular Percentage (%)', fontsize=12)
    ax2.set_ylabel('Diffuse Percentage (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add diagonal line showing inverse relationship
    max_val = max(df['Specular_Percentage'].max(), df['Diffuse_Percentage'].max())
    ax2.plot([0, max_val], [max_val, 0], 'k--', alpha=0.5, label='Inverse Relationship')
    
    # Add legend for group colors
    legend_elements2 = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='black', alpha=0.8, label=group) 
                       for group, color in group_colors.items()]
    ax2.legend(handles=legend_elements2, loc='upper right', title='Module Groups')
    
    plt.tight_layout()
    
    # Save the plots separately
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Plot 1: Diffuse Ranking
    save_path1 = os.path.join(output_dir, 'diffuse_ranking_bar_chart.png')
    fig1.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Save Plot 2: Specular vs Diffuse
    save_path2 = os.path.join(output_dir, 'specular_vs_diffuse_scatter.png')
    fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"\nPlots saved separately:")
    print(f"  Diffuse ranking bar chart: {save_path1}")
    print(f"  Specular vs Diffuse scatter: {save_path2}")
    
    # Print ranking summary
    print("\n" + "="*80)
    print("DIFFUSE SCATTERING RANKING SUMMARY")
    print("="*80)
    print(f"{'Rank':<4} {'Module':<12} {'Group':<20} {'Diffuse %':<10} {'Specular %':<12}")
    print("-"*80)
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"{i:<4} {row['Short_Name']:<12} {row['Group']:<20} {row['Diffuse_Percentage']:<10.1f} {row['Specular_Percentage']:<12.1f}")
    
    # Save ranking data to CSV
    csv_path = os.path.join(output_dir, 'all_modules_diffuse_ranking.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nRanking data saved to: {csv_path}")
    
    return df

if __name__ == "__main__":
    ranking_df = create_diffuse_ranking_plot()
    print(f"\n✅ Diffuse ranking analysis complete for {len(ranking_df)} modules!")
