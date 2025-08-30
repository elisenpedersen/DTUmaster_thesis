import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# Define the specific modules to analyze with their proper names
module_mapping = {
    'mdiff1': 'REF',
    'mdiff2': 'B5',
    'mdiff3': 'B10', 
    'mdiff4': 'B20',
    'mdiff5': 'Gl2.1S5',
    'mdiff6': 'Gl2.1S20',
    'mdiff7': 'Gl1.5S5',
    'mdiff8': 'Gl1.5S20',
    'mdiff9': 'Blue2.1S',
    'mdiff10': 'Blue2.1L',
    'mdiff11': 'Blue1.5S',
    'mdiff12': 'Blue1.5L',
    'mdiff13': 'BlueBa',
    'mdiff14': 'Gl2.1L5',
    'mdiff15': 'Gl2.1L20',
    'mdiff16': 'Gl1.5L5',
    'mdiff17': 'Gl1.5L20',
    'mdiff18': 'BrownC',
    'mdiff20': 'BrownBaC',
    'mdiff22': 'BrownGlC',
    'mdiff25': 'GreenC',
    'mdiff27': 'GreenBaC',
    'mdiff28': 'GreenGlC'
}

# Load the integrated diffuse reflectance data
try:
    data = pd.read_csv("integrated_diff_reflectance_all.csv")
    print("Loaded integrated diffuse reflectance data")
except FileNotFoundError:
    print("File 'integrated_diff_reflectance_all.csv' not found. Please run the previous analysis first.")
    exit()

# Filter data to only include the specified modules
available_modules = data['Module'].tolist()
requested_modules = list(module_mapping.keys())

# Find which requested modules are available
available_requested = [m for m in requested_modules if m in available_modules]
missing_modules = [m for m in requested_modules if m not in available_modules]

if missing_modules:
    print(f"Missing modules: {missing_modules}")

if not available_requested:
    print("None of the requested modules are available in the data")
    exit()

filtered_data = data[data['Module'].isin(available_requested)].copy()

# Add the proper names
filtered_data['Module Name'] = filtered_data['Module'].map(module_mapping)

# Separate data into two groups - fix the filtering logic
non_pigmented = filtered_data[filtered_data['Module Name'].str.startswith(('REF', 'B', 'Gl')) & 
                               ~filtered_data['Module Name'].str.startswith(('Blue', 'Brown', 'Green'))].copy()
pigmented = filtered_data[filtered_data['Module Name'].str.startswith(('Blue', 'Brown', 'Green'))].copy()

print(f"Non-pigmented modules found: {len(non_pigmented)}")
print(f"Pigmented modules found: {len(pigmented)}")
print("Non-pigmented modules:", non_pigmented['Module Name'].tolist())
print("Pigmented modules:", pigmented['Module Name'].tolist())

# Create colors for the groups
colors = {
    'REF': '#d62728',      # Red for reference
    'B': '#8c564b',        # Neutral brown for BaSO4
    'Gl': '#7f7f7f',       # Neutral gray for glass
    'Blue': '#1f77b4',     # Blue for blue modules
    'Brown': '#8c564b',    # Brown for brown modules
    'Green': '#2ca02c'     # Green for green modules
}

# Function to create a plot
def create_plot(data_subset, title, filename, colors_dict):
    if len(data_subset) == 0:
        print(f"No data for {title}")
        return
        
    # Sort by integrated reflectance
    data_subset = data_subset.sort_values('Integrated Reflectance')
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Assign colors based on module name prefix
    bar_colors = []
    for name in data_subset['Module Name']:
        if name.startswith('REF'):
            bar_colors.append(colors_dict['REF'])
        elif name.startswith('B') and not name.startswith(('Blue', 'Brown')):  # Exclude Blue modules
            bar_colors.append(colors_dict['B'])
        elif name.startswith('Gl'):
            bar_colors.append(colors_dict['Gl'])
        elif name.startswith('Blue'):
            bar_colors.append(colors_dict['Blue'])
        elif name.startswith('Brown'):
            bar_colors.append(colors_dict['Brown'])
        elif name.startswith('Green'):
            bar_colors.append(colors_dict['Green'])
        else:
            bar_colors.append('#7f7f7f')  # Gray for unknown
    
    print(f"Color assignment for {title}:")
    for i, (name, color) in enumerate(zip(data_subset['Module Name'], bar_colors)):
        print(f"  {name}: {color}")
    
    # Create the bar plot
    bars = plt.bar(range(len(data_subset)), data_subset['Integrated Reflectance'], 
                   color=bar_colors, edgecolor='black', linewidth=0.5)
    
    # Set x-axis labels
    plt.xticks(range(len(data_subset)), data_subset['Module Name'], rotation=45, ha='right', fontsize=12)
    
    # Add labels and title
    plt.xlabel('Module Type', fontsize=14)
    plt.ylabel('Integrated Diffuse Reflectance', fontsize=14)
    plt.title(title, fontsize=18)
    
    # Add grid
    plt.grid(True, axis='y', alpha=0.3)
    
    # Create legend for the groups present in this plot
    legend_elements = []
    legend_names = {
        'REF': 'Reference (BaSO₄)',
        'B': 'BaSO₄ Fillers', 
        'Gl': 'Glass Fillers',
        'Blue': 'Blue Pigments',
        'Brown': 'Brown Pigments',
        'Green': 'Green Pigments'
    }
    
    groups_present = set()
    for name in data_subset['Module Name']:
        if name.startswith('REF'):
            groups_present.add('REF')
        elif name.startswith('Blue'):
            groups_present.add('Blue')
        elif name.startswith('Brown'):
            groups_present.add('Brown')
        elif name.startswith('Green'):
            groups_present.add('Green')
        elif name.startswith('B') and not name.startswith(('Blue', 'Brown')):
            groups_present.add('B')
        elif name.startswith('Gl'):
            groups_present.add('Gl')
    
    print(f"Groups detected in {title}: {groups_present}")
    
    for group in groups_present:
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=colors_dict[group], edgecolor='black', 
                                           label=legend_names[group], linewidth=0.5))
    
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=True, 
               title='Module Groups', title_fontsize=13)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot as '{filename}'")
    

# Create the two plots
print(f"\nCreating plot 1: Non-pigmented modules ({len(non_pigmented)} modules)")
create_plot(non_pigmented, 'Diffuse Reflectance for Non-pigmented Modules', 
           'non_pigmented_modules_diffuse_reflectance.png', colors)

print(f"\nCreating plot 2: Pigmented modules ({len(pigmented)} modules)")
create_plot(pigmented, 'Diffuse Reflectance for Pigmented Modules', 
           'pigmented_modules_diffuse_reflectance.png', colors)

# Save the filtered data with proper names
output_file = "selected_modules_diffuse_reflectance.csv"
filtered_data.to_csv(output_file, index=False)
print(f"\nSaved filtered data as '{output_file}'")

# Print summary statistics by group
print("\nSummary by Module Group:")
print("=" * 50)

# Group by module type prefix
filtered_data['Group'] = filtered_data['Module Name'].str.extract(r'^([A-Za-z]+)')[0]

# Fix the grouping for special cases
filtered_data.loc[filtered_data['Module Name'] == 'BlueBa', 'Group'] = 'Blue'

group_stats = filtered_data.groupby('Group')['Integrated Reflectance'].agg(['mean', 'std', 'count']).round(4)

for group, stats in group_stats.iterrows():
    print(f"{group:8} | Mean: {stats['mean']:.4f} | Std: {stats['std']:.4f} | Count: {stats['count']}")