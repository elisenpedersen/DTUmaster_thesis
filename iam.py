import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# Module name mapping - short names only
module_names = {
    1: "REF",
    2: "BaSO4 5%", 
    3: "BaSO4 10%",
    4: "BaSO4 20%",
    5: "Gl2.1S5",
    6: "Gl2.1S20", 
    7: "Gl1.5S5",
    8: "Gl1.5S20",
    9: "Blue2.1S",
    10: "Blue2.1L",
    11: "Blue1.5S", 
    12: "Blue1.5L",
    13: "BlueBa",
    14: "Gl2.1L5",
    15: "Gl2.1L20",
    16: "Gl1.5L5", 
    17: "Gl1.5L20",
    18: "BrownC",
    20: "BrownBaC",
    22: "BrownGIC",
    25: "GreenC",
    27: "GreenBaC",
    28: "GreenGIC"
}

# Color schemes with different shades for better distinction
def get_baso4_colors(module_num):
    if module_num == 2:  # BaSO4 5%
        return '#FF0000'  # Bright red
    elif module_num == 3:  # BaSO4 10%
        return '#CC0000'  # Darker red
    elif module_num == 4:  # BaSO4 20%
        return '#990000'  # Even darker red
    else:
        return 'gray'

def get_glass_colors(module_num):
    if module_num == 5:  # Gl2.1S5
        return '#0000FF'  # Bright blue
    elif module_num == 6:  # Gl2.1S20
        return '#0000CC'  # Darker blue
    elif module_num == 7:  # Gl1.5S5
        return '#00FF00'  # Bright green
    elif module_num == 8:  # Gl1.5S20
        return '#00CC00'  # Darker green
    elif module_num == 14:  # Gl2.1L5
        return '#00FFFF'  # Bright cyan
    elif module_num == 15:  # Gl2.1L20
        return '#00CCCC'  # Darker cyan
    elif module_num == 16:  # Gl1.5L5
        return '#FF8000'  # Bright orange
    elif module_num == 17:  # Gl1.5L20
        return '#CC6600'  # Darker orange
    else:
        return 'gray'

def get_blue_colors(module_num):
    if module_num == 9:  # Blue2.1S
        return '#0000FF'  # Bright blue
    elif module_num == 10:  # Blue2.1L
        return '#0000CC'  # Darker blue
    elif module_num == 11:  # Blue1.5S
        return '#000099'  # Even darker blue
    elif module_num == 12:  # Blue1.5L
        return '#000066'  # Very dark blue
    elif module_num == 13:  # BlueBa
        return '#000033'  # Darkest blue
    else:
        return 'gray'

def get_brown_green_colors(module_num):
    if module_num == 18:  # BrownC
        return '#8B4513'  # Saddle brown
    elif module_num == 20:  # BrownBaC
        return '#654321'  # Dark brown
    elif module_num == 22:  # BrownGIC
        return '#A0522D'  # Sienna
    elif module_num == 25:  # GreenC
        return '#228B22'  # Forest green
    elif module_num == 27:  # GreenBaC
        return '#006400'  # Dark green
    elif module_num == 28:  # GreenGIC
        return '#32CD32'  # Lime green
    else:
        return 'gray'

# Module groupings
baso4_modules = [1, 2, 3, 4]  # REF + BaSO4 modules
glass_modules = [1, 5, 6, 7, 8, 14, 15, 16, 17]  # Glass modules + REF
blue_modules = [1, 9, 10, 11, 12, 13]  # Blue modules + REF
brown_green_modules = [1, 18, 20, 22, 25, 27, 28]  # Brown and Green modules + REF

# Create figure for BaSO4 modules
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))

for module in baso4_modules:
    try:
        module_file = os.path.join(".", f"M{module}_averaged.csv")
        df = pd.read_csv(module_file)
        
        if module == 1:  # REF module - make it prominent
            module_color = 'black'
            linewidth = 3  # Thicker line for REF
            linestyle = '--'  # Dashed line for REF
        else:
            module_color = get_baso4_colors(module)
            linewidth = 2  # Normal line for BaSO4 modules
            linestyle = '-'  # Solid line
        module_label = module_names.get(module, f"Module {module}")
        
        # Convert Isc from A to mA
        ax1.plot(df["Angle (°)"], df["Isc (A)"] * 1000, label=module_label, color=module_color, linewidth=linewidth, linestyle=linestyle)
        
    except FileNotFoundError:
        print(f"Warning: M{module}_averaged.csv not found. Skipping module {module}.")

ax1.set_title("BaSO4 and Reference Modules - Angular Current Response", fontsize=14)
ax1.set_xlabel("Incident Angle (°)", fontsize=12)
ax1.set_ylabel("Short-circuit Current (mA)", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11, loc='upper right')
ax1.set_xlim(0, 80)

plt.tight_layout()
plt.savefig("IAM_baso4.png", dpi=300, bbox_inches='tight')

# Create figure for Glass modules
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

for module in glass_modules:
    try:
        module_file = os.path.join(".", f"M{module}_averaged.csv")
        df = pd.read_csv(module_file)
        
        if module == 1:  # REF module
            module_color = 'black'
            linewidth = 3  # Thicker line for REF
            linestyle = '--'  # Dashed line for REF
        else:
            module_color = get_glass_colors(module)
            linewidth = 2  # Normal line for glass modules
            linestyle = '-'  # Solid line
        module_label = module_names.get(module, f"Module {module}")
        
        # Convert Isc from A to mA
        ax2.plot(df["Angle (°)"], df["Isc (A)"] * 1000, label=module_label, color=module_color, linewidth=linewidth, linestyle=linestyle)
        
    except FileNotFoundError:
        print(f"Warning: M{module}_averaged.csv not found. Skipping module {module}.")

ax2.set_title("Glass Filler Modules - Angular Current Response", fontsize=14)
ax2.set_xlabel("Incident Angle (°)", fontsize=12)
ax2.set_ylabel("Short-circuit Current (mA)", fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11, loc='upper right', ncol=2)
ax2.set_xlim(0, 80)

plt.tight_layout()
plt.savefig("IAM_glass.png", dpi=300, bbox_inches='tight')

# Create figure for Blue modules
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))

for module in blue_modules:
    try:
        module_file = os.path.join(".", f"M{module}_averaged.csv")
        df = pd.read_csv(module_file)
        
        if module == 1:  # REF module
            module_color = 'black'
            linewidth = 3  # Thicker line for REF
            linestyle = '--'  # Dashed line for REF
        else:
            module_color = get_blue_colors(module)
            linewidth = 2  # Normal line for blue modules
            linestyle = '-'  # Solid line
        module_label = module_names.get(module, f"Module {module}")
        
        # Convert Isc from A to mA
        ax3.plot(df["Angle (°)"], df["Isc (A)"] * 1000, label=module_label, color=module_color, linewidth=linewidth, linestyle=linestyle)
        
    except FileNotFoundError:
        print(f"Warning: M{module}_averaged.csv not found. Skipping module {module}.")

ax3.set_title("Blue Pigmented Modules - Angular Current Response", fontsize=14)
ax3.set_xlabel("Incident Angle (°)", fontsize=12)
ax3.set_ylabel("Short-circuit Current (mA)", fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11, loc='upper right')
ax3.set_xlim(0, 80)

plt.tight_layout()
plt.savefig("IAM_blue.png", dpi=300, bbox_inches='tight')

# Create figure for Brown and Green modules
fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))

for module in brown_green_modules:
    try:
        module_file = os.path.join(".", f"M{module}_averaged.csv")
        df = pd.read_csv(module_file)
        
        if module == 1:  # REF module
            module_color = 'black'
            linewidth = 3  # Thicker line for REF
            linestyle = '--'  # Dashed line for REF
        else:
            module_color = get_brown_green_colors(module)
            linewidth = 2  # Normal line for brown/green modules
            linestyle = '-'  # Solid line
        module_label = module_names.get(module, f"Module {module}")
        
        # Convert Isc from A to mA
        ax4.plot(df["Angle (°)"], df["Isc (A)"] * 1000, label=module_label, color=module_color, linewidth=linewidth, linestyle=linestyle)
        
    except FileNotFoundError:
        print(f"Warning: M{module}_averaged.csv not found. Skipping module {module}.")

ax4.set_title("Brown and Green Pigmented Modules - Angular Current Response", fontsize=14)
ax4.set_xlabel("Incident Angle (°)", fontsize=12)
ax4.set_ylabel("Short-circuit Current (mA)", fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=11, loc='upper right', ncol=2)
ax4.set_xlim(0, 80)

plt.tight_layout()
plt.savefig("IAM_brown_green.png", dpi=300, bbox_inches='tight')

# Create ratio plot (60°/0° ISC ratio)
print("Creating percentage drop plot...")

# Initialize lists to store percentage drops and module names
percentage_drops = []
drop_module_names = []

# Process each module's averaged file for percentage drop calculation
# Only process modules that are defined in module_names
for module_num in module_names.keys():
    try:
        module_file = os.path.join(".", f"M{module_num}_averaged.csv")
        if os.path.exists(module_file):
            df = pd.read_csv(module_file)
            
            # Get the ISC values for 0° and 60°
            isc_0 = df[df["Angle (°)"] == 0]["Isc (A)"].mean()
            isc_60 = df[df["Angle (°)"] == 60]["Isc (A)"].mean()
            
            if not pd.isna(isc_0) and not pd.isna(isc_60):
                # Calculate the percentage performance drop from 0° to 60°
                percentage_drop = ((isc_0 - isc_60) / isc_0) * 100
                percentage_drops.append(percentage_drop)
                drop_module_names.append(module_names[module_num])
                
    except FileNotFoundError:
        continue

# Sort the modules and percentage drops based on the drop values (ascending = better performance)
sorted_indices = np.argsort(percentage_drops)
sorted_modules = [drop_module_names[i] for i in sorted_indices]
sorted_drops = [percentage_drops[i] for i in sorted_indices]

# Create the percentage drop plot
fig5, ax5 = plt.subplots(1, 1, figsize=(12, 6))

# Define colors for different module types
def get_bar_color(module_name):
    if module_name == "REF":
        return 'black'
    elif module_name.startswith("BaSO4"):
        return '#404040'  # Dark grey
    elif module_name.startswith("Gl"):
        return '#808080'  # Grey
    elif module_name.startswith("Blue"):
        return 'blue'
    elif module_name.startswith("Brown"):
        return 'brown'
    elif module_name.startswith("Green"):
        return 'green'
    else:
        return 'steelblue'  # Default color

# Create colored bars
bar_colors = [get_bar_color(module_name) for module_name in sorted_modules]
bars = ax5.bar(range(len(sorted_modules)), sorted_drops, color=bar_colors, width=0.8)

# Customize the percentage drop plot
ax5.set_title("Performance Drop (0° to 60°) - All Modules", fontsize=14)
ax5.set_xlabel("Module", fontsize=12)
ax5.set_ylabel("Performance Drop (%)", fontsize=12)
ax5.set_xticks(range(len(sorted_modules)))
ax5.set_xticklabels(sorted_modules, rotation=45, ha='right', fontsize=10)
ax5.set_ylim(0, max(sorted_drops) * 1.1)  # Dynamic y-axis limit
ax5.grid(True, alpha=0.3, axis='y')

# Add the percentage drop value above each bar
for i, (bar, drop) in enumerate(zip(bars, sorted_drops)):
    yval = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2, yval + max(sorted_drops) * 0.01, f'{drop:.1f}%', 
              ha='center', va='bottom', fontsize=9, rotation=0)

plt.tight_layout()
plt.savefig("IAM_percentage_drop.png", dpi=300, bbox_inches='tight')

print("✅ Created five IAM plots:")
print("   - IAM_baso4.png: BaSO4 and Reference modules (mA units, larger fonts)")
print("   - IAM_glass.png: Glass filler modules (mA units, larger fonts)")
print("   - IAM_blue.png: Blue pigmented modules (mA units, larger fonts)")
print("   - IAM_brown_green.png: Brown and Green pigmented modules (mA units, larger fonts)")
print("   - IAM_percentage_drop.png: IAM performance drop (0° to 60°) for all modules")