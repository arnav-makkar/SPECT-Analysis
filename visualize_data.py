#!/usr/bin/env python3
"""
PPMI DaTSCAN Data Visualization Script
Creates visualizations for the analyzed data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_and_visualize_data():
    """Load the CSV data and create visualizations"""
    
    # Load the data
    try:
        df = pd.read_csv('ppmi_metadata_summary.csv')
        print(f"Loaded data with {len(df)} subjects")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print("CSV file not found. Please run data_analysis.py first.")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PPMI DaTSCAN Data Analysis - Control vs PD Groups', fontsize=16, fontweight='bold')
    
    # 1. Age Distribution
    ax1 = axes[0, 0]
    age_data = df[df['age'].notna()].copy()
    if not age_data.empty:
        for group in ['Control', 'PD']:
            group_data = age_data[age_data['group'] == group]['age']
            ax1.hist(group_data, alpha=0.7, label=group, bins=10)
        ax1.set_xlabel('Age (years)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Age Distribution by Group')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Sex Distribution
    ax2 = axes[0, 1]
    sex_data = df[df['sex'].notna()].copy()
    if not sex_data.empty:
        sex_counts = sex_data.groupby(['group', 'sex']).size().unstack(fill_value=0)
        sex_counts.plot(kind='bar', ax=ax2, color=['lightblue', 'lightcoral'])
        ax2.set_xlabel('Group')
        ax2.set_ylabel('Count')
        ax2.set_title('Sex Distribution by Group')
        ax2.legend(title='Sex')
        ax2.tick_params(axis='x', rotation=0)
        ax2.grid(True, alpha=0.3)
    
    # 3. Visit Types
    ax3 = axes[0, 2]
    visit_data = df[df['visit_type'].notna()].copy()
    if not visit_data.empty:
        visit_counts = visit_data.groupby(['group', 'visit_type']).size().unstack(fill_value=0)
        visit_counts.plot(kind='bar', ax=ax3, color=['lightgreen', 'orange'])
        ax3.set_xlabel('Group')
        ax3.set_ylabel('Count')
        ax3.set_title('Visit Types by Group')
        ax3.legend(title='Visit Type')
        ax3.tick_params(axis='x', rotation=0)
        ax3.grid(True, alpha=0.3)
    
    # 4. Age Box Plot
    ax4 = axes[1, 0]
    if not age_data.empty:
        age_data.boxplot(column='age', by='group', ax=ax4)
        ax4.set_xlabel('Group')
        ax4.set_ylabel('Age (years)')
        ax4.set_title('Age Distribution by Group (Box Plot)')
        ax4.grid(True, alpha=0.3)
    
    # 5. Protocol Parameters Comparison
    ax5 = axes[1, 1]
    protocol_cols = [col for col in df.columns if col.startswith('protocol_') and 'pixel_spacing' in col]
    if protocol_cols:
        # Get unique values for pixel spacing
        pixel_data = []
        for col in protocol_cols:
            for group in ['Control', 'PD']:
                group_data = df[df['group'] == group][col].dropna()
                for value in group_data:
                    try:
                        pixel_data.append({'group': group, 'value': float(value), 'parameter': col})
                    except:
                        continue
        
        if pixel_data:
            pixel_df = pd.DataFrame(pixel_data)
            pixel_df.boxplot(column='value', by='group', ax=ax5)
            ax5.set_xlabel('Group')
            ax5.set_ylabel('Pixel Spacing (mm)')
            ax5.set_title('Pixel Spacing by Group')
            ax5.grid(True, alpha=0.3)
    
    # 6. Summary Statistics Table
    ax6 = axes[1, 2]
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create summary table
    summary_data = []
    for group in ['Control', 'PD']:
        group_data = df[df['group'] == group]
        
        # Age stats
        age_stats = group_data['age'].describe()
        age_mean = f"{age_stats['mean']:.1f} ± {age_stats['std']:.1f}"
        
        # Sex distribution
        sex_dist = group_data['sex'].value_counts()
        sex_str = f"M: {sex_dist.get('M', 0)}, F: {sex_dist.get('F', 0)}"
        
        # Visit types
        visit_dist = group_data['visit_type'].value_counts()
        visit_str = f"Screening: {visit_dist.get('Screening', 0)}"
        
        summary_data.append([group, age_mean, sex_str, visit_str])
    
    summary_df = pd.DataFrame(summary_data, 
                            columns=['Group', 'Age (Mean±SD)', 'Sex (M/F)', 'Visit Type'])
    
    table = ax6.table(cellText=summary_df.values, 
                      colLabels=summary_df.columns,
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    ax6.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the plot
    plt.savefig('ppmi_data_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'ppmi_data_visualization.png'")
    
    # Show the plot
    plt.show()
    
    # Print additional insights
    print("\n=== KEY INSIGHTS ===")
    print(f"Total subjects analyzed: {len(df)}")
    print(f"Control subjects: {len(df[df['group'] == 'Control'])}")
    print(f"PD subjects: {len(df[df['group'] == 'PD'])}")
    
    if not age_data.empty:
        print(f"\nAge Analysis:")
        for group in ['Control', 'PD']:
            group_ages = age_data[age_data['group'] == group]['age']
            if not group_ages.empty:
                print(f"  {group}: Mean={group_ages.mean():.1f}±{group_ages.std():.1f}, Range={group_ages.min():.1f}-{group_ages.max():.1f}")
    
    # Protocol differences
    print(f"\nImaging Protocol Differences:")
    protocol_cols = [col for col in df.columns if col.startswith('protocol_')]
    if protocol_cols:
        key_params = ['protocol_manufacturer', 'protocol_pixel_spacing_row', 'protocol_frame_duration']
        for param in key_params:
            if param in df.columns:
                print(f"  {param.replace('protocol_', '').replace('_', ' ').title()}:")
                for group in ['Control', 'PD']:
                    group_data = df[df['group'] == group][param].dropna()
                    if not group_data.empty:
                        unique_values = group_data.unique()
                        print(f"    {group}: {list(unique_values)}")

if __name__ == "__main__":
    load_and_visualize_data()
