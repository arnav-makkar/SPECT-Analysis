# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: jupytext,-all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: jupytext,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %%
"""
# PPMI 3D Volume Analysis

This notebook focuses on analyzing the 3D SPECT volumes and extracting meaningful features.
"""

# %%
# Cell 1: Imports and Setup
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent / 'src'))

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
from mpl_toolkits.axes_grid1 import ImageGrid
import warnings
warnings.filterwarnings('ignore')

# %%
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# %%
print("Libraries imported successfully!")

# %%
# Cell 2: Load Data and Select Sample Volumes
from data.ppmi_custom_loader import load_ppmi_data

# %%
# Load PPMI data
mapping_df, summary = load_ppmi_data()
print(f"Dataset loaded: {summary['total_images']} images from {summary['unique_patients']} patients")

# %%
# Select sample volumes from different patients
sample_volumes = []
for patient_id in mapping_df['patient_id'].unique():
    patient_images = mapping_df[mapping_df['patient_id'] == patient_id]
    sample_volumes.append(patient_images.iloc[0])

# %%
print(f"Selected {len(sample_volumes)} sample volumes for analysis")

# %%
# Cell 3: 3D Volume Loading and Basic Analysis
def load_3d_volume(file_path):
    """Load a 3D DICOM volume."""
    try:
        ds = pydicom.dcmread(file_path)
        volume = ds.pixel_array
        return volume, ds
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

# %%
# Load first sample volume
sample_file = sample_volumes[0]['file_path']
print(f"Loading sample volume: {sample_file}")

# %%
volume, ds = load_3d_volume(sample_file)
if volume is not None:
    print(f"Volume shape: {volume.shape}")
    print(f"Data type: {volume.dtype}")
    print(f"Intensity range: [{volume.min()}, {volume.max()}]")
    print(f"Mean intensity: {volume.mean():.1f}")
    print(f"Standard deviation: {volume.std():.1f}")

# %%
# Cell 4: Multi-Planar Visualization
def visualize_3d_volume(volume, patient_id, sex, age):
    """Create comprehensive 3D volume visualization."""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create a grid of images showing different slices
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(3, 4),
                    axes_pad=0.1,
                    label_mode="L",
                    share_all=True)
    
    # Show different slices through the volume
    slices_to_show = [
        (volume.shape[0]//4, "Anterior"),
        (volume.shape[0]//2, "Central"),
        (3*volume.shape[0]//4, "Posterior")
    ]
    
    slice_idx = 0
    for row in range(3):
        for col in range(4):
            if slice_idx < len(slices_to_show):
                slice_num, slice_name = slices_to_show[row]
                
                # Get the slice
                if row == 0:  # Sagittal view
                    slice_data = volume[slice_num, :, :]
                    view_name = "Sagittal"
                elif row == 1:  # Coronal view
                    slice_data = volume[:, slice_num, :]
                    view_name = "Coronal"
                else:  # Axial view
                    slice_data = volume[:, :, slice_num]
                    view_name = "Axial"
                
                # Display slice
                im = grid[slice_idx].imshow(slice_data, cmap='hot', aspect='equal')
                grid[slice_idx].set_title(f'{view_name} {slice_name}\nSlice {slice_num}')
                grid[slice_idx].set_xlabel('X')
                grid[slice_idx].set_ylabel('Y')
                
                # Add colorbar
                plt.colorbar(im, ax=grid[slice_idx], shrink=0.8)
                
                slice_idx += 1
            else:
                # Show volume statistics
                grid[slice_idx].text(0.5, 0.5, 
                                   f'Volume Stats:\n'
                                   f'Shape: {volume.shape}\n'
                                   f'Min: {volume.min():.1f}\n'
                                   f'Max: {volume.max():.1f}\n'
                                   f'Mean: {volume.mean():.1f}\n'
                                   f'Std: {volume.std():.1f}',
                                   ha='center', va='center',
                                   transform=grid[slice_idx].transAxes,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                grid[slice_idx].set_title('Volume Statistics')
                slice_idx += 1
    
    # Main title
    fig.suptitle(f'Patient {patient_id} - 3D SPECT Volume Analysis\n{sex}, {age} years', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.show()

# %%
# Visualize the first sample volume
if volume is not None:
    patient_id = sample_volumes[0]['patient_id']
    sex = sample_volumes[0].get('sex', 'Unknown')
    age = sample_volumes[0].get('age', 'Unknown')
    
    print(f"Visualizing volume for Patient {patient_id} ({sex}, {age} years)")
    visualize_3d_volume(volume, patient_id, sex, age)

# %%
# Cell 5: Maximum Intensity Projections (MIP)
def create_mip_views(volume, patient_id, sex, age):
    """Create Maximum Intensity Projection views."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Axial MIP (top view)
    axial_mip = np.max(volume, axis=0)
    im1 = axes[0].imshow(axial_mip, cmap='hot', aspect='equal')
    axes[0].set_title(f'Patient {patient_id} - Axial MIP\n{sex}, {age} years')
    axes[0].set_xlabel('Left-Right')
    axes[0].set_ylabel('Anterior-Posterior')
    plt.colorbar(im1, ax=axes[0], label='Intensity')
    
    # Coronal MIP (front view)
    coronal_mip = np.max(volume, axis=1)
    im2 = axes[1].imshow(coronal_mip, cmap='hot', aspect='equal')
    axes[1].set_title(f'Patient {patient_id} - Coronal MIP\n{sex}, {age} years')
    axes[1].set_xlabel('Left-Right')
    axes[1].set_ylabel('Superior-Inferior')
    plt.colorbar(im2, ax=axes[1], label='Intensity')
    
    # Sagittal MIP (side view)
    sagittal_mip = np.max(volume, axis=2)
    im3 = axes[2].imshow(sagittal_mip, cmap='hot', aspect='equal')
    axes[2].set_title(f'Patient {patient_id} - Sagittal MIP\n{sex}, {age} years')
    axes[2].set_xlabel('Anterior-Posterior')
    axes[2].set_ylabel('Superior-Inferior')
    plt.colorbar(im3, ax=axes[2], label='Intensity')
    
    # Add striatal region annotations (approximate)
    for ax, mip, view_name in [(axes[0], axial_mip, 'Axial'), 
                               (axes[1], coronal_mip, 'Coronal'),
                               (axes[2], sagittal_mip, 'Sagittal')]:
        
        height, width = mip.shape
        
        if view_name == 'Axial':
            # Left and right striatum in axial view
            ax.plot([width//4, width//3], [height//2, height//2], 'g-', linewidth=3, label='Left Striatum')
            ax.plot([2*width//3, 3*width//4], [height//2, height//2], 'b-', linewidth=3, label='Right Striatum')
        elif view_name == 'Coronal':
            # Striatum in coronal view
            ax.plot([width//2, width//2], [height//3, height//2], 'r-', linewidth=3, label='Striatum')
        else:  # Sagittal
            # Striatum in sagittal view
            ax.plot([width//3, width//2], [height//2, height//2], 'r-', linewidth=3, label='Striatum')
            
        ax.legend()
    
    plt.tight_layout()
    plt.show()

# %%
# Create MIP views
if volume is not None:
    create_mip_views(volume, patient_id, sex, age)

# %%
# Cell 6: Volume Statistics and Feature Extraction
def extract_volume_features(volume):
    """Extract basic features from 3D volume."""
    
    features = {}
    
    # Basic statistics
    features['volume_shape'] = volume.shape
    features['total_voxels'] = volume.size
    features['min_intensity'] = float(volume.min())
    features['max_intensity'] = float(volume.max())
    features['mean_intensity'] = float(volume.mean())
    features['std_intensity'] = float(volume.std())
    features['median_intensity'] = float(np.median(volume))
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        features[f'p{p}_intensity'] = float(np.percentile(volume, p))
    
    # Volume statistics
    features['volume_range'] = features['max_intensity'] - features['min_intensity']
    features['coefficient_variation'] = features['std_intensity'] / features['mean_intensity'] if features['mean_intensity'] != 0 else 0
    
    # Entropy (measure of randomness)
    hist, _ = np.histogram(volume, bins=50)
    hist = hist[hist > 0]  # Remove zero bins
    if len(hist) > 0:
        hist_norm = hist / hist.sum()
        features['entropy'] = float(-np.sum(hist_norm * np.log2(hist_norm)))
    else:
        features['entropy'] = 0.0
    
    return features

# %%
# Extract features from the sample volume
if volume is not None:
    print("Extracting volume features...")
    volume_features = extract_volume_features(volume)
    
    print("\nVolume Features:")
    for key, value in volume_features.items():
        print(f"  {key}: {value}")
    
    # Create a summary DataFrame
    features_df = pd.DataFrame([volume_features])
    print(f"\nFeatures DataFrame shape: {features_df.shape}")

# %%
# Cell 7: Multiple Volume Comparison
def compare_volumes(sample_volumes, max_volumes=3):
    """Compare multiple volumes side by side."""
    
    volumes_to_compare = sample_volumes[:max_volumes]
    
    fig, axes = plt.subplots(len(volumes_to_compare), 3, figsize=(18, 6*len(volumes_to_compare)))
    
    for i, sample in enumerate(volumes_to_compare):
        try:
            volume, _ = load_3d_volume(sample['file_path'])
            if volume is None:
                continue
                
            patient_id = sample['patient_id']
            sex = sample.get('sex', 'Unknown')
            age = sample.get('age', 'Unknown')
            
            # Create MIP views
            axial_mip = np.max(volume, axis=0)
            coronal_mip = np.max(volume, axis=1)
            sagittal_mip = np.max(volume, axis=2)
            
            # Plot
            if len(volumes_to_compare) == 1:
                ax_row = axes
            else:
                ax_row = axes[i]
            
            ax_row[0].imshow(axial_mip, cmap='hot', aspect='equal')
            ax_row[0].set_title(f'Patient {patient_id} - Axial MIP\n{sex}, {age} years')
            ax_row[0].set_xlabel('Left-Right')
            ax_row[0].set_ylabel('Anterior-Posterior')
            
            ax_row[1].imshow(coronal_mip, cmap='hot', aspect='equal')
            ax_row[1].set_title(f'Patient {patient_id} - Coronal MIP\n{sex}, {age} years')
            ax_row[1].set_xlabel('Left-Right')
            ax_row[1].set_ylabel('Superior-Inferior')
            
            ax_row[2].imshow(sagittal_mip, cmap='hot', aspect='equal')
            ax_row[2].set_title(f'Patient {patient_id} - Sagittal MIP\n{sex}, {age} years')
            ax_row[2].set_xlabel('Anterior-Posterior')
            ax_row[2].set_ylabel('Superior-Inferior')
            
        except Exception as e:
            print(f"Error processing volume {i}: {e}")
    
    plt.tight_layout()
    plt.show()

# %%
# Compare multiple volumes
print("Comparing multiple volumes...")
compare_volumes(sample_volumes, max_volumes=3)

# %%
# Cell 8: Summary and Next Steps
print("🎉 3D Volume Analysis Complete! 🎉")
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

# %%
if volume is not None:
    print(f"📊 Volume Analysis:")
    print(f"  Shape: {volume.shape}")
    print(f"  Total voxels: {volume.size:,}")
    print(f"  Intensity range: [{volume.min():.1f}, {volume.max():.1f}]")
    print(f"  Mean intensity: {volume.mean():.1f}")
    print(f"  Standard deviation: {volume.std():.1f}")

# %%
print(f"\n🔬 Analysis Completed:")
print(f"  Multi-planar visualization")
print(f"  Maximum Intensity Projections")
print(f"  Volume feature extraction")
print(f"  Multi-volume comparison")

# %%
print("\n🚀 Next Steps:")
print("1. Implement SBR calculation for striatal regions")
print("2. Add texture and shape features")
print("3. Develop 3D CNN preprocessing pipeline")
print("4. Implement patient-level cross-validation")
