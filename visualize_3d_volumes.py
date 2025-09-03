#!/usr/bin/env python3
"""
Visualize 3D SPECT volumes from PPMI dataset.
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Visualize 3D SPECT volumes."""
    
    logger.info("🔍 Starting PPMI 3D Volume Visualization")
    logger.info("=" * 50)
    
    try:
        # Load the mapping data
        mapping_file = Path("data/metadata/ppmi_image_mapping.csv")
        if not mapping_file.exists():
            logger.error("Mapping file not found. Please run the data loading first.")
            return False
            
        mapping_df = pd.read_csv(mapping_file)
        logger.info(f"✅ Loaded mapping data: {mapping_df.shape}")
        
        # Select a few sample images from different patients
        sample_images = []
        
        # Get one image from each patient
        for patient_id in mapping_df['patient_id'].unique():
            patient_images = mapping_df[mapping_df['patient_id'] == patient_id]
            sample_images.append(patient_images.iloc[0])
            
        logger.info(f"Selected {len(sample_images)} sample volumes for visualization")
        
        # Visualize each sample
        for i, sample in enumerate(sample_images):
            try:
                visualize_3d_volume(sample, i + 1)
            except Exception as e:
                logger.warning(f"Failed to visualize volume {i+1}: {e}")
                
        logger.info("✅ 3D volume visualization complete!")
        return True
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_3d_volume(sample_row, volume_num):
    """Visualize a 3D SPECT volume."""
    
    file_path = sample_row['file_path']
    patient_id = sample_row['patient_id']
    age = sample_row.get('age', 'Unknown')
    sex = sample_row.get('sex', 'Unknown')
    
    logger.info(f"Visualizing volume {volume_num}: Patient {patient_id} ({sex}, {age} years)")
    
    try:
        # Load DICOM
        ds = pydicom.dcmread(file_path)
        
        # Get pixel data
        pixel_array = ds.pixel_array
        
        logger.info(f"  Volume shape: {pixel_array.shape}")
        logger.info(f"  Intensity range: [{pixel_array.min()}, {pixel_array.max()}]")
        logger.info(f"  Mean intensity: {pixel_array.mean():.1f}")
        
        # Create multi-slice visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Create a grid of images showing different slices
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(3, 4),
                        axes_pad=0.1,
                        label_mode="L",
                        share_all=True)
        
        # Show different slices through the volume
        slices_to_show = [
            (pixel_array.shape[0]//4, "Anterior"),
            (pixel_array.shape[0]//2, "Central"),
            (3*pixel_array.shape[0]//4, "Posterior")
        ]
        
        slice_idx = 0
        for row in range(3):
            for col in range(4):
                if slice_idx < len(slices_to_show):
                    slice_num, slice_name = slices_to_show[row]
                    
                    # Get the slice
                    if row == 0:  # Sagittal view
                        slice_data = pixel_array[slice_num, :, :]
                        view_name = "Sagittal"
                    elif row == 1:  # Coronal view
                        slice_data = pixel_array[:, slice_num, :]
                        view_name = "Coronal"
                    else:  # Axial view
                        slice_data = pixel_array[:, :, slice_num]
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
                                       f'Shape: {pixel_array.shape}\n'
                                       f'Min: {pixel_array.min():.1f}\n'
                                       f'Max: {pixel_array.max():.1f}\n'
                                       f'Mean: {pixel_array.mean():.1f}\n'
                                       f'Std: {pixel_array.std():.1f}',
                                       ha='center', va='center',
                                       transform=grid[slice_idx].transAxes,
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    grid[slice_idx].set_title('Volume Statistics')
                    slice_idx += 1
        
        # Main title
        fig.suptitle(f'Patient {patient_id} - 3D SPECT Volume Analysis\n{sex}, {age} years', 
                    fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path("results/plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"ppmi_3d_volume_{patient_id}.png", dpi=300, bbox_inches='tight')
        
        # Show DICOM metadata
        logger.info(f"  DICOM metadata:")
        logger.info(f"    Modality: {getattr(ds, 'Modality', 'N/A')}")
        logger.info(f"    Image size: {getattr(ds, 'Rows', 'N/A')} x {getattr(ds, 'Columns', 'N/A')}")
        if hasattr(ds, 'PixelSpacing'):
            logger.info(f"    Pixel spacing: {ds.PixelSpacing}")
        if hasattr(ds, 'SliceThickness'):
            logger.info(f"    Slice thickness: {ds.SliceThickness}")
            
        plt.show()
        
        # Also create a simple 3D projection view
        create_3d_projection(pixel_array, patient_id, sex, age)
        
    except Exception as e:
        logger.error(f"Failed to load DICOM {file_path}: {e}")


def create_3d_projection(volume, patient_id, sex, age):
    """Create a 3D projection view of the volume."""
    
    try:
        # Create maximum intensity projection (MIP) views
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
        
        # Save plot
        output_dir = Path("results/plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"ppmi_3d_mip_{patient_id}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except Exception as e:
        logger.warning(f"Failed to create 3D projection for patient {patient_id}: {e}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
