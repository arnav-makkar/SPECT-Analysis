#!/usr/bin/env python3
"""
Visualize sample DICOM images from PPMI dataset.
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Visualize sample DICOM images."""
    
    logger.info("🔍 Starting PPMI DICOM Visualization")
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
            
        logger.info(f"Selected {len(sample_images)} sample images for visualization")
        
        # Visualize each sample
        for i, sample in enumerate(sample_images):
            try:
                visualize_dicom(sample, i + 1)
            except Exception as e:
                logger.warning(f"Failed to visualize image {i+1}: {e}")
                
        logger.info("✅ DICOM visualization complete!")
        return True
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_dicom(sample_row, image_num):
    """Visualize a single DICOM image."""
    
    file_path = sample_row['file_path']
    patient_id = sample_row['patient_id']
    age = sample_row.get('age', 'Unknown')
    sex = sample_row.get('sex', 'Unknown')
    
    logger.info(f"Visualizing image {image_num}: Patient {patient_id} ({sex}, {age} years)")
    
    try:
        # Load DICOM
        ds = pydicom.dcmread(file_path)
        
        # Get pixel data
        pixel_array = ds.pixel_array
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        im1 = axes[0].imshow(pixel_array, cmap='hot', aspect='equal')
        axes[0].set_title(f'Patient {patient_id} - Original SPECT Image\n{sex}, {age} years')
        axes[0].set_xlabel('X (Left-Right)')
        axes[0].set_ylabel('Y (Anterior-Posterior)')
        plt.colorbar(im1, ax=axes[0], label='Intensity')
        
        # Enhanced image for better visualization
        # Apply some basic enhancement
        enhanced = np.clip(pixel_array * 1.5, 0, pixel_array.max())
        im2 = axes[1].imshow(enhanced, cmap='hot', aspect='equal')
        axes[1].set_title(f'Patient {patient_id} - Enhanced View\nLook for Striatal Regions')
        axes[1].set_xlabel('X (Left-Right)')
        axes[1].set_ylabel('Y (Anterior-Posterior)')
        plt.colorbar(im2, ax=axes[1], label='Intensity')
        
        # Add annotations for expected regions
        height, width = pixel_array.shape
        
        # Approximate striatal regions (these are estimates)
        # Left striatum
        axes[1].plot([width//4, width//3], [height//3, height//2], 'g-', linewidth=2, label='Left Striatum')
        # Right striatum  
        axes[1].plot([2*width//3, 3*width//4], [height//3, height//2], 'b-', linewidth=2, label='Right Striatum')
        
        axes[1].legend()
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path("results/plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"ppmi_sample_{patient_id}.png", dpi=300, bbox_inches='tight')
        
        # Show image info
        logger.info(f"  Image shape: {pixel_array.shape}")
        logger.info(f"  Intensity range: [{pixel_array.min()}, {pixel_array.max()}]")
        logger.info(f"  Mean intensity: {pixel_array.mean():.1f}")
        
        # Show DICOM metadata
        logger.info(f"  DICOM metadata:")
        logger.info(f"    Modality: {getattr(ds, 'Modality', 'N/A')}")
        logger.info(f"    Image size: {getattr(ds, 'Rows', 'N/A')} x {getattr(ds, 'Columns', 'N/A')}")
        if hasattr(ds, 'PixelSpacing'):
            logger.info(f"    Pixel spacing: {ds.PixelSpacing}")
        if hasattr(ds, 'SliceThickness'):
            logger.info(f"    Slice thickness: {ds.SliceThickness}")
            
        plt.show()
        
    except Exception as e:
        logger.error(f"Failed to load DICOM {file_path}: {e}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
