#!/usr/bin/env python3
"""
Quick DICOM Preview Script
Shows a single DICOM image with brain visualization
"""

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import ndimage
from pathlib import Path

def quick_dicom_preview():
    """Quick preview of a DICOM file"""
    
    # Find a DICOM file
    dcm_file = list(Path("control/PPMI").rglob("*.dcm"))[0]
    print(f"Loading: {dcm_file.name}")
    
    # Load DICOM
    ds = pydicom.dcmread(str(dcm_file))
    volume_data = ds.pixel_array
    
    print(f"Volume shape: {volume_data.shape}")
    
    # Extract central slice
    middle_slice = volume_data.shape[2] // 2
    central_slice = volume_data[:, :, middle_slice]
    
    # Normalize for visualization
    normalized_slice = np.maximum(central_slice, 0)
    if np.max(normalized_slice) > 0:
        normalized_slice = normalized_slice / np.max(normalized_slice)
    
    # Create brain colormap (blue to red)
    colors = ['darkblue', 'blue', 'lightblue', 'green', 'yellow', 'orange', 'red']
    cmap = mcolors.LinearSegmentedColormap.from_list('brain_cmap', colors, N=100)
    
    # Apply smoothing
    smoothed_slice = ndimage.gaussian_filter(normalized_slice, sigma=0.5)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original slice
    im1 = axes[0].imshow(central_slice, cmap='gray', aspect='equal')
    axes[0].set_title('Original DICOM Slice\n(Gray Scale)')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # Normalized slice
    im2 = axes[1].imshow(normalized_slice, cmap='gray', aspect='equal')
    axes[1].set_title('Normalized Slice\n(0-1 Range)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # Brain activity overlay
    im3 = axes[2].imshow(smoothed_slice, cmap=cmap, aspect='equal')
    axes[2].set_title('Brain Activity Overlay\n(Blue=Low, Red=High)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], shrink=0.8, label='Activity Level')
    
    fig.suptitle(f'PPMI DaTSCAN Brain Visualization\n{ds.PatientID} - {ds.Modality}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('quick_preview.png', dpi=300, bbox_inches='tight')
    print("Saved quick preview to: quick_preview.png")
    plt.show()
    
    # Print some statistics
    print(f"\nImage Statistics:")
    print(f"  Min value: {np.min(central_slice):.2f}")
    print(f"  Max value: {np.max(central_slice):.2f}")
    print(f"  Mean value: {np.mean(central_slice):.2f}")
    print(f"  Std deviation: {np.std(central_slice):.2f}")
    print(f"  Slice dimensions: {central_slice.shape}")
    print(f"  Volume dimensions: {volume_data.shape}")

if __name__ == "__main__":
    quick_dicom_preview()
