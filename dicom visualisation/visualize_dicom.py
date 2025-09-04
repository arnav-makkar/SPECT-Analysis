#!/usr/bin/env python3
"""
PPMI DaTSCAN DICOM Visualization Script
Loads and displays DICOM files with brain imaging visualization
"""

import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class DICOMVisualizer:
    def __init__(self, control_dir="control", pd_dir="PD"):
        self.control_dir = Path(control_dir)
        self.pd_dir = Path(pd_dir)
        
    def load_dicom_volume(self, dcm_file):
        """Load DICOM file and extract image data"""
        try:
            ds = pydicom.dcmread(dcm_file)
            
            # Extract image data
            if hasattr(ds, 'pixel_array'):
                image_data = ds.pixel_array
            else:
                print(f"No pixel data found in {dcm_file}")
                return None, None
            
            # Get metadata
            metadata = {
                'modality': getattr(ds, 'Modality', 'Unknown'),
                'image_size': (getattr(ds, 'Rows', 0), getattr(ds, 'Columns', 0)),
                'pixel_spacing': getattr(ds, 'PixelSpacing', [1, 1]),
                'slice_thickness': getattr(ds, 'SliceThickness', 1),
                'study_date': getattr(ds, 'StudyDate', 'Unknown'),
                'patient_id': getattr(ds, 'PatientID', 'Unknown')
            }
            
            return image_data, metadata
            
        except Exception as e:
            print(f"Error loading {dcm_file}: {e}")
            return None, None
    
    def extract_central_slice(self, volume_data):
        """Extract central slice from 3D volume for 2D visualization"""
        if volume_data is None:
            return None
            
        if volume_data.ndim == 3:
            # Extract middle slice from the third dimension
            middle_slice_idx = volume_data.shape[2] // 2
            central_slice = volume_data[:, :, middle_slice_idx]
            print(f"Extracted central slice {middle_slice_idx} from 3D volume {volume_data.shape}")
            return central_slice
        elif volume_data.ndim == 2:
            return volume_data
        else:
            print(f"Unexpected data shape: {volume_data.shape}")
            return None
    
    def normalize_image(self, image_data):
        """Normalize image data for better visualization"""
        if image_data is None:
            return None
            
        # Remove any negative values and normalize to 0-1
        image_data = np.maximum(image_data, 0)
        
        if np.max(image_data) > 0:
            image_data = image_data / np.max(image_data)
        
        return image_data
    
    def create_brain_overlay(self, image_data, metadata):
        """Create brain visualization with activity overlay"""
        if image_data is None:
            return None
            
        # Normalize the image
        normalized_data = self.normalize_image(image_data)
        
        # Create a colormap similar to the reference image
        # Blue (low activity) to Red (high activity)
        colors = ['darkblue', 'blue', 'lightblue', 'green', 'yellow', 'orange', 'red']
        n_bins = 100
        cmap = mcolors.LinearSegmentedColormap.from_list('brain_cmap', colors, N=n_bins)
        
        # Apply some smoothing for better visualization
        if normalized_data.ndim == 2:
            smoothed_data = ndimage.gaussian_filter(normalized_data, sigma=0.5)
        else:
            smoothed_data = normalized_data
        
        return smoothed_data, cmap
    
    def visualize_single_dicom(self, dcm_file, title="DICOM Image", save_path=None):
        """Visualize a single DICOM file"""
        print(f"Loading: {os.path.basename(dcm_file)}")
        
        # Load DICOM data
        volume_data, metadata = self.load_dicom_volume(dcm_file)
        
        if volume_data is None:
            print("Failed to load DICOM data")
            return
        
        # Extract central slice for 2D visualization
        image_data = self.extract_central_slice(volume_data)
        
        if image_data is None:
            print("Failed to extract 2D slice")
            return
        
        # Create brain overlay
        overlay_data, cmap = self.create_brain_overlay(image_data, metadata)
        
        if overlay_data is None:
            print("Failed to create overlay")
            return
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        im1 = axes[0].imshow(image_data, cmap='gray', aspect='equal')
        axes[0].set_title(f'Original Image\n{image_data.shape[0]}x{image_data.shape[1]} pixels')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # Brain overlay
        im2 = axes[1].imshow(overlay_data, cmap=cmap, aspect='equal')
        axes[1].set_title(f'Brain Activity Overlay\n{title}')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], shrink=0.8, label='Activity Level')
        
        # Add metadata
        fig.suptitle(f'DICOM Visualization: {title}', fontsize=16, fontweight='bold')
        
        # Add metadata text
        metadata_text = f"""
        Modality: {metadata['modality']}
        Volume Size: {volume_data.shape}
        Slice Size: {image_data.shape[0]} x {image_data.shape[1]}
        Pixel Spacing: {metadata['pixel_spacing'][0]:.2f} x {metadata['pixel_spacing'][1]:.2f} mm
        Slice Thickness: {metadata['slice_thickness']} mm
        Study Date: {metadata['study_date']}
        Patient ID: {metadata['patient_id']}
        """
        
        fig.text(0.02, 0.02, metadata_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        plt.show()
        
        return overlay_data, cmap
    
    def compare_groups(self, num_samples=3):
        """Compare DICOM images between control and PD groups"""
        print("Comparing Control vs PD DICOM images...")
        
        # Find DICOM files
        control_files = list((self.control_dir / "PPMI").rglob("*.dcm"))[:num_samples]
        pd_files = list((self.pd_dir / "PPMI").rglob("*.dcm"))[:num_samples]
        
        if not control_files or not pd_files:
            print("No DICOM files found")
            return
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, num_samples, figsize=(5*num_samples, 10))
        fig.suptitle('Control vs PD Group Comparison', fontsize=16, fontweight='bold')
        
        # Control group
        for i, dcm_file in enumerate(control_files):
            volume_data, metadata = self.load_dicom_volume(dcm_file)
            if volume_data is not None:
                image_data = self.extract_central_slice(volume_data)
                if image_data is not None:
                    overlay_data, cmap = self.create_brain_overlay(image_data, metadata)
                    if overlay_data is not None:
                        im = axes[0, i].imshow(overlay_data, cmap=cmap, aspect='equal')
                        axes[0, i].set_title(f'Control {i+1}\n{os.path.basename(dcm_file)[:20]}...')
                        axes[0, i].axis('off')
        
        # PD group
        for i, dcm_file in enumerate(pd_files):
            volume_data, metadata = self.load_dicom_volume(dcm_file)
            if volume_data is not None:
                image_data = self.extract_central_slice(volume_data)
                if image_data is not None:
                    overlay_data, cmap = self.create_brain_overlay(image_data, metadata)
                    if overlay_data is not None:
                        im = axes[1, i].imshow(overlay_data, cmap=cmap, aspect='equal')
                        axes[1, i].set_title(f'PD {i+1}\n{os.path.basename(dcm_file)[:20]}...')
                        axes[1, i].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='Activity Level')
        
        plt.tight_layout()
        plt.savefig('control_vs_pd_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved comparison visualization to: control_vs_pd_comparison.png")
        plt.show()
    
    def create_3d_visualization(self, dcm_file, save_path=None):
        """Create a 3D visualization of the brain data"""
        print(f"Creating 3D visualization for: {os.path.basename(dcm_file)}")
        
        volume_data, metadata = self.load_dicom_volume(dcm_file)
        
        if volume_data is None:
            print("Failed to load DICOM data")
            return
        
        # Extract central slice for 2D visualization
        image_data = self.extract_central_slice(volume_data)
        
        if image_data is None:
            print("Failed to extract 2D slice")
            return
        
        # Normalize data
        normalized_data = self.normalize_image(image_data)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(normalized_data.shape[1]), np.arange(normalized_data.shape[0]))
        
        # Plot surface with activity as color
        surf = ax.plot_surface(x, y, normalized_data, 
                              facecolors=plt.cm.viridis(normalized_data),
                              alpha=0.8)
        
        ax.set_title(f'3D Brain Activity Visualization\n{os.path.basename(dcm_file)}')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_zlabel('Activity Level')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved 3D visualization to: {save_path}")
        
        plt.show()

def main():
    """Main function to demonstrate DICOM visualization"""
    visualizer = DICOMVisualizer()
    
    print("PPMI DaTSCAN DICOM Visualization")
    print("=" * 50)
    
    # Find a sample DICOM file from each group
    control_files = list((visualizer.control_dir / "PPMI").rglob("*.dcm"))
    pd_files = list((visualizer.pd_dir / "PPMI").rglob("*.dcm"))
    
    if control_files and pd_files:
        print(f"Found {len(control_files)} control DICOM files and {len(pd_files)} PD DICOM files")
        
        # Visualize sample from control group
        print("\n1. Visualizing Control Group Sample:")
        control_sample = control_files[0]
        visualizer.visualize_single_dicom(control_sample, "Control Subject", "control_sample.png")
        
        # Visualize sample from PD group
        print("\n2. Visualizing PD Group Sample:")
        pd_sample = pd_files[0]
        visualizer.visualize_single_dicom(pd_sample, "PD Subject", "pd_sample.png")
        
        # Create group comparison
        print("\n3. Creating Group Comparison:")
        visualizer.compare_groups(num_samples=3)
        
        # Create 3D visualization
        print("\n4. Creating 3D Visualization:")
        visualizer.create_3d_visualization(control_sample, "control_3d.png")
        
    else:
        print("No DICOM files found. Please check the directory structure.")

if __name__ == "__main__":
    main()
