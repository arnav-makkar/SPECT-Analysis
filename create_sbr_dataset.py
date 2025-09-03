#!/usr/bin/env python3
"""
PPMI SBR Dataset Creation Script
Creates a comprehensive dataset with Striatal Binding Ratio (SBR) calculations
for all control and PD patients
"""

import os
import pydicom
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class SBRCalculator:
    def __init__(self, control_dir="control", pd_dir="PD"):
        self.control_dir = Path(control_dir)
        self.pd_dir = Path(pd_dir)
        self.patient_data = []
        
    def load_dicom_volume(self, dcm_file):
        """Load DICOM file and extract 3D volume data"""
        try:
            ds = pydicom.dcmread(dcm_file)
            
            if hasattr(ds, 'pixel_array'):
                volume_data = ds.pixel_array
                print(f"Loaded volume: {volume_data.shape}")
                return volume_data, ds
            else:
                print(f"No pixel data found in {dcm_file}")
                return None, None
                
        except Exception as e:
            print(f"Error loading {dcm_file}: {e}")
            return None, None
    
    def extract_central_slice(self, volume_data):
        """Extract central slice from 3D volume for ROI analysis"""
        if volume_data is None:
            return None
            
        if volume_data.ndim == 3:
            # Extract middle slice from the third dimension
            middle_slice_idx = volume_data.shape[2] // 2
            central_slice = volume_data[:, :, middle_slice_idx]
            return central_slice, middle_slice_idx
        elif volume_data.ndim == 2:
            return volume_data, 0
        else:
            print(f"Unexpected data shape: {volume_data.shape}")
            return None, None
    
    def define_rois_automatically(self, central_slice):
        """Automatically define ROIs based on typical brain anatomy"""
        # These coordinates are approximate and may need adjustment
        # Based on typical 91x109 brain slice dimensions
        
        height, width = central_slice.shape
        
        # Define ROI coordinates [x_start, y_start, width, height]
        roi_coords = {
            "L_Putamen": [int(width*0.35), int(height*0.45), int(width*0.12), int(height*0.15)],
            "R_Putamen": [int(width*0.65), int(height*0.45), int(width*0.12), int(height*0.15)],
            "L_Caudate": [int(width*0.30), int(height*0.40), int(width*0.10), int(height*0.12)],
            "R_Caudate": [int(width*0.70), int(height*0.40), int(width*0.10), int(height*0.12)],
            "Occipital": [int(width*0.40), int(height*0.80), int(width*0.20), int(height*0.15)]
        }
        
        return roi_coords
    
    def calculate_roi_intensities(self, central_slice, roi_coords):
        """Calculate average intensities for each ROI"""
        intensities = {}
        
        for roi_name, (x, y, w, h) in roi_coords.items():
            # Ensure coordinates are within bounds
            x = max(0, min(x, central_slice.shape[1] - w))
            y = max(0, min(y, central_slice.shape[0] - h))
            
            # Extract ROI data
            roi_data = central_slice[y:y+h, x:x+w]
            
            # Calculate statistics
            intensities[roi_name] = {
                'mean': np.mean(roi_data),
                'std': np.std(roi_data),
                'min': np.min(roi_data),
                'max': np.max(roi_data),
                'coordinates': [x, y, w, h]
            }
        
        return intensities
    
    def calculate_sbr(self, roi_intensities):
        """Calculate Striatal Binding Ratio (SBR) using occipital as reference"""
        sbr_values = {}
        
        # Get reference region (occipital)
        occipital_mean = roi_intensities['Occipital']['mean']
        
        if occipital_mean <= 0:
            print("Warning: Occipital reference value is too low")
            return None
        
        # Calculate SBR for each striatal region
        striatal_regions = ['L_Putamen', 'R_Putamen', 'L_Caudate', 'R_Caudate']
        
        for region in striatal_regions:
            if region in roi_intensities:
                target_mean = roi_intensities[region]['mean']
                sbr = (target_mean / occipital_mean) - 1
                sbr_values[f'{region}_SBR'] = sbr
        
        # Calculate average SBR for left and right sides
        if 'L_Putamen_SBR' in sbr_values and 'L_Caudate_SBR' in sbr_values:
            sbr_values['L_Striatum_SBR'] = (sbr_values['L_Putamen_SBR'] + sbr_values['L_Caudate_SBR']) / 2
        
        if 'R_Putamen_SBR' in sbr_values and 'R_Caudate_SBR' in sbr_values:
            sbr_values['R_Striatum_SBR'] = (sbr_values['R_Putamen_SBR'] + sbr_values['R_Caudate_SBR']) / 2
        
        # Overall striatal SBR
        if 'L_Striatum_SBR' in sbr_values and 'R_Striatum_SBR' in sbr_values:
            sbr_values['Overall_Striatum_SBR'] = (sbr_values['L_Striatum_SBR'] + sbr_values['R_Striatum_SBR']) / 2
        
        return sbr_values
    
    def visualize_rois(self, central_slice, roi_coords, roi_intensities, title="ROI Analysis", save_path=None):
        """Visualize ROIs on the brain slice"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original slice with ROIs
        im1 = axes[0].imshow(central_slice, cmap='gray', aspect='equal')
        axes[0].set_title(f'ROI Placement\n{title}')
        axes[0].axis('off')
        
        # Draw ROIs
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (roi_name, (x, y, w, h)) in enumerate(roi_coords.items()):
            color = colors[i % len(colors)]
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            axes[0].add_patch(rect)
            axes[0].text(x, y-2, roi_name, color=color, fontsize=10, fontweight='bold')
        
        # ROI intensity bar chart
        roi_names = list(roi_intensities.keys())
        roi_means = [roi_intensities[name]['mean'] for name in roi_names]
        
        bars = axes[1].bar(range(len(roi_names)), roi_means, color=colors[:len(roi_names)])
        axes[1].set_title('ROI Average Intensities')
        axes[1].set_xlabel('Brain Regions')
        axes[1].set_ylabel('Pixel Intensity')
        axes[1].set_xticks(range(len(roi_names)))
        axes[1].set_xticklabels(roi_names, rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, roi_means):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved ROI visualization to: {save_path}")
        
        plt.show()
    
    def process_all_patients(self):
        """Process all patients and calculate SBR values"""
        print("Processing all patients for SBR calculation...")
        
        # Process control group
        print("\n=== Processing Control Group ===")
        control_files = list((self.control_dir / "PPMI").rglob("*.dcm"))
        
        for i, dcm_file in enumerate(control_files):
            print(f"\nProcessing control patient {i+1}/{len(control_files)}: {dcm_file.name}")
            
            # Load DICOM volume
            volume_data, ds = self.load_dicom_volume(dcm_file)
            if volume_data is None:
                continue
            
            # Extract central slice
            central_slice, slice_idx = self.extract_central_slice(volume_data)
            if central_slice is None:
                continue
            
            # Define ROIs
            roi_coords = self.define_rois_automatically(central_slice)
            
            # Calculate ROI intensities
            roi_intensities = self.calculate_roi_intensities(central_slice, roi_coords)
            
            # Calculate SBR values
            sbr_values = self.calculate_sbr(roi_intensities)
            if sbr_values is None:
                continue
            
            # Store patient data
            patient_info = {
                'PatientID': ds.get('PatientID', f'Control_{i+1}'),
                'Group': 'Control',
                'Diagnosis': 0,  # 0 for control
                'DICOM_File': str(dcm_file),
                'Volume_Shape': str(volume_data.shape),
                'Central_Slice_Index': slice_idx,
                'Slice_Shape': str(central_slice.shape)
            }
            
            # Add ROI intensities
            for roi_name, stats in roi_intensities.items():
                patient_info[f'{roi_name}_Mean'] = stats['mean']
                patient_info[f'{roi_name}_Std'] = stats['std']
                patient_info[f'{roi_name}_Min'] = stats['min']
                patient_info[f'{roi_name}_Max'] = stats['max']
            
            # Add SBR values
            patient_info.update(sbr_values)
            
            self.patient_data.append(patient_info)
            
            # Visualize first few patients
            if i < 3:
                save_path = f"control_patient_{i+1}_roi_analysis.png"
                self.visualize_rois(central_slice, roi_coords, roi_intensities, 
                                   f"Control Patient {i+1}", save_path)
        
        # Process PD group
        print("\n=== Processing PD Group ===")
        pd_files = list((self.pd_dir / "PPMI").rglob("*.dcm"))
        
        for i, dcm_file in enumerate(pd_files):
            print(f"\nProcessing PD patient {i+1}/{len(pd_files)}: {dcm_file.name}")
            
            # Load DICOM volume
            volume_data, ds = self.load_dicom_volume(dcm_file)
            if volume_data is None:
                continue
            
            # Extract central slice
            central_slice, slice_idx = self.extract_central_slice(volume_data)
            if central_slice is None:
                continue
            
            # Define ROIs
            roi_coords = self.define_rois_automatically(central_slice)
            
            # Calculate ROI intensities
            roi_intensities = self.calculate_roi_intensities(central_slice, roi_coords)
            
            # Calculate SBR values
            sbr_values = self.calculate_sbr(roi_intensities)
            if sbr_values is None:
                continue
            
            # Store patient data
            patient_info = {
                'PatientID': ds.get('PatientID', f'PD_{i+1}'),
                'Group': 'PD',
                'Diagnosis': 1,  # 1 for PD
                'DICOM_File': str(dcm_file),
                'Volume_Shape': str(volume_data.shape),
                'Central_Slice_Index': slice_idx,
                'Slice_Shape': str(central_slice.shape)
            }
            
            # Add ROI intensities
            for roi_name, stats in roi_intensities.items():
                patient_info[f'{roi_name}_Mean'] = stats['mean']
                patient_info[f'{roi_name}_Std'] = stats['std']
                patient_info[f'{roi_name}_Min'] = stats['min']
                patient_info[f'{roi_name}_Max'] = stats['max']
            
            # Add SBR values
            patient_info.update(sbr_values)
            
            self.patient_data.append(patient_info)
            
            # Visualize first few patients
            if i < 3:
                save_path = f"pd_patient_{i+1}_roi_analysis.png"
                self.visualize_rois(central_slice, roi_coords, roi_intensities, 
                                   f"PD Patient {i+1}", save_path)
        
        print(f"\nTotal patients processed: {len(self.patient_data)}")
        return self.patient_data
    
    def save_dataset(self, filename="ppmi_sbr_dataset.csv"):
        """Save the complete dataset to CSV"""
        if not self.patient_data:
            print("No data to save. Run process_all_patients() first.")
            return
        
        df = pd.DataFrame(self.patient_data)
        
        # Reorder columns for better readability
        column_order = [
            'PatientID', 'Group', 'Diagnosis', 'DICOM_File', 'Volume_Shape', 
            'Central_Slice_Index', 'Slice_Shape'
        ]
        
        # Add ROI intensity columns
        roi_columns = [col for col in df.columns if col.endswith('_Mean') and not col.startswith('Overall')]
        column_order.extend(sorted(roi_columns))
        
        # Add SBR columns
        sbr_columns = [col for col in df.columns if col.endswith('_SBR')]
        column_order.extend(sorted(sbr_columns))
        
        # Add remaining columns
        remaining_columns = [col for col in df.columns if col not in column_order]
        column_order.extend(remaining_columns)
        
        # Reorder and save
        df_ordered = df[column_order]
        df_ordered.to_csv(filename, index=False)
        print(f"Dataset saved to: {filename}")
        
        # Print summary statistics
        print(f"\nDataset Summary:")
        print(f"Total patients: {len(df_ordered)}")
        print(f"Control patients: {len(df_ordered[df_ordered['Group'] == 'Control'])}")
        print(f"PD patients: {len(df_ordered[df_ordered['Group'] == 'PD'])}")
        
        # Print SBR statistics by group
        sbr_cols = [col for col in df_ordered.columns if col.endswith('_SBR')]
        print(f"\nSBR Statistics by Group:")
        for sbr_col in sbr_cols:
            control_sbr = df_ordered[df_ordered['Group'] == 'Control'][sbr_col].mean()
            pd_sbr = df_ordered[df_ordered['Group'] == 'PD'][sbr_col].mean()
            print(f"  {sbr_col}: Control={control_sbr:.3f}, PD={pd_sbr:.3f}")
        
        return df_ordered

def main():
    """Main function to create the SBR dataset"""
    print("PPMI SBR Dataset Creation")
    print("=" * 50)
    
    # Initialize calculator
    calculator = SBRCalculator()
    
    # Process all patients
    patient_data = calculator.process_all_patients()
    
    # Save dataset
    if patient_data:
        df = calculator.save_dataset()
        print(f"\nDataset created successfully with {len(df)} patients!")
        print(f"Columns: {list(df.columns)}")
    else:
        print("No data was processed. Check your DICOM files.")

if __name__ == "__main__":
    main()
