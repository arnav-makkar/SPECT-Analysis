"""
Custom PPMI data loader for the specific data structure provided by the user.
Handles PPMI_Data 1, PPMI_Data 2 folders and the specific CSV files.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPMIDataLoader:
    """Custom loader for PPMI data structure."""
    
    def __init__(self, root_path: str = "."):
        """Initialize PPMI data loader.
        
        Args:
            root_path: Root directory containing PPMI_Data folders and CSV files
        """
        self.root_path = Path(root_path)
        
        # Try to find the data directories by looking in parent directories if needed
        if not (self.root_path / "PPMI_Data 1").exists() and not (self.root_path / "PPMI_Data 2").exists():
            # Look in parent directories
            current = self.root_path
            for _ in range(3):  # Look up to 3 levels up
                if (current / "PPMI_Data 1").exists() or (current / "PPMI_Data 2").exists():
                    self.root_path = current
                    break
                current = current.parent
        
        self.ppmi_data_1 = self.root_path / "PPMI_Data 1"
        self.ppmi_data_2 = self.root_path / "PPMI_Data 2"
        self.metadata_folder = self.root_path / "metadata"
        
        # Clinical data files
        self.ida_search_file = self.root_path / "idaSearch_9_03_2025.csv"
        self.datscan_imaging_file = self.root_path / "DaTscan_Imaging_03Sep2025.csv"
        
    def load_clinical_data(self) -> Dict[str, pd.DataFrame]:
        """Load clinical data from CSV files.
        
        Returns:
            Dictionary containing clinical data DataFrames
        """
        logger.info("Loading PPMI clinical data...")
        
        clinical_data = {}
        
        # Load idaSearch data
        if self.ida_search_file.exists():
            logger.info(f"Loading {self.ida_search_file}")
            ida_df = pd.read_csv(self.ida_search_file)
            clinical_data['ida_search'] = ida_df
            logger.info(f"Loaded idaSearch: {ida_df.shape}")
        else:
            logger.warning(f"idaSearch file not found: {self.ida_search_file}")
            
        # Load DaTscan imaging data
        if self.datscan_imaging_file.exists():
            logger.info(f"Loading {self.datscan_imaging_file}")
            datscan_df = pd.read_csv(self.datscan_imaging_file)
            clinical_data['datscan_imaging'] = datscan_df
            logger.info(f"Loaded DaTscan imaging: {datscan_df.shape}")
        else:
            logger.warning(f"DaTscan imaging file not found: {self.datscan_imaging_file}")
            
        return clinical_data
    
    def find_dicom_files(self) -> List[Path]:
        """Find all DICOM files in PPMI_Data folders.
        
        Returns:
            List of paths to DICOM files
        """
        dicom_files = []
        
        # Search in PPMI_Data 1
        if self.ppmi_data_1.exists():
            logger.info(f"Searching for DICOM files in {self.ppmi_data_1}")
            for dicom_file in self.ppmi_data_1.rglob("*.dcm"):
                dicom_files.append(dicom_file)
                
        # Search in PPMI_Data 2
        if self.ppmi_data_2.exists():
            logger.info(f"Searching for DICOM files in {self.ppmi_data_2}")
            for dicom_file in self.ppmi_data_2.rglob("*.dcm"):
                dicom_files.append(dicom_file)
                
        logger.info(f"Found {len(dicom_files)} DICOM files")
        return dicom_files
    
    def extract_patient_info_from_path(self, dicom_path: Path) -> Dict[str, str]:
        """Extract patient information from DICOM file path.
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            Dictionary with patient information
        """
        path_str = str(dicom_path)
        
        # Extract patient ID from path
        # Pattern: PPMI_Data X/100XXX/.../PPMI_100XXX_...
        patient_id_match = re.search(r'PPMI_Data\s*[12]/(\d+)/', path_str)
        patient_id = patient_id_match.group(1) if patient_id_match else "unknown"
        
        # Extract additional info from filename
        filename = dicom_path.name
        study_info = {}
        
        # Parse PPMI filename: PPMI_100001_NM_AC-018100001-4-DAT-G6N1__br_raw_20240926190052673_1.dcm
        if filename.startswith("PPMI_"):
            parts = filename.split("_")
            if len(parts) >= 3:
                study_info['patient_id_from_filename'] = parts[1]
                study_info['modality'] = parts[2] if len(parts) > 2 else "unknown"
                study_info['acquisition_info'] = parts[3] if len(parts) > 3 else "unknown"
                
        return {
            'patient_id': patient_id,
            'file_path': str(dicom_path),
            'filename': filename,
            'data_folder': 'PPMI_Data 1' if 'PPMI_Data 1' in path_str else 'PPMI_Data 2',
            **study_info
        }
    
    def create_image_mapping(self) -> pd.DataFrame:
        """Create mapping between DICOM files and patient information.
        
        Returns:
            DataFrame with image mapping
        """
        logger.info("Creating image-to-patient mapping...")
        
        # Find all DICOM files
        dicom_files = self.find_dicom_files()
        
        # Extract information from each file
        mapping_data = []
        for dicom_file in dicom_files:
            try:
                patient_info = self.extract_patient_info_from_path(dicom_file)
                mapping_data.append(patient_info)
            except Exception as e:
                logger.warning(f"Failed to extract info from {dicom_file}: {e}")
                
        # Create DataFrame
        if not mapping_data:
            logger.warning("No DICOM files found! Creating empty DataFrame with expected columns")
            mapping_df = pd.DataFrame(columns=[
                'patient_id', 'file_path', 'filename', 'data_folder', 
                'patient_id_from_filename', 'modality', 'acquisition_info'
            ])
        else:
            mapping_df = pd.DataFrame(mapping_data)
            # Clean up patient IDs
            mapping_df['patient_id'] = mapping_df['patient_id'].astype(str)
        
        logger.info(f"Created mapping for {len(mapping_df)} images")
        if len(mapping_df) > 0:
            logger.info(f"Unique patients: {mapping_df['patient_id'].nunique()}")
        else:
            logger.warning("No images found - dataset will be empty")
        
        return mapping_df
    
    def merge_with_clinical_data(self, mapping_df: pd.DataFrame) -> pd.DataFrame:
        """Merge image mapping with clinical data.
        
        Args:
            mapping_df: DataFrame with image mapping
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging with clinical data...")
        
        # Load clinical data
        clinical_data = self.load_clinical_data()
        
        # Start with image mapping
        merged_df = mapping_df.copy()
        
        # If mapping_df is empty, return early
        if len(merged_df) == 0:
            logger.warning("No images to merge with clinical data")
            return merged_df
        
        # Merge with idaSearch data
        if 'ida_search' in clinical_data:
            ida_df = clinical_data['ida_search']
            
            # Clean up patient ID in idaSearch
            ida_df['patient_id'] = ida_df['Subject ID'].astype(str)
            
            # Merge on patient ID
            merged_df = merged_df.merge(
                ida_df[['patient_id', 'Sex', 'Age', 'Description']], 
                on='patient_id', 
                how='left'
            )
            
            # Rename columns
            merged_df = merged_df.rename(columns={
                'Sex': 'sex',
                'Age': 'age',
                'Description': 'description'
            })
            
        # Merge with DaTscan imaging data
        if 'datscan_imaging' in clinical_data:
            datscan_df = clinical_data['datscan_imaging']
            
            # Clean up patient ID in DaTscan imaging
            datscan_df['patient_id'] = datscan_df['PATNO'].astype(str)
            
            # Get unique patient info from DaTscan imaging
            patient_datscan = datscan_df[['patient_id', 'EVENT_ID', 'INFODT']].drop_duplicates()
            
            # Merge on patient ID
            merged_df = merged_df.merge(
                patient_datscan, 
                on='patient_id', 
                how='left'
            )
            
        # Add basic metadata
        merged_df['image_type'] = 'spect'
        merged_df['modality'] = 'NM'  # Nuclear Medicine
        
        logger.info(f"Final merged dataset: {len(merged_df)} images")
        
        return merged_df
    
    def get_dataset_summary(self, merged_df: pd.DataFrame) -> Dict:
        """Get summary statistics for the dataset.
        
        Args:
            merged_df: Merged DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_images': len(merged_df),
            'unique_patients': merged_df['patient_id'].nunique() if len(merged_df) > 0 else 0,
            'data_folders': merged_df['data_folder'].value_counts().to_dict() if len(merged_df) > 0 else {},
            'modalities': merged_df['modality'].value_counts().to_dict() if len(merged_df) > 0 else {}
        }
        
        # Add clinical data summary if available
        if 'sex' in merged_df.columns and len(merged_df) > 0:
            summary['sex_distribution'] = merged_df['sex'].value_counts().to_dict()
            
        if 'age' in merged_df.columns:
            age_data = pd.to_numeric(merged_df['age'], errors='coerce').dropna()
            if len(age_data) > 0:
                summary['age_stats'] = {
                    'mean': float(age_data.mean()),
                    'std': float(age_data.std()),
                    'min': float(age_data.min()),
                    'max': float(age_data.max())
                }
                
        return summary
    
    def save_mapping(self, merged_df: pd.DataFrame, output_path: str = "data/metadata"):
        """Save the merged mapping to files.
        
        Args:
            merged_df: Merged DataFrame
            output_path: Output directory
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_file = output_path / "ppmi_image_mapping.csv"
        merged_df.to_csv(csv_file, index=False)
        logger.info(f"Saved image mapping to {csv_file}")
        
        # Save summary
        summary = self.get_dataset_summary(merged_df)
        summary_file = output_path / "ppmi_dataset_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("PPMI Dataset Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Total images: {summary['total_images']}\n")
            f.write(f"Unique patients: {summary['unique_patients']}\n\n")
            
            f.write("Data folder distribution:\n")
            for folder, count in summary['data_folders'].items():
                f.write(f"  {folder}: {count} images\n")
                
            f.write(f"\nModality distribution:\n")
            for modality, count in summary['modalities'].items():
                f.write(f"  {modality}: {count} images\n")
                
            if 'sex_distribution' in summary:
                f.write(f"\nSex distribution:\n")
                for sex, count in summary['sex_distribution'].items():
                    f.write(f"  {sex}: {count} images\n")
                    
            if 'age_stats' in summary:
                age_stats = summary['age_stats']
                f.write(f"\nAge statistics:\n")
                f.write(f"  Mean: {age_stats['mean']:.1f} years\n")
                f.write(f"  Std: {age_stats['std']:.1f} years\n")
                f.write(f"  Range: {age_stats['min']:.1f} - {age_stats['max']:.1f} years\n")
                
        logger.info(f"Saved dataset summary to {summary_file}")
        
        return summary


def load_ppmi_data(root_path: str = ".") -> Tuple[pd.DataFrame, Dict]:
    """Convenience function to load PPMI data.
    
    Args:
        root_path: Root directory containing PPMI data
        
    Returns:
        Tuple of (merged_dataframe, summary_statistics)
    """
    loader = PPMIDataLoader(root_path)
    
    # Create image mapping
    mapping_df = loader.create_image_mapping()
    
    # Merge with clinical data
    merged_df = loader.merge_with_clinical_data(mapping_df)
    
    # Get summary
    summary = loader.get_dataset_summary(merged_df)
    
    # Save mapping
    loader.save_mapping(merged_df)
    
    return merged_df, summary
