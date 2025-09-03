"""
Data linking utilities to connect PPMI DICOM images with clinical data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPMIDataLinker:
    """Links PPMI DICOM images with clinical data to create training dataset."""
    
    def __init__(self, config):
        """Initialize data linker.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.raw_dicom_path = config.get_data_path('raw')
        self.clinical_data_path = config.get_data_path('clinical')
        self.metadata_path = config.get_data_path('metadata')
        self.processed_path = config.get_data_path('processed')
        
    def create_image_to_patient_mapping(self) -> pd.DataFrame:
        """Create mapping between image paths and patient information.
        
        Returns:
            DataFrame with image paths, patient IDs, and labels
        """
        logger.info("Creating image-to-patient mapping...")
        
        # Get all DICOM series directories
        dicom_series = self._find_dicom_series()
        logger.info(f"Found {len(dicom_series)} DICOM series")
        
        # Extract patient IDs from DICOM paths
        image_patient_mapping = self._extract_patient_ids_from_paths(dicom_series)
        
        # Load clinical data
        clinical_data = self._load_clinical_data()
        
        # Merge image paths with clinical data
        final_mapping = self._merge_with_clinical_data(image_patient_mapping, clinical_data)
        
        # Save mapping
        self._save_image_mapping(final_mapping)
        
        return final_mapping
    
    def _find_dicom_series(self) -> List[Path]:
        """Find all DICOM series directories.
        
        Returns:
            List of paths to DICOM series
        """
        series_dirs = []
        
        for item in self.raw_dicom_path.iterdir():
            if item.is_dir():
                # Check if directory contains DICOM files
                dicom_files = list(item.glob("*.dcm"))
                if dicom_files:
                    series_dirs.append(item)
                    
        return series_dirs
    
    def _extract_patient_ids_from_paths(self, dicom_series: List[Path]) -> pd.DataFrame:
        """Extract patient IDs from DICOM directory paths.
        
        Args:
            dicom_series: List of DICOM series directories
            
        Returns:
            DataFrame with image paths and extracted patient IDs
        """
        mapping_data = []
        
        for series_path in dicom_series:
            # Try to extract patient ID from path
            patient_id = self._extract_patient_id_from_path(series_path)
            
            # Count DICOM files in series
            dicom_files = list(series_path.glob("*.dcm"))
            
            mapping_data.append({
                'series_path': str(series_path),
                'series_name': series_path.name,
                'extracted_patient_id': patient_id,
                'num_dicom_files': len(dicom_files),
                'dicom_files': [str(f) for f in dicom_files]
            })
            
        return pd.DataFrame(mapping_data)
    
    def _extract_patient_id_from_path(self, series_path: Path) -> Optional[str]:
        """Extract patient ID from DICOM series path.
        
        Args:
            series_path: Path to DICOM series
            
        Returns:
            Extracted patient ID or None
        """
        # Common PPMI path patterns
        path_str = str(series_path)
        
        # Pattern 1: Patient ID in directory name
        # Example: /path/to/PPMI_12345_visit1/
        patient_id_patterns = [
            r'PPMI_(\d+)_',  # PPMI_12345_
            r'(\d{5,})',     # 5+ digit number
            r'PAT(\d+)',     # PAT12345
            r'(\d+)_visit',  # 12345_visit
        ]
        
        for pattern in patient_id_patterns:
            match = re.search(pattern, path_str)
            if match:
                return match.group(1)
                
        # Pattern 2: Patient ID in parent directory
        parent_dir = series_path.parent.name
        for pattern in patient_id_patterns:
            match = re.search(pattern, parent_dir)
            if match:
                return match.group(1)
                
        # Pattern 3: Look for patient ID in DICOM metadata
        try:
            from .dicom_loader import DICOMLoader
            loader = DICOMLoader(str(series_path))
            dicom_files = list(series_path.glob("*.dcm"))
            if dicom_files:
                # Load first DICOM file to get metadata
                ds = loader.load_scan(series_path)[0][0]  # Get first slice
                if hasattr(ds, 'PatientID'):
                    return ds.PatientID
        except Exception as e:
            logger.debug(f"Could not extract patient ID from DICOM metadata: {e}")
            
        logger.warning(f"Could not extract patient ID from path: {series_path}")
        return None
    
    def _load_clinical_data(self) -> pd.DataFrame:
        """Load clinical data with patient diagnoses.
        
        Returns:
            DataFrame with patient clinical information
        """
        from .clinical_loader import ClinicalDataLoader
        
        loader = ClinicalDataLoader(str(self.clinical_data_path))
        loader.load_clinical_data()
        
        # Get patient master list
        clinical_data = loader.create_patient_master_list()
        
        return clinical_data
    
    def _merge_with_clinical_data(self, image_mapping: pd.DataFrame, clinical_data: pd.DataFrame) -> pd.DataFrame:
        """Merge image mapping with clinical data.
        
        Args:
            image_mapping: DataFrame with image paths and extracted patient IDs
            clinical_data: DataFrame with clinical information
            
        Returns:
            Merged DataFrame with complete image-to-label mapping
        """
        logger.info("Merging image mapping with clinical data...")
        
        # Merge on patient ID
        merged_data = image_mapping.merge(
            clinical_data, 
            left_on='extracted_patient_id', 
            right_on='patient_id', 
            how='left'
        )
        
        # Check merge quality
        total_images = len(merged_data)
        matched_images = merged_data['patient_id'].notna().sum()
        unmatched_images = total_images - matched_images
        
        logger.info(f"Image matching results:")
        logger.info(f"  Total images: {total_images}")
        logger.info(f"  Matched with clinical data: {matched_images}")
        logger.info(f"  Unmatched: {unmatched_images}")
        
        if unmatched_images > 0:
            logger.warning(f"Unmatched images: {unmatched_images}")
            # Show some examples of unmatched images
            unmatched = merged_data[merged_data['patient_id'].isna()]
            logger.warning(f"Examples of unmatched paths: {unmatched['series_path'].head().tolist()}")
            
        # Filter to only matched images
        matched_data = merged_data[merged_data['patient_id'].notna()].copy()
        
        # Add binary label
        matched_data['label'] = matched_data['is_pd'].astype(int)
        
        # Add image type indicator
        matched_data['image_type'] = 'spect'
        
        # Select relevant columns
        final_columns = [
            'series_path', 'series_name', 'patient_id', 'label', 'image_type',
            'num_dicom_files', 'is_pd'
        ]
        
        # Add any additional clinical columns that might be useful
        clinical_cols = ['sex', 'birthdt', 'enroll_date']
        for col in clinical_cols:
            if col in matched_data.columns:
                final_columns.append(col)
                
        final_mapping = matched_data[final_columns].copy()
        
        logger.info(f"Final dataset: {len(final_mapping)} images from {final_mapping['patient_id'].nunique()} patients")
        logger.info(f"Class distribution: {final_mapping['label'].value_counts().to_dict()}")
        
        return final_mapping
    
    def _save_image_mapping(self, mapping_df: pd.DataFrame):
        """Save image mapping to files.
        
        Args:
            mapping_df: DataFrame with image mapping
        """
        # Save main mapping
        mapping_file = self.metadata_path / "image_to_patient_mapping.csv"
        mapping_df.to_csv(mapping_file, index=False)
        logger.info(f"Saved image mapping to {mapping_file}")
        
        # Save summary statistics
        summary_file = self.metadata_path / "dataset_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("PPMI Dataset Summary\n")
            f.write("=" * 20 + "\n\n")
            f.write(f"Total images: {len(mapping_df)}\n")
            f.write(f"Unique patients: {mapping_df['patient_id'].nunique()}\n")
            f.write(f"PD images: {(mapping_df['label'] == 1).sum()}\n")
            f.write(f"Control images: {(mapping_df['label'] == 0).sum()}\n\n")
            
            # Patient-level statistics
            patient_stats = mapping_df.groupby('patient_id').agg({
                'label': 'first',
                'series_path': 'count'
            }).rename(columns={'series_path': 'num_images'})
            
            f.write("Patient-level statistics:\n")
            f.write(f"  PD patients: {(patient_stats['label'] == 1).sum()}\n")
            f.write(f"  Control patients: {(patient_stats['label'] == 0).sum()}\n")
            f.write(f"  Mean images per patient: {patient_stats['num_images'].mean():.1f}\n")
            f.write(f"  Min images per patient: {patient_stats['num_images'].min()}\n")
            f.write(f"  Max images per patient: {patient_stats['num_images'].max()}\n")
            
        logger.info(f"Saved dataset summary to {summary_file}")
        
        # Save JSON version for programmatic access
        json_file = self.metadata_path / "image_mapping.json"
        mapping_dict = mapping_df.to_dict('records')
        with open(json_file, 'w') as f:
            json.dump(mapping_dict, f, indent=2, default=str)
        logger.info(f"Saved JSON mapping to {json_file}")
    
    def create_data_splits(self, mapping_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create train/validation/test splits by patient.
        
        Args:
            mapping_df: DataFrame with image mapping
            
        Returns:
            Dictionary with train, validation, and test splits
        """
        logger.info("Creating data splits by patient...")
        
        # Get unique patients
        patients = mapping_df[['patient_id', 'label']].drop_duplicates()
        
        # Get splitting parameters
        train_ratio = self.config.get('splitting.train_ratio', 0.7)
        val_ratio = self.config.get('splitting.val_ratio', 0.15)
        test_ratio = self.config.get('splitting.test_ratio', 0.15)
        random_seed = self.config.get('splitting.random_seed', 42)
        
        # Stratified split by patient
        from sklearn.model_selection import train_test_split
        
        # First split: train vs (val + test)
        train_patients, temp_patients = train_test_split(
            patients, 
            test_size=(val_ratio + test_ratio),
            stratify=patients['label'],
            random_state=random_seed
        )
        
        # Second split: val vs test
        val_patients, test_patients = train_test_split(
            temp_patients,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp_patients['label'],
            random_state=random_seed
        )
        
        # Create splits
        train_data = mapping_df[mapping_df['patient_id'].isin(train_patients['patient_id'])]
        val_data = mapping_df[mapping_df['patient_id'].isin(val_patients['patient_id'])]
        test_data = mapping_df[mapping_df['patient_id'].isin(test_patients['patient_id'])]
        
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        # Save splits
        self._save_data_splits(splits)
        
        # Log split information
        for split_name, split_data in splits.items():
            n_patients = split_data['patient_id'].nunique()
            n_images = len(split_data)
            pd_count = (split_data['label'] == 1).sum()
            control_count = (split_data['label'] == 0).sum()
            
            logger.info(f"{split_name.capitalize()} split:")
            logger.info(f"  Patients: {n_patients}")
            logger.info(f"  Images: {n_images}")
            logger.info(f"  PD: {pd_count}, Control: {control_count}")
            
        return splits
    
    def _save_data_splits(self, splits: Dict[str, pd.DataFrame]):
        """Save data splits to files.
        
        Args:
            splits: Dictionary with train, validation, and test splits
        """
        for split_name, split_data in splits.items():
            # Save CSV
            csv_file = self.metadata_path / f"{split_name}_split.csv"
            split_data.to_csv(csv_file, index=False)
            
            # Save JSON
            json_file = self.metadata_path / f"{split_name}_split.json"
            split_dict = split_data.to_dict('records')
            with open(json_file, 'w') as f:
                json.dump(split_dict, f, indent=2, default=str)
                
        logger.info("Saved all data splits to metadata directory")


def create_ppmi_dataset(config) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Convenience function to create complete PPMI dataset.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (full_mapping, data_splits)
    """
    linker = PPMIDataLinker(config)
    
    # Create image-to-patient mapping
    mapping_df = linker.create_image_to_patient_mapping()
    
    # Create data splits
    data_splits = linker.create_data_splits(mapping_df)
    
    return mapping_df, data_splits
