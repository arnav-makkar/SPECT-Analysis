"""
Simplified SBR Calculator for PPMI Data

This is a simplified version that works with the current data structure
and provides basic features for demonstration purposes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from scipy import ndimage
from skimage import measure, morphology
import pydicom
from datetime import datetime
from tqdm import tqdm

logger = logging.getLogger(__name__)

class SimpleSBRCalculator:
    """Simplified SBR calculator for PPMI data."""
    
    def __init__(self):
        """Initialize the calculator."""
        self.striatal_regions = ['left_caudate', 'right_caudate', 'left_putamen', 'right_putamen']
        
    def calculate_sbr_dataset(self, mapping_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SBR features for a dataset.
        
        Args:
            mapping_df: DataFrame with image mapping information
            
        Returns:
            DataFrame with calculated features
        """
        logger.info(f"Calculating SBR features for {len(mapping_df)} images...")
        
        all_features = []
        
        for idx, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc="Calculating SBR"):
            try:
                # Load DICOM file
                file_path = row['file_path']
                ds = pydicom.dcmread(file_path)
                volume = ds.pixel_array
                
                # Calculate basic features
                features = self._calculate_basic_features(volume)
                
                # Add metadata
                features['patient_id'] = row['patient_id']
                features['file_path'] = file_path
                features['sex'] = row.get('sex', 'Unknown')
                features['age'] = pd.to_numeric(row.get('age', 65), errors='coerce')
                
                # Add mock SBR features for demonstration
                features.update(self._calculate_mock_sbr_features(volume))
                
                all_features.append(features)
                
            except Exception as e:
                logger.error(f"Failed to calculate features for {file_path}: {e}")
                # Add failed row with NaN values
                failed_features = {
                    'patient_id': row['patient_id'],
                    'file_path': file_path,
                    'sex': row.get('sex', 'Unknown'),
                    'age': pd.to_numeric(row.get('age', 65), errors='coerce'),
                    'volume_shape': str((0, 0, 0)),
                    'total_voxels': 0,
                    'min_intensity': np.nan,
                    'max_intensity': np.nan,
                    'mean_intensity': np.nan,
                    'std_intensity': np.nan,
                    'median_intensity': np.nan,
                    'volume_range': np.nan,
                    'coefficient_variation': np.nan,
                    'entropy': np.nan,
                    'sbr_left_caudate': np.nan,
                    'sbr_right_caudate': np.nan,
                    'sbr_left_putamen': np.nan,
                    'sbr_right_putamen': np.nan,
                    'asymmetry_index': np.nan,
                    'texture_variance': np.nan,
                    'texture_entropy': np.nan
                }
                all_features.append(failed_features)
        
        # Create features DataFrame
        features_df = pd.DataFrame(all_features)
        
        logger.info(f"Calculated features for {len(features_df)} images")
        logger.info(f"Feature columns: {list(features_df.columns)}")
        
        return features_df
    
    def _calculate_basic_features(self, volume: np.ndarray) -> Dict:
        """Calculate basic volume features.
        
        Args:
            volume: 3D volume array
            
        Returns:
            Dictionary of basic features
        """
        features = {}
        
        # Basic statistics
        features['volume_shape'] = str(volume.shape)
        features['total_voxels'] = volume.size
        features['min_intensity'] = float(volume.min())
        features['max_intensity'] = float(volume.max())
        features['mean_intensity'] = float(volume.mean())
        features['std_intensity'] = float(volume.std())
        features['median_intensity'] = float(np.median(volume))
        
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
    
    def _calculate_mock_sbr_features(self, volume: np.ndarray) -> Dict:
        """Calculate mock SBR features for demonstration.
        
        Args:
            volume: 3D volume array
            
        Returns:
            Dictionary of mock SBR features
        """
        features = {}
        
        # Mock SBR values (in real implementation, these would be calculated from striatal ROIs)
        np.random.seed(hash(str(volume.shape)) % 2**32)  # Deterministic based on volume shape
        
        # Generate mock SBR values
        features['sbr_left_caudate'] = np.random.normal(2.5, 0.5)
        features['sbr_right_caudate'] = np.random.normal(2.4, 0.5)
        features['sbr_left_putamen'] = np.random.normal(2.3, 0.5)
        features['sbr_right_putamen'] = np.random.normal(2.2, 0.5)
        
        # Calculate asymmetry index
        left_sbr = (features['sbr_left_caudate'] + features['sbr_left_putamen']) / 2
        right_sbr = (features['sbr_right_caudate'] + features['sbr_right_putamen']) / 2
        features['asymmetry_index'] = abs(left_sbr - right_sbr) / ((left_sbr + right_sbr) / 2)
        
        # Mock texture features
        features['texture_variance'] = np.random.normal(1000, 200)
        features['texture_entropy'] = np.random.normal(8.0, 1.0)
        
        return features
    
    def save_features(self, features_df: pd.DataFrame, output_path: Path):
        """Save features to files.
        
        Args:
            features_df: DataFrame with features
            output_path: Directory to save features
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_file = output_path / "simple_sbr_features.csv"
        features_df.to_csv(csv_file, index=False)
        logger.info(f"Features saved to {csv_file}")
        
        # Save summary statistics
        summary_file = output_path / "simple_sbr_features_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Simple SBR Features Summary\n")
            f.write("=" * 30 + "\n\n")
            
            f.write(f"Total images: {len(features_df)}\n")
            f.write(f"Total features: {len(features_df.columns) - 4}\n")  # Exclude metadata columns
            
            # SBR statistics
            sbr_cols = [col for col in features_df.columns if col.startswith('sbr_')]
            f.write(f"\nSBR Features ({len(sbr_cols)}):\n")
            for col in sbr_cols:
                values = features_df[col].dropna()
                if len(values) > 0:
                    f.write(f"  {col}: mean={values.mean():.3f}, std={values.std():.3f}\n")
                    
            # Basic feature statistics
            basic_cols = ['mean_intensity', 'std_intensity', 'entropy', 'asymmetry_index']
            f.write(f"\nBasic Features:\n")
            for col in basic_cols:
                if col in features_df.columns:
                    values = features_df[col].dropna()
                    if len(values) > 0:
                        f.write(f"  {col}: mean={values.mean():.3f}, std={values.std():.3f}\n")
        
        logger.info(f"Summary saved to {summary_file}")


def calculate_simple_sbr_features(mapping_df: pd.DataFrame, output_path: str = "results/features") -> pd.DataFrame:
    """Convenience function to calculate simple SBR features.
    
    Args:
        mapping_df: DataFrame with image mapping
        output_path: Directory to save results
        
    Returns:
        DataFrame with calculated features
    """
    calculator = SimpleSBRCalculator()
    features_df = calculator.calculate_sbr_dataset(mapping_df)
    
    # Save results
    calculator.save_features(features_df, Path(output_path))
    
    return features_df
