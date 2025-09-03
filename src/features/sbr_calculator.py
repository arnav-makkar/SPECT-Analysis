"""
Striatal Binding Ratio (SBR) calculation for PPMI SPECT images.
This implements the classical biomarker approach for PD detection.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from scipy import ndimage
from skimage import measure, morphology
import SimpleITK as sitk
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SBRCalculator:
    """Calculator for Striatal Binding Ratio (SBR) from SPECT images."""
    
    def __init__(self, config):
        """Initialize SBR calculator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.sbr_params = config.get_sbr_params()
        
        # Define striatal and reference regions
        self.striatal_regions = self.sbr_params.get('striatal_regions', [
            'left_caudate', 'right_caudate', 'left_putamen', 'right_putamen'
        ])
        self.reference_regions = self.sbr_params.get('reference_regions', [
            'occipital_cortex', 'cerebellum'
        ])
        
        # SBR calculation parameters
        self.roi_size = 15  # Size of ROI in voxels
        self.threshold_percentile = 90  # Percentile for ROI thresholding
        
    def calculate_sbr_features(self, volume: np.ndarray, metadata: Dict) -> Dict[str, float]:
        """Calculate SBR features for a SPECT volume.
        
        Args:
            volume: 3D SPECT volume
            metadata: Image metadata
            
        Returns:
            Dictionary with SBR features
        """
        logger.debug("Calculating SBR features...")
        
        # Ensure volume is in standard orientation
        volume = self._standardize_orientation(volume)
        
        # Find central slice (usually shows striatum best)
        central_slice_idx = volume.shape[0] // 2
        central_slice = volume[central_slice_idx, :, :]
        
        # Calculate SBR for each striatal region
        sbr_features = {}
        
        for region in self.striatal_regions:
            try:
                sbr_value = self._calculate_region_sbr(central_slice, region)
                sbr_features[f'sbr_{region}'] = sbr_value
            except Exception as e:
                logger.warning(f"Failed to calculate SBR for {region}: {e}")
                sbr_features[f'sbr_{region}'] = np.nan
                
        # Calculate additional features
        sbr_features.update(self._calculate_additional_features(volume, central_slice))
        
        # Add metadata features
        sbr_features.update(self._extract_metadata_features(metadata))
        
        logger.debug(f"Calculated {len(sbr_features)} SBR features")
        return sbr_features
    
    def _standardize_orientation(self, volume: np.ndarray) -> np.ndarray:
        """Ensure volume is in standard orientation (RAS).
        
        Args:
            volume: 3D volume
            
        Returns:
            Standardized volume
        """
        # For now, assume volume is already in correct orientation
        # TODO: Implement orientation detection and correction
        return volume
    
    def _calculate_region_sbr(self, slice_2d: np.ndarray, region: str) -> float:
        """Calculate SBR for a specific brain region.
        
        Args:
            slice_2d: 2D slice image
            region: Brain region name
            
        Returns:
            SBR value
        """
        # Define region coordinates (approximate, should be atlas-based)
        region_coords = self._get_region_coordinates(slice_2d.shape, region)
        
        if region_coords is None:
            return np.nan
            
        # Extract striatal ROI
        striatal_roi = self._extract_roi(slice_2d, region_coords)
        
        # Find reference region (occipital cortex or cerebellum)
        reference_roi = self._find_reference_region(slice_2d, region_coords)
        
        if striatal_roi is None or reference_roi is None:
            return np.nan
            
        # Calculate SBR
        striatal_mean = np.mean(striatal_roi)
        reference_mean = np.mean(reference_roi)
        
        if reference_mean <= 0:
            return np.nan
            
        sbr = (striatal_mean - reference_mean) / reference_mean
        
        return float(sbr)
    
    def _get_region_coordinates(self, image_shape: Tuple[int, int], region: str) -> Optional[Tuple[int, int, int, int]]:
        """Get approximate coordinates for brain regions.
        
        Args:
            image_shape: Shape of the image (height, width)
            region: Brain region name
            
        Returns:
            Tuple of (x1, y1, x2, y2) coordinates or None
        """
        height, width = image_shape
        
        # Approximate coordinates (these should be replaced with atlas-based coordinates)
        if region == 'left_caudate':
            # Left side, upper middle
            x1, y1 = width // 4, height // 3
            x2, y2 = width // 3, height // 2
        elif region == 'right_caudate':
            # Right side, upper middle
            x1, y1 = 2 * width // 3, height // 3
            x2, y2 = 3 * width // 4, height // 2
        elif region == 'left_putamen':
            # Left side, lower middle
            x1, y1 = width // 4, height // 2
            x2, y2 = width // 3, 2 * height // 3
        elif region == 'right_putamen':
            # Right side, lower middle
            x1, y1 = 2 * width // 3, height // 2
            x2, y2 = 3 * width // 4, 2 * height // 3
        else:
            return None
            
        return (x1, y1, x2, y2)
    
    def _extract_roi(self, image: np.ndarray, coords: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract region of interest from image.
        
        Args:
            image: 2D image
            coords: ROI coordinates (x1, y1, x2, y2)
            
        Returns:
            ROI array or None
        """
        x1, y1, x2, y2 = coords
        
        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        x2 = max(x1 + 1, min(x2, image.shape[1]))
        y2 = max(y1 + 1, min(y2, image.shape[0]))
        
        # Extract ROI
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
            
        return roi
    
    def _find_reference_region(self, image: np.ndarray, striatal_coords: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Find reference region for SBR calculation.
        
        Args:
            image: 2D image
            striatal_coords: Coordinates of striatal region
            
        Returns:
            Reference region array or None
        """
        # Use occipital cortex as reference (posterior part of brain)
        height, width = image.shape
        
        # Define occipital region (posterior, middle)
        x1 = width // 3
        x2 = 2 * width // 3
        y1 = 3 * height // 4  # Posterior part
        y2 = height
        
        # Ensure we don't overlap with striatal regions
        sx1, sy1, sx2, sy2 = striatal_coords
        if (x1 < sx2 and x2 > sx1 and y1 < sy2 and y2 > sy1):
            # Overlap detected, adjust coordinates
            y1 = max(y1, sy2 + 5)
            
        if y1 >= y2:
            return None
            
        return self._extract_roi(image, (x1, y1, x2, y2))
    
    def _calculate_additional_features(self, volume: np.ndarray, central_slice: np.ndarray) -> Dict[str, float]:
        """Calculate additional features beyond SBR.
        
        Args:
            volume: 3D volume
            central_slice: Central 2D slice
            
        Returns:
            Dictionary with additional features
        """
        features = {}
        
        # Volume-based features
        features['volume_mean'] = float(np.mean(volume))
        features['volume_std'] = float(np.std(volume))
        features['volume_min'] = float(np.min(volume))
        features['volume_max'] = float(np.max(volume))
        
        # Central slice features
        features['central_slice_mean'] = float(np.mean(central_slice))
        features['central_slice_std'] = float(np.std(central_slice))
        
        # Asymmetry features (left vs right)
        height, width = central_slice.shape
        left_half = central_slice[:, :width//2]
        right_half = central_slice[:, width//2:]
        
        features['left_right_asymmetry'] = float(np.mean(left_half) - np.mean(right_half))
        features['left_right_ratio'] = float(np.mean(left_half) / (np.mean(right_half) + 1e-8))
        
        # Texture features (simple)
        features['texture_variance'] = float(ndimage.variance(central_slice))
        features['texture_entropy'] = float(self._calculate_entropy(central_slice))
        
        return features
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy.
        
        Args:
            image: 2D image
            
        Returns:
            Entropy value
        """
        # Normalize to 0-255 range
        img_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Calculate histogram
        hist, _ = np.histogram(img_norm, bins=256, range=(0, 256))
        hist = hist[hist > 0]  # Remove zero bins
        
        # Calculate entropy
        prob = hist / hist.sum()
        entropy = -np.sum(prob * np.log2(prob))
        
        return entropy
    
    def _extract_metadata_features(self, metadata: Dict) -> Dict[str, float]:
        """Extract features from DICOM metadata.
        
        Args:
            metadata: DICOM metadata dictionary
            
        Returns:
            Dictionary with metadata features
        """
        features = {}
        
        # Age (if available)
        if 'patient_birth_date' in metadata and metadata['patient_birth_date'] != 'Unknown':
            try:
                from datetime import datetime
                birth_date = datetime.strptime(metadata['patient_birth_date'], '%Y%m%d')
                age = (datetime.now() - birth_date).days / 365.25
                features['age'] = float(age)
            except:
                features['age'] = np.nan
        else:
            features['age'] = np.nan
            
        # Sex (encoded as numeric)
        if 'patient_sex' in metadata:
            sex = metadata['patient_sex']
            if sex == 'M':
                features['sex'] = 1.0
            elif sex == 'F':
                features['sex'] = 0.0
            else:
                features['sex'] = np.nan
        else:
            features['sex'] = np.nan
            
        # Technical parameters
        if 'pixel_spacing' in metadata:
            pixel_spacing = metadata['pixel_spacing']
            if isinstance(pixel_spacing, list) and len(pixel_spacing) >= 2:
                features['pixel_spacing_x'] = float(pixel_spacing[0])
                features['pixel_spacing_y'] = float(pixel_spacing[1])
            else:
                features['pixel_spacing_x'] = np.nan
                features['pixel_spacing_y'] = np.nan
        else:
            features['pixel_spacing_x'] = np.nan
            features['pixel_spacing_y'] = np.nan
            
        if 'slice_thickness' in metadata:
            features['slice_thickness'] = float(metadata['slice_thickness'])
        else:
            features['slice_thickness'] = np.nan
            
        return features
    
    def calculate_sbr_dataset(self, mapping_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SBR features for entire dataset.
        
        Args:
            mapping_df: DataFrame with image mapping
            
        Returns:
            DataFrame with SBR features
        """
        logger.info("Calculating SBR features for entire dataset...")
        
        all_features = []
        
        for idx, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc="Calculating SBR"):
            try:
                # Load processed volume
                processed_path = row.get('processed_path')
                if processed_path and Path(processed_path).exists():
                    volume = np.load(processed_path)
                    # Load metadata from DICOM
                    from ..data.dicom_loader import DICOMLoader
                    loader = DICOMLoader(row['series_path'])
                    slices, _ = loader.load_scan(Path(row['series_path']))
                    metadata = loader.extract_metadata(slices)
                else:
                    # Load from DICOM if processed not available
                    from ..data.dicom_loader import DICOMLoader
                    loader = DICOMLoader(row['series_path'])
                    slices, volume = loader.load_scan(Path(row['series_path']))
                    metadata = loader.extract_metadata(slices)
                    volume = loader.apply_rescale(volume, metadata)
                    
                # Calculate features
                features = self.calculate_sbr_features(volume, metadata)
                
                # Add basic information
                features['series_path'] = row['series_path']
                features['patient_id'] = row['patient_id']
                features['label'] = row['label']
                
                all_features.append(features)
                
            except Exception as e:
                logger.error(f"Failed to calculate features for {row['series_path']}: {e}")
                # Add failed row
                failed_features = {
                    'series_path': row['series_path'],
                    'patient_id': row['patient_id'],
                    'label': row['label']
                }
                # Add NaN for all feature columns
                for region in self.striatal_regions:
                    failed_features[f'sbr_{region}'] = np.nan
                failed_features.update({
                    'volume_mean': np.nan, 'volume_std': np.nan,
                    'volume_min': np.nan, 'volume_max': np.nan,
                    'central_slice_mean': np.nan, 'central_slice_std': np.nan,
                    'left_right_asymmetry': np.nan, 'left_right_ratio': np.nan,
                    'texture_variance': np.nan, 'texture_entropy': np.nan,
                    'age': np.nan, 'sex': np.nan,
                    'pixel_spacing_x': np.nan, 'pixel_spacing_y': np.nan,
                    'slice_thickness': np.nan
                })
                all_features.append(failed_features)
                
        # Create features DataFrame
        features_df = pd.DataFrame(all_features)
        
        logger.info(f"Calculated SBR features for {len(features_df)} images")
        logger.info(f"Feature columns: {list(features_df.columns)}")
        
        return features_df
    
    def save_sbr_features(self, features_df: pd.DataFrame, output_path: Path):
        """Save SBR features to files.
        
        Args:
            features_df: DataFrame with SBR features
            output_path: Directory to save features
        """
        # Save CSV
        csv_file = output_path / "sbr_features.csv"
        features_df.to_csv(csv_file, index=False)
        
        # Save summary statistics
        summary_file = output_path / "sbr_features_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("SBR Features Summary\n")
            f.write("=" * 20 + "\n\n")
            
            f.write(f"Total images: {len(features_df)}\n")
            f.write(f"Total features: {len(features_df.columns) - 3}\n")  # Exclude path, patient_id, label
            
            # SBR statistics
            sbr_cols = [col for col in features_df.columns if col.startswith('sbr_')]
            f.write(f"\nSBR Features ({len(sbr_cols)}):\n")
            for col in sbr_cols:
                values = features_df[col].dropna()
                if len(values) > 0:
                    f.write(f"  {col}: mean={values.mean():.3f}, std={values.std():.3f}\n")
                    
            # Feature correlation with label
            f.write(f"\nFeature correlation with label:\n")
            for col in features_df.columns:
                if col not in ['series_path', 'patient_id', 'label']:
                    corr = features_df[col].corr(features_df['label'])
                    f.write(f"  {col}: {corr:.3f}\n")
                    
        # Save JSON version
        json_file = output_path / "sbr_features.json"
        features_dict = features_df.to_dict('records')
        with open(json_file, 'w') as f:
            json.dump(features_dict, f, indent=2, default=str)
            
        logger.info(f"Saved SBR features to {csv_file}")
        logger.info(f"Saved SBR summary to {summary_file}")


def calculate_ppmi_sbr_features(config, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to calculate SBR features for PPMI dataset.
    
    Args:
        config: Configuration object
        mapping_df: DataFrame with image mapping
        
    Returns:
        DataFrame with SBR features
    """
    calculator = SBRCalculator(config)
    features_df = calculator.calculate_sbr_dataset(mapping_df)
    
    # Save features
    output_path = config.get_data_path('metadata')
    calculator.save_sbr_features(features_df, output_path)
    
    return features_df
