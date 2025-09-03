"""
Main preprocessing pipeline for PPMI SPECT images.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
from tqdm import tqdm
import SimpleITK as sitk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPMIPreprocessor:
    """Main preprocessing pipeline for PPMI SPECT images."""
    
    def __init__(self, config):
        """Initialize preprocessor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.raw_dicom_path = config.get_data_path('raw')
        self.processed_path = config.get_data_path('processed')
        self.metadata_path = config.get_data_path('metadata')
        
        # Get preprocessing parameters
        self.preprocessing_params = config.get_preprocessing_params()
        self.target_shape = self.preprocessing_params.get('target_shape', [128, 128, 128])
        
    def run_preprocessing_pipeline(self, mapping_df: pd.DataFrame) -> pd.DataFrame:
        """Run complete preprocessing pipeline.
        
        Args:
            mapping_df: DataFrame with image mapping
            
        Returns:
            DataFrame with preprocessing results
        """
        logger.info("Starting PPMI preprocessing pipeline...")
        
        # Load existing splits if available
        data_splits = self._load_data_splits()
        
        # Process each image
        processed_results = []
        
        for idx, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc="Processing images"):
            try:
                result = self._process_single_image(row)
                processed_results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {row['series_path']}: {e}")
                # Add failed result
                processed_results.append({
                    'series_path': row['series_path'],
                    'patient_id': row['patient_id'],
                    'label': row['label'],
                    'status': 'failed',
                    'error': str(e),
                    'processed_path': None
                })
                
        # Create results DataFrame
        results_df = pd.DataFrame(processed_results)
        
        # Save preprocessing results
        self._save_preprocessing_results(results_df)
        
        # Update data splits with processed paths
        updated_splits = self._update_data_splits(data_splits, results_df)
        
        logger.info("Preprocessing pipeline completed!")
        return results_df
    
    def _process_single_image(self, row: pd.Series) -> Dict:
        """Process a single SPECT image.
        
        Args:
            row: Row from mapping DataFrame
            
        Returns:
            Dictionary with processing results
        """
        series_path = row['series_path']
        patient_id = row['patient_id']
        label = row['label']
        
        logger.debug(f"Processing {series_path}")
        
        # Load DICOM series
        from .dicom_loader import DICOMLoader
        loader = DICOMLoader(series_path)
        slices, volume = loader.load_scan(Path(series_path))
        metadata = loader.extract_metadata(slices)
        
        # Apply rescale
        volume = loader.apply_rescale(volume, metadata)
        
        # Preprocess volume
        processed_volume = self._preprocess_volume(volume, metadata)
        
        # Save processed volume
        processed_path = self._save_processed_volume(processed_volume, patient_id, series_path)
        
        return {
            'series_path': series_path,
            'patient_id': patient_id,
            'label': label,
            'status': 'success',
            'error': None,
            'processed_path': str(processed_path),
            'original_shape': volume.shape,
            'processed_shape': processed_volume.shape,
            'metadata': metadata
        }
    
    def _preprocess_volume(self, volume: np.ndarray, metadata: Dict) -> np.ndarray:
        """Apply preprocessing to 3D volume.
        
        Args:
            volume: 3D volume array
            metadata: DICOM metadata
            
        Returns:
            Preprocessed volume
        """
        # Step 1: Spatial normalization (registration)
        if self.preprocessing_params.get('registration', {}).get('enabled', False):
            volume = self._spatial_normalization(volume, metadata)
            
        # Step 2: Intensity normalization
        volume = self._intensity_normalization(volume)
        
        # Step 3: Resize to target dimensions
        volume = self._resize_volume(volume)
        
        # Step 4: ROI extraction (if enabled)
        if self.preprocessing_params.get('roi', {}).get('extract_striatum', False):
            volume = self._extract_striatum_roi(volume)
            
        return volume
    
    def _spatial_normalization(self, volume: np.ndarray, metadata: Dict) -> np.ndarray:
        """Apply spatial normalization (registration) to volume.
        
        Args:
            volume: 3D volume array
            metadata: DICOM metadata
            
        Returns:
            Registered volume
        """
        logger.debug("Applying spatial normalization...")
        
        # Convert numpy array to SimpleITK image
        sitk_image = sitk.GetImageFromArray(volume)
        
        # Set spacing and origin from metadata
        if 'pixel_spacing' in metadata and 'slice_thickness' in metadata:
            spacing = list(metadata['pixel_spacing']) + [metadata['slice_thickness']]
            sitk_image.SetSpacing(spacing)
            
        # For now, return original volume (registration requires template)
        # TODO: Implement full registration pipeline
        logger.warning("Spatial normalization not fully implemented - using original volume")
        
        return volume
    
    def _intensity_normalization(self, volume: np.ndarray) -> np.ndarray:
        """Apply intensity normalization to volume.
        
        Args:
            volume: 3D volume array
            
        Returns:
            Normalized volume
        """
        logger.debug("Applying intensity normalization...")
        
        normalization_method = self.preprocessing_params.get('normalization_method', 'zscore')
        
        if normalization_method == 'zscore':
            # Z-score normalization
            brain_mask = volume > np.percentile(volume, 10)  # Simple brain mask
            brain_voxels = volume[brain_mask]
            
            if len(brain_voxels) > 0:
                mean_val = np.mean(brain_voxels)
                std_val = np.std(brain_voxels)
                
                if std_val > 0:
                    volume = (volume - mean_val) / std_val
                else:
                    volume = volume - mean_val
                    
        elif normalization_method == 'minmax':
            # Min-max normalization to [0, 1]
            brain_mask = volume > np.percentile(volume, 10)
            brain_voxels = volume[brain_mask]
            
            if len(brain_voxels) > 0:
                min_val = np.min(brain_voxels)
                max_val = np.max(brain_voxels)
                
                if max_val > min_val:
                    volume = (volume - min_val) / (max_val - min_val)
                    
        elif normalization_method == 'percentile':
            # Percentile-based normalization
            brain_mask = volume > np.percentile(volume, 10)
            brain_voxels = volume[brain_mask]
            
            if len(brain_voxels) > 0:
                p2 = np.percentile(brain_voxels, 2)
                p98 = np.percentile(brain_voxels, 98)
                
                if p98 > p2:
                    volume = np.clip((volume - p2) / (p98 - p2), 0, 1)
                    
        return volume.astype(np.float32)
    
    def _resize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Resize volume to target dimensions.
        
        Args:
            volume: 3D volume array
            
        Returns:
            Resized volume
        """
        logger.debug(f"Resizing volume from {volume.shape} to {self.target_shape}")
        
        from skimage.transform import resize
        
        # Resize to target dimensions
        resized_volume = resize(
            volume, 
            self.target_shape, 
            preserve_range=True, 
            anti_aliasing=True
        )
        
        return resized_volume.astype(np.float32)
    
    def _extract_striatum_roi(self, volume: np.ndarray) -> np.ndarray:
        """Extract striatum region of interest.
        
        Args:
            volume: 3D volume array
            
        Returns:
            Volume with striatum ROI extracted
        """
        logger.debug("Extracting striatum ROI...")
        
        # For now, return the full volume
        # TODO: Implement striatum ROI extraction using brain atlases
        logger.warning("Striatum ROI extraction not implemented - using full volume")
        
        return volume
    
    def _save_processed_volume(self, volume: np.ndarray, patient_id: str, series_path: str) -> Path:
        """Save processed volume to file.
        
        Args:
            volume: Processed 3D volume
            patient_id: Patient ID
            series_path: Original series path
            
        Returns:
            Path to saved file
        """
        # Create patient directory
        patient_dir = self.processed_path / f"patient_{patient_id}"
        patient_dir.mkdir(exist_ok=True)
        
        # Create filename from series path
        series_name = Path(series_path).name
        filename = f"{series_name}_processed.npy"
        
        # Save as numpy array
        output_path = patient_dir / filename
        np.save(output_path, volume)
        
        # Also save as NIfTI for compatibility with other tools
        nifti_path = patient_dir / f"{series_name}_processed.nii.gz"
        sitk_image = sitk.GetImageFromArray(volume)
        sitk.WriteImage(sitk_image, str(nifti_path))
        
        logger.debug(f"Saved processed volume to {output_path}")
        return output_path
    
    def _load_data_splits(self) -> Dict[str, pd.DataFrame]:
        """Load existing data splits.
        
        Returns:
            Dictionary with data splits
        """
        splits = {}
        
        for split_name in ['train', 'validation', 'test']:
            split_file = self.metadata_path / f"{split_name}_split.csv"
            if split_file.exists():
                splits[split_name] = pd.read_csv(split_file)
                logger.info(f"Loaded {split_name} split: {len(splits[split_name])} images")
            else:
                logger.warning(f"Split file not found: {split_file}")
                
        return splits
    
    def _update_data_splits(self, data_splits: Dict[str, pd.DataFrame], 
                           results_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Update data splits with processed paths.
        
        Args:
            data_splits: Original data splits
            results_df: Preprocessing results
            
        Returns:
            Updated data splits
        """
        # Create mapping from series_path to processed_path
        path_mapping = results_df.set_index('series_path')['processed_path'].to_dict()
        
        updated_splits = {}
        
        for split_name, split_df in data_splits.items():
            # Add processed path column
            split_df['processed_path'] = split_df['series_path'].map(path_mapping)
            
            # Filter to only successfully processed images
            successful_df = split_df[split_df['processed_path'].notna()].copy()
            
            updated_splits[split_name] = successful_df
            
            logger.info(f"Updated {split_name} split: {len(successful_df)} successful images")
            
            # Save updated split
            output_file = self.metadata_path / f"{split_name}_split_processed.csv"
            successful_df.to_csv(output_file, index=False)
            
        return updated_splits
    
    def _save_preprocessing_results(self, results_df: pd.DataFrame):
        """Save preprocessing results to files.
        
        Args:
            results_df: DataFrame with preprocessing results
        """
        # Save main results
        results_file = self.metadata_path / "preprocessing_results.csv"
        results_df.to_csv(results_file, index=False)
        
        # Save summary
        summary_file = self.metadata_path / "preprocessing_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("PPMI Preprocessing Summary\n")
            f.write("=" * 25 + "\n\n")
            
            total_images = len(results_df)
            successful = (results_df['status'] == 'success').sum()
            failed = (results_df['status'] == 'failed').sum()
            
            f.write(f"Total images: {total_images}\n")
            f.write(f"Successfully processed: {successful}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"Success rate: {successful/total_images*100:.1f}%\n\n")
            
            if failed > 0:
                f.write("Failed images:\n")
                failed_images = results_df[results_df['status'] == 'failed']
                for _, row in failed_images.iterrows():
                    f.write(f"  {row['series_path']}: {row['error']}\n")
                    
        # Save JSON version
        json_file = self.metadata_path / "preprocessing_results.json"
        results_dict = results_df.to_dict('records')
        with open(json_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
            
        logger.info(f"Saved preprocessing results to {results_file}")
        logger.info(f"Saved preprocessing summary to {summary_file}")


def run_ppmi_preprocessing(config) -> pd.DataFrame:
    """Convenience function to run PPMI preprocessing.
    
    Args:
        config: Configuration object
        
    Returns:
        DataFrame with preprocessing results
    """
    # Load image mapping
    mapping_file = config.get_data_path('metadata') / "image_to_patient_mapping.csv"
    if not mapping_file.exists():
        raise FileNotFoundError(f"Image mapping file not found: {mapping_file}")
        
    mapping_df = pd.read_csv(mapping_file)
    
    # Run preprocessing
    preprocessor = PPMIPreprocessor(config)
    results_df = preprocessor.run_preprocessing_pipeline(mapping_df)
    
    return results_df


if __name__ == "__main__":
    from ..utils.config import get_config
    
    config = get_config()
    results = run_ppmi_preprocessing(config)
    print(f"Preprocessing completed for {len(results)} images")
