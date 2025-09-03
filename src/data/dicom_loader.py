"""
DICOM loading and preprocessing utilities for PPMI SPECT images.
"""

import os
import pydicom
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DICOMLoader:
    """Loader for DICOM SPECT images from PPMI dataset."""
    
    def __init__(self, dicom_path: str):
        """Initialize DICOM loader.
        
        Args:
            dicom_path: Path to directory containing DICOM files
        """
        self.dicom_path = Path(dicom_path)
        if not self.dicom_path.exists():
            raise FileNotFoundError(f"DICOM path does not exist: {dicom_path}")
            
    def find_dicom_series(self) -> List[Path]:
        """Find all DICOM series directories.
        
        Returns:
            List of paths to DICOM series directories
        """
        series_dirs = []
        
        for item in self.dicom_path.iterdir():
            if item.is_dir():
                # Check if directory contains DICOM files
                dicom_files = list(item.glob("*.dcm"))
                if dicom_files:
                    series_dirs.append(item)
                    
        logger.info(f"Found {len(series_dirs)} DICOM series directories")
        return series_dirs
    
    def load_scan(self, series_path: Path) -> Tuple[List[pydicom.Dataset], np.ndarray]:
        """Load a complete DICOM series into a 3D volume.
        
        Args:
            series_path: Path to directory containing DICOM series
            
        Returns:
            Tuple of (slices, 3D_volume)
        """
        # Get all DICOM files in the series
        dicom_files = list(series_path.glob("*.dcm"))
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {series_path}")
            
        # Load all slices
        slices = []
        for dcm_file in dicom_files:
            try:
                ds = pydicom.dcmread(str(dcm_file))
                slices.append(ds)
            except Exception as e:
                logger.warning(f"Failed to load {dcm_file}: {e}")
                continue
                
        if not slices:
            raise ValueError(f"No valid DICOM files could be loaded from {series_path}")
            
        # Sort slices by position (Z coordinate)
        slices = self._sort_slices_by_position(slices)
        
        # Create 3D volume
        volume = self._create_3d_volume(slices)
        
        return slices, volume
    
    def _sort_slices_by_position(self, slices: List[pydicom.Dataset]) -> List[pydicom.Dataset]:
        """Sort DICOM slices by their Z position.
        
        Args:
            slices: List of DICOM datasets
            
        Returns:
            Sorted list of DICOM datasets
        """
        try:
            # Try to sort by ImagePositionPatient[2] (Z coordinate)
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except (AttributeError, IndexError, ValueError):
            try:
                # Fallback: sort by InstanceNumber
                slices.sort(key=lambda x: int(x.InstanceNumber))
            except (AttributeError, ValueError):
                # Final fallback: sort by filename
                slices.sort(key=lambda x: x.filename)
                logger.warning("Could not sort slices by position, using filename order")
                
        return slices
    
    def _create_3d_volume(self, slices: List[pydicom.Dataset]) -> np.ndarray:
        """Create 3D volume from sorted DICOM slices.
        
        Args:
            slices: Sorted list of DICOM datasets
            
        Returns:
            3D numpy array representing the volume
        """
        # Get dimensions from first slice
        first_slice = slices[0]
        height = first_slice.Rows
        width = first_slice.Columns
        depth = len(slices)
        
        # Create 3D array
        volume = np.zeros((depth, height, width), dtype=np.int16)
        
        # Fill volume with pixel data
        for i, slice_ds in enumerate(slices):
            try:
                pixel_array = slice_ds.pixel_array
                if pixel_array.shape != (height, width):
                    logger.warning(f"Slice {i} has unexpected shape: {pixel_array.shape}")
                    # Resize if necessary
                    from skimage.transform import resize
                    pixel_array = resize(pixel_array, (height, width), preserve_range=True)
                    
                volume[i, :, :] = pixel_array
                
            except Exception as e:
                logger.warning(f"Failed to process slice {i}: {e}")
                continue
                
        return volume
    
    def extract_metadata(self, slices: List[pydicom.Dataset]) -> Dict[str, any]:
        """Extract metadata from DICOM series.
        
        Args:
            slices: List of DICOM datasets
            
        Returns:
            Dictionary of metadata
        """
        if not slices:
            return {}
            
        first_slice = slices[0]
        metadata = {}
        
        # Patient information
        try:
            metadata['patient_id'] = getattr(first_slice, 'PatientID', 'Unknown')
            metadata['patient_name'] = getattr(first_slice, 'PatientName', 'Unknown')
            metadata['patient_birth_date'] = getattr(first_slice, 'PatientBirthDate', 'Unknown')
            metadata['patient_sex'] = getattr(first_slice, 'PatientSex', 'Unknown')
        except Exception as e:
            logger.warning(f"Failed to extract patient info: {e}")
            
        # Study information
        try:
            metadata['study_date'] = getattr(first_slice, 'StudyDate', 'Unknown')
            metadata['study_time'] = getattr(first_slice, 'StudyTime', 'Unknown')
            metadata['study_description'] = getattr(first_slice, 'StudyDescription', 'Unknown')
            metadata['study_instance_uid'] = getattr(first_slice, 'StudyInstanceUID', 'Unknown')
        except Exception as e:
            logger.warning(f"Failed to extract study info: {e}")
            
        # Image information
        try:
            metadata['modality'] = getattr(first_slice, 'Modality', 'Unknown')
            metadata['image_type'] = getattr(first_slice, 'ImageType', [])
            metadata['manufacturer'] = getattr(first_slice, 'Manufacturer', 'Unknown')
            metadata['manufacturer_model_name'] = getattr(first_slice, 'ManufacturerModelName', 'Unknown')
        except Exception as e:
            logger.warning(f"Failed to extract image info: {e}")
            
        # Technical parameters
        try:
            metadata['pixel_spacing'] = getattr(first_slice, 'PixelSpacing', [1.0, 1.0])
            metadata['slice_thickness'] = getattr(first_slice, 'SliceThickness', 1.0)
            metadata['rows'] = first_slice.Rows
            metadata['columns'] = first_slice.Columns
            metadata['bits_allocated'] = getattr(first_slice, 'BitsAllocated', 16)
            metadata['samples_per_pixel'] = getattr(first_slice, 'SamplesPerPixel', 1)
        except Exception as e:
            logger.warning(f"Failed to extract technical info: {e}")
            
        # Rescale parameters (important for SPECT)
        try:
            metadata['rescale_slope'] = getattr(first_slice, 'RescaleSlope', 1.0)
            metadata['rescale_intercept'] = getattr(first_slice, 'RescaleIntercept', 0.0)
        except Exception as e:
            logger.warning(f"Failed to extract rescale info: {e}")
            
        return metadata
    
    def apply_rescale(self, volume: np.ndarray, metadata: Dict[str, any]) -> np.ndarray:
        """Apply rescale slope and intercept to volume.
        
        Args:
            volume: 3D volume array
            metadata: DICOM metadata dictionary
            
        Returns:
            Rescaled volume
        """
        slope = metadata.get('rescale_slope', 1.0)
        intercept = metadata.get('rescale_intercept', 0.0)
        
        if slope != 1.0 or intercept != 0.0:
            volume = volume.astype(np.float32) * slope + intercept
            logger.info(f"Applied rescale: slope={slope}, intercept={intercept}")
            
        return volume
    
    def get_volume_info(self, volume: np.ndarray, metadata: Dict[str, any]) -> Dict[str, any]:
        """Get comprehensive information about the 3D volume.
        
        Args:
            volume: 3D volume array
            metadata: DICOM metadata dictionary
            
        Returns:
            Dictionary with volume information
        """
        info = {
            'shape': volume.shape,
            'dtype': str(volume.dtype),
            'min_value': float(np.min(volume)),
            'max_value': float(np.max(volume)),
            'mean_value': float(np.mean(volume)),
            'std_value': float(np.std(volume)),
            'memory_mb': volume.nbytes / (1024 * 1024)
        }
        
        # Add metadata
        info.update(metadata)
        
        return info
    
    def save_volume_info(self, series_path: Path, info: Dict[str, any], output_path: Path):
        """Save volume information to file.
        
        Args:
            series_path: Path to DICOM series
            info: Volume information dictionary
            output_path: Path to save information
        """
        import json
        
        # Add series path to info
        info['series_path'] = str(series_path)
        
        # Create output file path
        output_file = output_path / f"{series_path.name}_info.json"
        
        with open(output_file, 'w') as f:
            json.dump(info, f, indent=2, default=str)
            
        logger.info(f"Saved volume info to {output_file}")


def load_ppmi_scan(dicom_path: str) -> Tuple[np.ndarray, Dict[str, any]]:
    """Convenience function to load a PPMI scan.
    
    Args:
        dicom_path: Path to DICOM series directory
        
    Returns:
        Tuple of (3D_volume, metadata)
    """
    loader = DICOMLoader(dicom_path)
    series_path = Path(dicom_path)
    
    try:
        slices, volume = loader.load_scan(series_path)
        metadata = loader.extract_metadata(slices)
        volume = loader.apply_rescale(volume, metadata)
        
        logger.info(f"Successfully loaded scan: {volume.shape}")
        return volume, metadata
        
    except Exception as e:
        logger.error(f"Failed to load scan from {dicom_path}: {e}")
        raise
