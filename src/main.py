#!/usr/bin/env python3
"""
Main script for PPMI Parkinson's Disease Detection project.
This script orchestrates the entire pipeline from data organization to preprocessing.
"""

import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the PPMI pipeline."""
    
    logger.info("Starting PPMI Parkinson's Disease Detection Pipeline")
    logger.info("=" * 60)
    
    try:
        # Import configuration
        from utils.config import get_config
        config = get_config()
        
        # Validate configuration
        config.validate_config()
        logger.info("Configuration validated successfully")
        
        # Step 1: Data Organization and Linking
        logger.info("\nStep 1: Data Organization and Linking")
        logger.info("-" * 40)
        
        from data.data_linker import create_ppmi_dataset
        
        logger.info("Creating image-to-patient mapping...")
        mapping_df, data_splits = create_ppmi_dataset(config)
        
        logger.info(f"✓ Dataset created: {len(mapping_df)} images from {mapping_df['patient_id'].nunique()} patients")
        logger.info(f"✓ Data splits created: Train={len(data_splits['train'])}, Val={len(data_splits['validation'])}, Test={len(data_splits['test'])}")
        
        # Step 2: SBR Feature Calculation
        logger.info("\nStep 2: SBR Feature Calculation")
        logger.info("-" * 40)
        
        from features.sbr_calculator import calculate_ppmi_sbr_features
        
        logger.info("Calculating SBR features...")
        sbr_features_df = calculate_ppmi_sbr_features(config, mapping_df)
        
        logger.info(f"✓ SBR features calculated: {len(sbr_features_df.columns) - 3} features")
        
        # Step 3: Image Preprocessing
        logger.info("\nStep 3: Image Preprocessing")
        logger.info("-" * 40)
        
        from data.preprocessing import run_ppmi_preprocessing
        
        logger.info("Running preprocessing pipeline...")
        preprocessing_results = run_ppmi_preprocessing(config)
        
        successful = (preprocessing_results['status'] == 'success').sum()
        total = len(preprocessing_results)
        logger.info(f"✓ Preprocessing completed: {successful}/{total} images processed successfully")
        
        # Step 4: Final Summary
        logger.info("\nStep 4: Pipeline Summary")
        logger.info("-" * 40)
        
        logger.info("🎉 PPMI Pipeline Completed Successfully! 🎉")
        logger.info("\nFinal Dataset Statistics:")
        logger.info(f"  Total images: {len(mapping_df)}")
        logger.info(f"  Unique patients: {mapping_df['patient_id'].nunique()}")
        logger.info(f"  PD images: {(mapping_df['label'] == 1).sum()}")
        logger.info(f"  Control images: {(mapping_df['label'] == 0).sum()}")
        logger.info(f"  SBR features: {len(sbr_features_df.columns) - 3}")
        logger.info(f"  Preprocessed images: {successful}")
        
        logger.info("\nOutput Files Created:")
        metadata_path = config.get_data_path('metadata')
        for file_path in metadata_path.glob("*"):
            if file_path.is_file():
                logger.info(f"  {file_path.name}")
                
        logger.info("\nNext Steps:")
        logger.info("1. Review the data exploration notebook: notebooks/01_data_exploration.ipynb")
        logger.info("2. Train classical ML model: python src/models/train_classical.py")
        logger.info("3. Train CNN model: python src/models/train_cnn.py")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
