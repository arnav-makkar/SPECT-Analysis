#!/usr/bin/env python3
"""
Simple baseline analysis for PPMI dataset without requiring labels.
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run simple baseline analysis."""
    
    logger.info("🚀 Starting PPMI Simple Baseline Analysis")
    logger.info("=" * 50)
    
    try:
        # Load the mapping data
        mapping_file = Path("data/metadata/ppmi_image_mapping.csv")
        if not mapping_file.exists():
            logger.error("Mapping file not found. Please run the data loading first.")
            return False
            
        mapping_df = pd.read_csv(mapping_file)
        logger.info(f"✅ Loaded mapping data: {mapping_df.shape}")
        
        # Basic dataset statistics
        logger.info("\n📊 Dataset Overview:")
        logger.info(f"  Total images: {len(mapping_df)}")
        logger.info(f"  Unique patients: {mapping_df['patient_id'].nunique()}")
        logger.info(f"  Data folders: {mapping_df['data_folder'].nunique()}")
        
        # Data distribution
        logger.info("\n📁 Data Distribution:")
        folder_counts = mapping_df['data_folder'].value_counts()
        for folder, count in folder_counts.items():
            logger.info(f"  {folder}: {count} images")
            
        # Patient distribution
        logger.info("\n👥 Patient Distribution:")
        patient_counts = mapping_df['patient_id'].value_counts()
        logger.info(f"  Images per patient - Mean: {patient_counts.mean():.1f}")
        logger.info(f"  Images per patient - Min: {patient_counts.min()}")
        logger.info(f"  Images per patient - Max: {patient_counts.max()}")
        
        # Clinical data summary
        if 'sex' in mapping_df.columns:
            logger.info("\n👤 Sex Distribution:")
            sex_counts = mapping_df['sex'].value_counts()
            for sex, count in sex_counts.items():
                logger.info(f"  {sex}: {count} images")
                
        if 'age' in mapping_df.columns:
            logger.info("\n📅 Age Statistics:")
            age_data = pd.to_numeric(mapping_df['age'], errors='coerce').dropna()
            if len(age_data) > 0:
                logger.info(f"  Mean age: {age_data.mean():.1f} years")
                logger.info(f"  Age range: {age_data.min():.1f} - {age_data.max():.1f} years")
                logger.info(f"  Age std: {age_data.std():.1f} years")
                
        # Modality information
        if 'modality' in mapping_df.columns:
            logger.info("\n🔬 Modality Information:")
            modality_counts = mapping_df['modality'].value_counts()
            for modality, count in modality_counts.items():
                logger.info(f"  {modality}: {count} images")
                
        # Create visualizations
        logger.info("\n🎨 Creating visualizations...")
        create_visualizations(mapping_df)
        
        # Basic feature analysis
        logger.info("\n🔍 Basic Feature Analysis:")
        basic_features = extract_basic_features(mapping_df)
        
        # Save results
        logger.info("\n💾 Saving results...")
        save_results(mapping_df, basic_features)
        
        logger.info("\n🎉 Analysis Complete!")
        logger.info("=" * 30)
        logger.info(f"📊 Dataset: {len(mapping_df)} images from {mapping_df['patient_id'].nunique()} patients")
        logger.info(f"🔬 Basic features extracted: {len(basic_features.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_visualizations(mapping_df):
    """Create basic visualizations."""
    
    # Set up plotting
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Data folder distribution
    folder_counts = mapping_df['data_folder'].value_counts()
    axes[0,0].pie(folder_counts.values, labels=folder_counts.index, autopct='%1.1f%%')
    axes[0,0].set_title('Images by Data Folder')
    
    # 2. Patient distribution
    patient_counts = mapping_df['patient_id'].value_counts()
    axes[0,1].hist(patient_counts.values, bins=10, alpha=0.7, edgecolor='black')
    axes[0,1].set_title('Images per Patient Distribution')
    axes[0,1].set_xlabel('Number of Images')
    axes[0,1].set_ylabel('Number of Patients')
    
    # 3. Sex distribution
    if 'sex' in mapping_df.columns:
        sex_counts = mapping_df['sex'].value_counts()
        axes[1,0].bar(sex_counts.index, sex_counts.values, alpha=0.7)
        axes[1,0].set_title('Sex Distribution')
        axes[1,0].set_ylabel('Number of Images')
    
    # 4. Age distribution
    if 'age' in mapping_df.columns:
        age_data = pd.to_numeric(mapping_df['age'], errors='coerce').dropna()
        if len(age_data) > 0:
            axes[1,1].hist(age_data, bins=15, alpha=0.7, edgecolor='black')
            axes[1,1].set_title('Age Distribution')
            axes[1,1].set_xlabel('Age (years)')
            axes[1,1].set_ylabel('Number of Images')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "ppmi_dataset_overview.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"✅ Visualizations saved to {output_dir}")


def extract_basic_features(mapping_df):
    """Extract basic features from the dataset."""
    
    features = []
    
    for idx, row in mapping_df.iterrows():
        feature_dict = {
            'patient_id': row['patient_id'],
            'data_folder': row['data_folder'],
            'modality': row.get('modality', 'unknown'),
            'sex': row.get('sex', 'unknown'),
            'age': pd.to_numeric(row.get('age'), errors='coerce'),
            'description': row.get('description', ''),
            'event_id': row.get('EVENT_ID', ''),
            'infodt': row.get('INFODT', '')
        }
        
        # Add derived features
        if pd.notna(feature_dict['age']):
            feature_dict['age_group'] = get_age_group(feature_dict['age'])
        else:
            feature_dict['age_group'] = 'unknown'
            
        features.append(feature_dict)
    
    features_df = pd.DataFrame(features)
    
    # Add some aggregate features
    patient_stats = mapping_df.groupby('patient_id').agg({
        'data_folder': 'count',
        'age': lambda x: pd.to_numeric(x, errors='coerce').mean()
    }).rename(columns={'data_folder': 'num_images', 'age': 'mean_age'})
    
    features_df = features_df.merge(patient_stats, on='patient_id', how='left')
    
    logger.info(f"✅ Extracted {len(features_df.columns)} features")
    
    return features_df


def get_age_group(age):
    """Categorize age into groups."""
    if age < 55:
        return 'young'
    elif age < 65:
        return 'middle'
    elif age < 75:
        return 'senior'
    else:
        return 'elderly'


def save_results(mapping_df, features_df):
    """Save analysis results."""
    
    output_dir = Path("results/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save features
    features_df.to_csv(output_dir / "ppmi_basic_features.csv", index=False)
    
    # Save summary report
    with open(output_dir / "ppmi_analysis_summary.txt", 'w') as f:
        f.write("PPMI Dataset Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Dataset Overview:\n")
        f.write(f"  Total images: {len(mapping_df)}\n")
        f.write(f"  Unique patients: {mapping_df['patient_id'].nunique()}\n")
        f.write(f"  Data folders: {mapping_df['data_folder'].nunique()}\n\n")
        
        f.write(f"Data Distribution:\n")
        folder_counts = mapping_df['data_folder'].value_counts()
        for folder, count in folder_counts.items():
            f.write(f"  {folder}: {count} images\n")
            
        f.write(f"\nPatient Statistics:\n")
        patient_counts = mapping_df['patient_id'].value_counts()
        f.write(f"  Mean images per patient: {patient_counts.mean():.1f}\n")
        f.write(f"  Min images per patient: {patient_counts.min()}\n")
        f.write(f"  Max images per patient: {patient_counts.max()}\n")
        
        if 'sex' in mapping_df.columns:
            f.write(f"\nSex Distribution:\n")
            sex_counts = mapping_df['sex'].value_counts()
            for sex, count in sex_counts.items():
                f.write(f"  {sex}: {count} images\n")
                
        if 'age' in mapping_df.columns:
            f.write(f"\nAge Statistics:\n")
            age_data = pd.to_numeric(mapping_df['age'], errors='coerce').dropna()
            if len(age_data) > 0:
                f.write(f"  Mean age: {age_data.mean():.1f} years\n")
                f.write(f"  Age range: {age_data.min():.1f} - {age_data.max():.1f} years\n")
                f.write(f"  Age std: {age_data.std():.1f} years\n")
                
        f.write(f"\nFeature Summary:\n")
        f.write(f"  Total features extracted: {len(features_df.columns)}\n")
        f.write(f"  Feature columns: {', '.join(features_df.columns)}\n")
        
    logger.info(f"✅ Results saved to {output_dir}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
