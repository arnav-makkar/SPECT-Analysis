# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: jupytext,-all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: jupytext,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %%
"""
# PPMI Data Exploration & Analysis

This notebook provides comprehensive exploration and analysis of the PPMI dataset.
"""

# %%
# Cell 1: Imports and Setup
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent / 'src'))

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# %%
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# %%
print("Libraries imported successfully!")

# %%
# Cell 2: Load PPMI Data
from data.ppmi_custom_loader import load_ppmi_data

# %%
print("Loading PPMI data...")
mapping_df, summary = load_ppmi_data()

# %%
print(f"\nDataset loaded: {summary['total_images']} images from {summary['unique_patients']} patients")
print(f"\nData folder distribution:")
for folder, count in summary['data_folders'].items():
    print(f"  {folder}: {count} images")

# %%
mapping_df.head()

# %%
# Cell 3: Data Overview & Visualization
# Basic dataset statistics
print("Dataset Overview:")
print(f"Shape: {mapping_df.shape}")
print(f"Columns: {list(mapping_df.columns)}")
print(f"\nMissing values:")
print(mapping_df.isnull().sum())

# %%
# Visualize data distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# %%
# Data folder distribution
axes[0,0].pie(summary['data_folders'].values(), labels=summary['data_folders'].keys(), autopct='%1.1f%%')
axes[0,0].set_title('Images by Data Folder')

# %%
# Patient distribution
patient_counts = mapping_df['patient_id'].value_counts()
axes[0,1].hist(patient_counts.values, bins=20, alpha=0.7, edgecolor='black')
axes[0,1].set_title('Images per Patient Distribution')
axes[0,1].set_xlabel('Number of Images')
axes[0,1].set_ylabel('Number of Patients')

# %%
# Sex distribution (if available)
if 'sex' in mapping_df.columns:
    sex_counts = mapping_df['sex'].value_counts()
    axes[1,0].bar(sex_counts.index, sex_counts.values, alpha=0.7)
    axes[1,0].set_title('Sex Distribution')
    axes[1,0].set_ylabel('Number of Images')

# %%
# Age distribution (if available)
if 'age' in mapping_df.columns:
    age_data = pd.to_numeric(mapping_df['age'], errors='coerce').dropna()
    if len(age_data) > 0:
        axes[1,1].hist(age_data, bins=20, alpha=0.7, edgecolor='black')
        axes[1,1].set_title('Age Distribution')
        axes[1,1].set_xlabel('Age (years)')
        axes[1,1].set_ylabel('Number of Images')

# %%
plt.tight_layout()
plt.show()

# %%
# Cell 4: DICOM Image Exploration
from data.dicom_loader import DICOMLoader
import pydicom

# %%
# Load a sample DICOM file
sample_file = mapping_df.iloc[0]['file_path']
print(f"Loading sample DICOM: {sample_file}")

# %%
try:
    ds = pydicom.dcmread(sample_file)
    print(f"\nDICOM metadata:")
    print(f"Patient ID: {getattr(ds, 'PatientID', 'N/A')}")
    print(f"Modality: {getattr(ds, 'Modality', 'N/A')}")
    print(f"Image size: {getattr(ds, 'Rows', 'N/A')} x {getattr(ds, 'Columns', 'N/A')}")
    print(f"Pixel spacing: {getattr(ds, 'PixelSpacing', 'N/A')}")
    print(f"Slice thickness: {getattr(ds, 'SliceThickness', 'N/A')}")
    
    # Display image
    plt.figure(figsize=(10, 8))
    plt.imshow(ds.pixel_array, cmap='hot')
    plt.title(f'SPECT Image - Patient {ds.PatientID if hasattr(ds, "PatientID") else "Unknown"}')
    plt.colorbar(label='Intensity')
    plt.show()

# %%
except Exception as e:
    print(f"Error loading DICOM: {e}")

# %%
# Cell 5: SBR Feature Calculation & Baseline
from features.sbr_calculator import SBRCalculator
from utils.config import get_config

# %%
config = get_config()
sbr_calculator = SBRCalculator(config)

# %%
# Calculate SBR features for a subset
sample_size = min(20, len(mapping_df))
sample_mapping = mapping_df.sample(n=sample_size, random_state=42)

# %%
print(f"Calculating SBR features for {sample_size} sample images...")
sample_features = sbr_calculator.calculate_sbr_dataset(sample_mapping)

# %%
print(f"\nSBR features calculated: {len(sample_features.columns) - 3} features")
print(f"Feature columns: {list(sample_features.columns)}")

# %%
# Show feature summary
sample_features.describe()

# %%
# Cell 6: Baseline Model Performance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# %%
# Prepare features for baseline model
feature_cols = [col for col in sample_features.columns 
                if col not in ['series_path', 'patient_id', 'label']]

# %%
X = sample_features[feature_cols].fillna(0)
y = sample_features['label']

# %%
print(f"Features: {X.shape}")
print(f"Labels: {y.shape}")
print(f"Class distribution: {y.value_counts().to_dict()}")

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# %%
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Train baseline model
print("\nTraining baseline Random Forest model...")
baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_model.fit(X_train_scaled, y_train)

# %%
# Evaluate
y_pred = baseline_model.predict(X_test_scaled)
y_pred_proba = baseline_model.predict_proba(X_test_scaled)[:, 1]

# %%
print("\nBaseline Model Performance:")
print("=" * 40)
print(f"Accuracy: {baseline_model.score(X_test_scaled, y_test):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# %%
# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': baseline_model.feature_importances_
}).sort_values('importance', ascending=False)

# %%
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importance)), feature_importance['importance'])
plt.title('Feature Importance - Baseline Model')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(range(len(feature_importance)), feature_importance['feature'], rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# %%
# Cell 7: Summary & Next Steps
print("🎉 PPMI EDA & Baseline Analysis Complete! 🎉")
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

# %%
print(f"📊 Dataset: {summary['total_images']} images from {summary['unique_patients']} patients")
print(f"📁 Data folders: {', '.join(summary['data_folders'].keys())}")
print(f"🔬 Features calculated: {len(feature_cols)}")
print(f"🎯 Baseline performance: ROC AUC = {roc_auc_score(y_test, y_pred_proba):.3f}")

# %%
print("\n🚀 Next Steps:")
print("1. Run full preprocessing pipeline: python src/main.py")
print("2. Train CNN models for comparison")
print("3. Implement cross-validation")
print("4. Add more sophisticated feature engineering")
# Test comment for auto-sync
