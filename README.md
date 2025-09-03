# PPMI Parkinson's Disease Detection

This project implements automated detection of Parkinson's disease using SPECT imaging data from the PPMI (Parkinson's Progression Markers Initiative) dataset.

## Project Overview

The goal is to develop machine learning models that can distinguish between healthy controls and Parkinson's disease patients based on DaTscan (Dopamine Transporter Scan) SPECT images. The key visual biomarker is the shape of the striatum - healthy individuals show "comma" or "tadpole" shapes, while PD patients show reduced dopamine uptake appearing as "periods" or ovals.

## Project Structure

```
PPMI/
├── data/                   # Data organization
│   ├── raw/               # Raw DICOM files
│   ├── processed/         # Preprocessed images
│   ├── clinical/          # Clinical data CSVs
│   └── metadata/          # Image-to-label mappings
├── src/                   # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature engineering (SBR calculation)
│   ├── models/            # ML models (classical + CNN)
│   ├── evaluation/        # Model evaluation and visualization
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for exploration
├── results/               # Model outputs and visualizations
├── requirements.txt       # Python dependencies
└── config/                # Configuration files
```

## Implementation Phases

### Phase 1: Setup and Data Organization ✅
- [x] Project structure setup
- [ ] Data organization and linking
- [ ] Environment setup

### Phase 2: Exploratory Data Analysis (EDA)
- [ ] DICOM loading and visualization
- [ ] Metadata analysis
- [ ] Class balance assessment

### Phase 3: Preprocessing Pipeline
- [ ] Spatial normalization (registration)
- [ ] Intensity normalization
- [ ] ROI extraction
- [ ] Data splitting by patient

### Phase 4: Modeling
- [ ] Classical ML baseline (SBR features)
- [ ] 2D CNN implementation
- [ ] Transfer learning approaches

### Phase 5: Evaluation
- [ ] Performance metrics
- [ ] Interpretability (Grad-CAM)
- [ ] Comparison with SBR baseline

## Setup Instructions

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Organize Data:**
   - Place PPMI DICOM files in `data/raw/`
   - Place clinical data CSVs in `data/clinical/`
   - Update configuration in `config/data_config.yaml`

3. **Run Analysis:**
```bash
# Start with EDA
jupyter notebook notebooks/01_data_exploration.ipynb

# Run preprocessing pipeline
python src/data/preprocessing.py

# Train models
python src/models/train_classical.py
python src/models/train_cnn.py
```

## Key Features

- **SBR-based Classical Model**: Implements Striatal Binding Ratio calculation as a scientific baseline
- **2D CNN**: Deep learning approach for end-to-end image classification
- **Comprehensive Preprocessing**: Spatial and intensity normalization pipeline
- **Patient-level Splitting**: Prevents data leakage in evaluation
- **Interpretability**: Grad-CAM visualization for CNN decisions

## Expected Outcomes

- Classical SBR model should achieve high performance (AUC > 0.9)
- CNN model should match or exceed SBR performance
- Visual confirmation that models focus on striatal regions
- Reproducible pipeline for PPMI data analysis

## Data Requirements

- PPMI SPECT DICOM files
- Clinical data CSVs with patient diagnoses
- Patient ID mappings between images and clinical data

## Notes

- Start with the SBR baseline to establish scientific validity
- Focus on proper data preprocessing (registration is critical)
- Use patient-level splitting to avoid data leakage
- Validate CNN decisions align with known PD biomarkers
