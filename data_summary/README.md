# PPMI DaTSCAN Data Analysis Report

## Executive Summary

This report presents a comprehensive analysis of the PPMI (Parkinson's Progression Markers Initiative) DaTSCAN dataset, comparing 20 healthy control subjects with 20 Parkinson's Disease (PD) patients. The analysis covers demographic characteristics, imaging protocols, and DICOM file properties.

## Dataset Overview

- **Total Subjects**: 40
- **Control Group**: 20 subjects (healthy individuals)
- **PD Group**: 20 subjects (Parkinson's Disease patients)
- **Data Type**: DaTSCAN SPECT imaging data
- **Format**: DICOM files with XML metadata

## Demographic Analysis

### Age Distribution
- **Control Group**: Mean = 54.4 ± 12.4 years, Range = 30.6 - 76.8 years
- **PD Group**: Mean = 64.1 ± 5.7 years, Range = 50.7 - 74.9 years

**Key Finding**: The PD group is significantly older (mean difference: 9.7 years) and has less age variability compared to the control group.

### Sex Distribution
- **Control Group**: 11 Male (55%), 9 Female (45%)
- **PD Group**: 14 Male (70%), 6 Female (30%)

**Key Finding**: Higher male predominance in the PD group, which aligns with known epidemiological patterns of Parkinson's disease.

### Visit Types
- **Screening Visit**: 36 subjects (90%)
- **Unscheduled Visit 01**: 4 subjects (10%)

## Imaging Protocol Analysis

### Scanner Manufacturers
- **Control Group**: PICKER, SIEMENS NM, GE MEDICAL SYSTEMS
- **PD Group**: SIEMENS NM, PICKER, Marconi Medical Systems, NM Division

### Key Protocol Parameters

#### Pixel Spacing
- **Control Group**: Range from 2.7 to 3.9 mm
- **PD Group**: Range from 2.7 to 4.8 mm

#### Frame Duration
- **Control Group**: Primarily 15,000-20,000 ms
- **PD Group**: More variable, ranging from 15,000 to 40,000 ms

## DICOM File Analysis

### Image Characteristics
- **Modality**: NM (Nuclear Medicine)
- **Image Size**: 109 x 91 pixels (consistent across groups)
- **Pixel Spacing**: 2.0 x 2.0 mm (reconstructed data)
- **Slice Thickness**: 2.0 mm

### Data Quality
- All DICOM files are successfully readable
- Consistent image dimensions across subjects
- Standardized reconstruction parameters

## Key Insights

### 1. Age Disparity
The significant age difference between groups (9.7 years) suggests the need for age-matching in future analyses or age-adjusted statistical models.

### 2. Protocol Variability
Both groups show considerable variation in imaging protocols, which may impact image quality and comparability. This variability should be considered in analysis pipelines.

### 3. Data Consistency
Despite protocol differences, the reconstructed DICOM data shows remarkable consistency in image dimensions and slice thickness, indicating standardized post-processing.

### 4. Male Predominance in PD
The higher male-to-female ratio in the PD group (2.33:1) is consistent with known Parkinson's disease epidemiology.

## Recommendations

### For Analysis
1. **Age Adjustment**: Use age as a covariate in statistical models
2. **Protocol Standardization**: Consider protocol parameters as confounding variables
3. **Quality Control**: Implement automated quality checks for DICOM consistency

### For Future Studies
1. **Age Matching**: Consider age-matched control groups
2. **Protocol Harmonization**: Standardize imaging protocols across sites
3. **Longitudinal Analysis**: Leverage the PPMI longitudinal design for progression studies

## Technical Notes

- **XML Parsing**: Successfully extracted metadata from 40 XML files
- **DICOM Reading**: All 40 DICOM files are accessible and readable
- **Data Export**: Complete metadata exported to CSV format for further analysis
- **Visualization**: Comprehensive plots generated showing group comparisons

## Files Generated

1. `ppmi_metadata_summary.csv` - Complete metadata in tabular format
2. `ppmi_data_visualization.png` - Comprehensive visualization plots
3. `data_analysis.py` - Analysis script for reproducibility
4. `visualize_data.py` - Visualization script for data exploration

## Conclusion

The PPMI DaTSCAN dataset provides a well-characterized collection of SPECT imaging data suitable for Parkinson's disease research. The demographic differences between groups highlight the importance of proper statistical controls, while the protocol variability suggests the need for careful consideration of technical factors in analysis pipelines.

The data quality is high, with consistent DICOM formatting and comprehensive metadata, making this dataset valuable for both clinical research and methodological development in neuroimaging analysis.

---

*Report generated on: $(date)*
*Analysis performed using Python with pandas, matplotlib, seaborn, and pydicom libraries*
