"""
Clinical data loading and processing utilities for PPMI dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalDataLoader:
    """Loader for PPMI clinical data."""
    
    def __init__(self, clinical_data_path: str):
        """Initialize clinical data loader.
        
        Args:
            clinical_data_path: Path to directory containing clinical data CSVs
        """
        self.clinical_data_path = Path(clinical_data_path)
        if not self.clinical_data_path.exists():
            raise FileNotFoundError(f"Clinical data path does not exist: {clinical_data_path}")
            
        self.patient_status_df = None
        self.enrollment_df = None
        self.imaging_metadata_df = None
        
    def load_clinical_data(self) -> Dict[str, pd.DataFrame]:
        """Load all clinical data files.
        
        Returns:
            Dictionary containing all clinical data DataFrames
        """
        logger.info("Loading clinical data files...")
        
        # Load patient status data
        patient_status_file = self.clinical_data_path / "Patient_Status.csv"
        if patient_status_file.exists():
            self.patient_status_df = pd.read_csv(patient_status_file)
            logger.info(f"Loaded patient status data: {self.patient_status_df.shape}")
        else:
            logger.warning("Patient_Status.csv not found")
            
        # Load enrollment data
        enrollment_file = self.clinical_data_path / "Enrollment.csv"
        if enrollment_file.exists():
            self.enrollment_df = pd.read_csv(enrollment_file)
            logger.info(f"Loaded enrollment data: {self.enrollment_df.shape}")
        else:
            logger.warning("Enrollment.csv not found")
            
        # Load imaging metadata
        imaging_metadata_file = self.clinical_data_path / "Imaging_Metadata.csv"
        if imaging_metadata_file.exists():
            self.imaging_metadata_df = pd.read_csv(imaging_metadata_file)
            logger.info(f"Loaded imaging metadata: {self.imaging_metadata_df.shape}")
        else:
            logger.warning("Imaging_Metadata.csv not found")
            
        return {
            'patient_status': self.patient_status_df,
            'enrollment': self.enrollment_df,
            'imaging_metadata': self.imaging_metadata_df
        }
    
    def get_patient_diagnoses(self) -> pd.DataFrame:
        """Extract patient diagnoses from clinical data.
        
        Returns:
            DataFrame with patient ID and diagnosis information
        """
        if self.patient_status_df is None:
            raise ValueError("Patient status data not loaded. Call load_clinical_data() first.")
            
        # Look for diagnosis-related columns
        diagnosis_columns = []
        for col in self.patient_status_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['diagnosis', 'status', 'group', 'type']):
                diagnosis_columns.append(col)
                
        if not diagnosis_columns:
            logger.warning("No diagnosis columns found in patient status data")
            # Try to infer from other columns
            diagnosis_columns = self.patient_status_df.columns[:3]  # First few columns
            
        logger.info(f"Using diagnosis columns: {diagnosis_columns}")
        
        # Create diagnosis DataFrame
        diagnosis_df = self.patient_status_df[['PATNO'] + diagnosis_columns].copy()
        
        # Clean up column names
        diagnosis_df.columns = ['patient_id'] + [f'diagnosis_{i+1}' for i in range(len(diagnosis_columns))]
        
        return diagnosis_df
    
    def get_patient_demographics(self) -> pd.DataFrame:
        """Extract patient demographic information.
        
        Returns:
            DataFrame with patient demographics
        """
        if self.enrollment_df is None:
            raise ValueError("Enrollment data not loaded. Call load_clinical_data() first.")
            
        # Look for demographic columns
        demo_columns = ['PATNO', 'SEX', 'BIRTHDT', 'ENROLL_DATE']
        available_columns = [col for col in demo_columns if col in self.enrollment_df.columns]
        
        if not available_columns:
            logger.warning("No demographic columns found")
            return pd.DataFrame()
            
        demographics_df = self.enrollment_df[available_columns].copy()
        
        # Clean up column names
        demographics_df.columns = ['patient_id'] + [col.lower() for col in available_columns[1:]]
        
        # Convert dates
        for col in ['birthdt', 'enroll_date']:
            if col in demographics_df.columns:
                demographics_df[col] = pd.to_datetime(demographics_df[col], errors='coerce')
                
        return demographics_df
    
    def get_imaging_metadata(self) -> pd.DataFrame:
        """Extract imaging metadata information.
        
        Returns:
            DataFrame with imaging metadata
        """
        if self.imaging_metadata_df is None:
            raise ValueError("Imaging metadata not loaded. Call load_clinical_data() first.")
            
        # Look for relevant imaging columns
        imaging_columns = ['PATNO', 'EXAMDATE', 'IMAGING_TYPE', 'SCAN_DATE']
        available_columns = [col for col in imaging_columns if col in self.imaging_metadata_df.columns]
        
        if not available_columns:
            logger.warning("No imaging metadata columns found")
            return pd.DataFrame()
            
        imaging_df = self.imaging_metadata_df[available_columns].copy()
        
        # Clean up column names
        imaging_df.columns = ['patient_id'] + [col.lower() for col in available_columns[1:]]
        
        # Convert dates
        for col in ['examdate', 'scan_date']:
            if col in imaging_df.columns:
                imaging_df[col] = pd.to_datetime(imaging_df[col], errors='coerce')
                
        return imaging_df
    
    def create_patient_master_list(self) -> pd.DataFrame:
        """Create a master list linking patients to their diagnoses and metadata.
        
        Returns:
            DataFrame with comprehensive patient information
        """
        logger.info("Creating patient master list...")
        
        # Get individual data components
        diagnosis_df = self.get_patient_diagnoses()
        demographics_df = self.get_patient_demographics()
        imaging_df = self.get_imaging_metadata()
        
        # Start with diagnosis data
        master_df = diagnosis_df.copy()
        
        # Merge with demographics
        if not demographics_df.empty:
            master_df = master_df.merge(demographics_df, on='patient_id', how='left')
            
        # Merge with imaging metadata
        if not imaging_df.empty:
            master_df = master_df.merge(imaging_df, on='patient_id', how='left')
            
        # Clean up the data
        master_df = self._clean_patient_data(master_df)
        
        logger.info(f"Created master list with {len(master_df)} patients")
        return master_df
    
    def _clean_patient_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize patient data.
        
        Args:
            df: Raw patient data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Remove duplicates
        df = df.drop_duplicates(subset=['patient_id'])
        
        # Handle missing values
        df = df.replace(['', 'nan', 'NaN', 'NULL'], np.nan)
        
        # Convert diagnosis columns to categorical
        diagnosis_cols = [col for col in df.columns if col.startswith('diagnosis_')]
        for col in diagnosis_cols:
            df[col] = df[col].astype('category')
            
        # Add binary PD indicator
        df = self._add_pd_indicator(df)
        
        return df
    
    def _add_pd_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary Parkinson's disease indicator.
        
        Args:
            df: Patient data DataFrame
            
        Returns:
            DataFrame with PD indicator added
        """
        # Look for PD-related values in diagnosis columns
        diagnosis_cols = [col for col in df.columns if col.startswith('diagnosis_')]
        
        if not diagnosis_cols:
            logger.warning("No diagnosis columns found for PD indicator")
            df['is_pd'] = np.nan
            return df
            
        # Create PD indicator
        df['is_pd'] = False
        
        for col in diagnosis_cols:
            # Look for PD-related values
            pd_values = ['PD', 'Parkinson', 'Parkinson\'s', 'Parkinsons']
            for pd_val in pd_values:
                mask = df[col].astype(str).str.contains(pd_val, case=False, na=False)
                df.loc[mask, 'is_pd'] = True
                
        # Look for control/healthy values
        control_values = ['Control', 'Healthy', 'Normal', 'HC']
        for col in diagnosis_cols:
            for control_val in control_values:
                mask = df[col].astype(str).str.contains(control_val, case=False, na=False)
                df.loc[mask, 'is_pd'] = False
                
        logger.info(f"PD indicator created: {df['is_pd'].sum()} PD patients, {(~df['is_pd']).sum()} controls")
        
        return df
    
    def get_pd_controls_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split patients into PD and control groups.
        
        Returns:
            Tuple of (pd_patients, control_patients) DataFrames
        """
        master_df = self.create_patient_master_list()
        
        if 'is_pd' not in master_df.columns:
            raise ValueError("PD indicator not found. Check diagnosis data.")
            
        # Split by PD status
        pd_patients = master_df[master_df['is_pd'] == True].copy()
        control_patients = master_df[master_df['is_pd'] == False].copy()
        
        # Remove patients with unclear status
        unclear_patients = master_df[master_df['is_pd'].isna()].copy()
        if len(unclear_patients) > 0:
            logger.warning(f"Found {len(unclear_patients)} patients with unclear PD status")
            
        logger.info(f"Split complete: {len(pd_patients)} PD patients, {len(control_patients)} controls")
        
        return pd_patients, control_patients
    
    def save_patient_master_list(self, output_path: Path, filename: str = "patient_master_list.csv"):
        """Save patient master list to CSV.
        
        Args:
            output_path: Directory to save the file
            filename: Name of the output file
        """
        master_df = self.create_patient_master_list()
        
        output_file = output_path / filename
        master_df.to_csv(output_file, index=False)
        
        logger.info(f"Saved patient master list to {output_file}")
        
        # Also save summary statistics
        summary_file = output_path / "patient_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("PPMI Patient Data Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Total patients: {len(master_df)}\n")
            f.write(f"PD patients: {master_df['is_pd'].sum()}\n")
            f.write(f"Controls: {(~master_df['is_pd']).sum()}\n")
            f.write(f"Unclear status: {master_df['is_pd'].isna().sum()}\n\n")
            
            if 'sex' in master_df.columns:
                f.write("Sex distribution:\n")
                f.write(str(master_df['sex'].value_counts()) + "\n\n")
                
            if 'birthdt' in master_df.columns:
                f.write("Age distribution:\n")
                df_with_age = master_df.dropna(subset=['birthdt'])
                if len(df_with_age) > 0:
                    ages = (pd.Timestamp.now() - df_with_age['birthdt']).dt.total_seconds() / (365.25 * 24 * 3600)
                    f.write(f"Mean age: {ages.mean():.1f} years\n")
                    f.write(f"Age range: {ages.min():.1f} - {ages.max():.1f} years\n")
                    
        logger.info(f"Saved patient summary to {summary_file}")


def load_ppmi_clinical_data(clinical_data_path: str) -> ClinicalDataLoader:
    """Convenience function to load PPMI clinical data.
    
    Args:
        clinical_data_path: Path to clinical data directory
        
    Returns:
        ClinicalDataLoader instance with loaded data
    """
    loader = ClinicalDataLoader(clinical_data_path)
    loader.load_clinical_data()
    return loader
