#!/usr/bin/env python3
"""
PPMI DaTSCAN Data Analysis Script
Analyzes control vs PD patient data from the PPMI dataset
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pydicom
import warnings
warnings.filterwarnings('ignore')

class PPMIDataAnalyzer:
    def __init__(self, control_dir="control", pd_dir="PD"):
        self.control_dir = Path(control_dir)
        self.pd_dir = Path(pd_dir)
        self.control_data = []
        self.pd_data = []
        
    def parse_xml_metadata(self, xml_file):
        """Parse XML metadata file and extract relevant information"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            data = {}
            
            # Basic subject info - search without namespace since elements don't have ns prefix
            subject = root.find('.//subject')
            if subject is not None:
                subject_id_elem = subject.find('subjectIdentifier')
                if subject_id_elem is not None and subject_id_elem.text:
                    data['subject_id'] = subject_id_elem.text
                else:
                    data['subject_id'] = 'Unknown'
                
                research_group_elem = subject.find('researchGroup')
                if research_group_elem is not None and research_group_elem.text:
                    data['research_group'] = research_group_elem.text
                else:
                    data['research_group'] = 'Unknown'
                
                sex_elem = subject.find('subjectSex')
                if sex_elem is not None and sex_elem.text:
                    data['sex'] = sex_elem.text
                else:
                    data['sex'] = 'Unknown'
            
            # Visit info
            visit = root.find('.//visit')
            if visit is not None:
                visit_id_elem = visit.find('visitIdentifier')
                if visit_id_elem is not None and visit_id_elem.text:
                    data['visit_type'] = visit_id_elem.text
                else:
                    data['visit_type'] = 'Unknown'
            
            # Study info
            study = root.find('.//study')
            if study is not None:
                study_id_elem = study.find('studyIdentifier')
                if study_id_elem is not None and study_id_elem.text:
                    data['study_id'] = study_id_elem.text
                else:
                    data['study_id'] = 'Unknown'
                
                age_elem = study.find('subjectAge')
                if age_elem is not None and age_elem.text:
                    try:
                        data['age'] = float(age_elem.text)
                    except ValueError:
                        data['age'] = None
                else:
                    data['age'] = None
                
                age_qual_elem = study.find('ageQualifier')
                if age_qual_elem is not None and age_qual_elem.text:
                    data['age_qualifier'] = age_qual_elem.text
                else:
                    data['age_qualifier'] = 'Unknown'
            
            # Series info
            series = root.find('.//series')
            if series is not None:
                series_id_elem = series.find('seriesIdentifier')
                if series_id_elem is not None and series_id_elem.text:
                    data['series_id'] = series_id_elem.text
                else:
                    data['series_id'] = 'Unknown'
                
                modality_elem = series.find('modality')
                if modality_elem is not None and modality_elem.text:
                    data['modality'] = modality_elem.text
                else:
                    data['modality'] = 'Unknown'
                
                date_elem = series.find('dateAcquired')
                if date_elem is not None and date_elem.text:
                    data['date_acquired'] = date_elem.text
                else:
                    data['date_acquired'] = 'Unknown'
            
            # Protocol info - look for protocol elements with term attributes
            protocol_terms = root.findall('.//protocolTerm')
            for term in protocol_terms:
                # Find all protocol elements within this protocolTerm
                protocols = term.findall('protocol')
                for i, protocol in enumerate(protocols):
                    # The protocol elements have term attributes
                    term_name = protocol.get('term')
                    term_value = protocol.text
                    if term_name and term_value:
                        data[f'protocol_{term_name.lower().replace(" ", "_")}'] = term_value
                    elif term_value:  # If no term attribute, use index
                        data[f'protocol_param_{i}'] = term_value
            
            return data
            
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            return {}
    
    def collect_metadata(self):
        """Collect metadata from all XML files"""
        print("Collecting control group metadata...")
        control_metadata_dir = self.control_dir / "metadata"
        for xml_file in control_metadata_dir.glob("*.xml"):
            data = self.parse_xml_metadata(xml_file)
            if data:
                data['group'] = 'Control'
                data['source_file'] = str(xml_file)
                self.control_data.append(data)
        
        print("Collecting PD group metadata...")
        pd_metadata_dir = self.pd_dir / "metadata"
        for xml_file in pd_metadata_dir.glob("*.xml"):
            data = self.parse_xml_metadata(xml_file)
            if data:
                data['group'] = 'PD'
                data['source_file'] = str(xml_file)
                self.pd_data.append(data)
        
        print(f"Collected metadata for {len(self.control_data)} control subjects and {len(self.pd_data)} PD subjects")
    
    def analyze_demographics(self):
        """Analyze demographic information"""
        print("\n=== DEMOGRAPHIC ANALYSIS ===")
        
        # Combine all data
        all_data = self.control_data + self.pd_data
        if not all_data:
            print("No metadata collected. Check XML files and parsing.")
            return None
            
        df = pd.DataFrame(all_data)
        
        # Basic counts
        print(f"Total subjects: {len(df)}")
        print(f"Control subjects: {len(df[df['group'] == 'Control'])}")
        print(f"PD subjects: {len(df[df['group'] == 'PD'])}")
        
        # Age analysis
        age_data = df[df['age'].notna()].copy()
        if not age_data.empty:
            print(f"\nAge Statistics (n={len(age_data)}):")
            for group in ['Control', 'PD']:
                group_ages = age_data[age_data['group'] == group]['age']
                if not group_ages.empty:
                    print(f"{group}: Mean={group_ages.mean():.1f}±{group_ages.std():.1f}, Range={group_ages.min():.1f}-{group_ages.max():.1f}")
        
        # Sex distribution
        sex_data = df[df['sex'].notna()].copy()
        if not sex_data.empty:
            print(f"\nSex Distribution:")
            for group in ['Control', 'PD']:
                group_sex = sex_data[sex_data['group'] == group]['sex'].value_counts()
                print(f"{group}: {dict(group_sex)}")
        
        # Visit types
        visit_data = df[df['visit_type'].notna()].copy()
        if not visit_data.empty:
            print(f"\nVisit Types:")
            for group in ['Control', 'PD']:
                group_visits = visit_data[visit_data['group'] == group]['visit_type'].value_counts()
                print(f"{group}: {dict(group_visits)}")
        
        return df
    
    def analyze_imaging_protocols(self):
        """Analyze imaging protocol differences"""
        print("\n=== IMAGING PROTOCOL ANALYSIS ===")
        
        all_data = self.control_data + self.pd_data
        if not all_data:
            print("No data available for protocol analysis")
            return
            
        df = pd.DataFrame(all_data)
        
        # Protocol parameters
        protocol_cols = [col for col in df.columns if col.startswith('protocol_')]
        
        if protocol_cols:
            print("Protocol Parameters Found:")
            for col in protocol_cols:
                print(f"  {col}")
            
            # Compare key parameters between groups
            key_params = ['protocol_manufacturer', 'protocol_mgf_model', 'protocol_pixel_spacing_row', 
                         'protocol_pixel_spacing_col', 'protocol_frame_duration']
            
            for param in key_params:
                if param in df.columns:
                    print(f"\n{param.replace('protocol_', '').replace('_', ' ').title()}:")
                    for group in ['Control', 'PD']:
                        group_data = df[df['group'] == group][param].dropna()
                        if not group_data.empty:
                            unique_values = group_data.unique()
                            print(f"  {group}: {list(unique_values)}")
    
    def find_dicom_files(self):
        """Find and count DICOM files"""
        print("\n=== DICOM FILE ANALYSIS ===")
        
        # Count DICOM files in control group
        control_dcm_count = 0
        control_dcm_files = []
        for dcm_file in (self.control_dir / "PPMI").rglob("*.dcm"):
            control_dcm_count += 1
            control_dcm_files.append(str(dcm_file))
        
        # Count DICOM files in PD group
        pd_dcm_count = 0
        pd_dcm_files = []
        for dcm_file in (self.pd_dir / "PPMI").rglob("*.dcm"):
            pd_dcm_count += 1
            pd_dcm_files.append(str(dcm_file))
        
        print(f"Control group DICOM files: {control_dcm_count}")
        print(f"PD group DICOM files: {pd_dcm_count}")
        print(f"Total DICOM files: {control_dcm_count + pd_dcm_count}")
        
        return control_dcm_files, pd_dcm_files
    
    def analyze_dicom_headers(self, sample_files):
        """Analyze DICOM headers from sample files"""
        print("\n=== DICOM HEADER ANALYSIS ===")
        
        if not sample_files:
            print("No DICOM files found for analysis")
            return
        
        # Analyze first few files from each group
        sample_size = min(3, len(sample_files))
        sample_files = sample_files[:sample_size]
        
        for i, dcm_file in enumerate(sample_files):
            try:
                ds = pydicom.dcmread(dcm_file)
                print(f"\nFile {i+1}: {os.path.basename(dcm_file)}")
                print(f"  Modality: {getattr(ds, 'Modality', 'Unknown')}")
                print(f"  Image Size: {getattr(ds, 'Rows', 'Unknown')} x {getattr(ds, 'Columns', 'Unknown')}")
                print(f"  Pixel Spacing: {getattr(ds, 'PixelSpacing', 'Unknown')}")
                print(f"  Slice Thickness: {getattr(ds, 'SliceThickness', 'Unknown')}")
                print(f"  Study Date: {getattr(ds, 'StudyDate', 'Unknown')}")
                
            except Exception as e:
                print(f"Error reading {dcm_file}: {e}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*60)
        print("PPMI DaTSCAN DATA ANALYSIS SUMMARY REPORT")
        print("="*60)
        
        # Collect all data
        self.collect_metadata()
        
        # Analyze demographics
        df = self.analyze_demographics()
        
        # Analyze imaging protocols
        self.analyze_imaging_protocols()
        
        # Find DICOM files
        control_dcm_files, pd_dcm_files = self.find_dicom_files()
        
        # Analyze DICOM headers
        if control_dcm_files:
            print(f"\nAnalyzing sample control DICOM files...")
            self.analyze_dicom_headers(control_dcm_files)
        
        if pd_dcm_files:
            print(f"\nAnalyzing sample PD DICOM files...")
            self.analyze_dicom_headers(pd_dcm_files)
        
        # Save data to CSV for further analysis
        if df is not None and not df.empty:
            output_file = "ppmi_metadata_summary.csv"
            df.to_csv(output_file, index=False)
            print(f"\nMetadata summary saved to: {output_file}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)

def main():
    """Main function to run the analysis"""
    analyzer = PPMIDataAnalyzer()
    analyzer.generate_summary_report()

if __name__ == "__main__":
    main()
