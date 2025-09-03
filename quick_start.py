#!/usr/bin/env python3
"""
Quick start script for PPMI Parkinson's Disease Detection project.
This script will help you get started quickly with the project.
"""

import os
import sys
from pathlib import Path

def main():
    """Quick start function."""
    
    print("🚀 PPMI Parkinson's Disease Detection - Quick Start")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("README.md").exists():
        print("❌ Error: Please run this script from the project root directory")
        return False
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Check if virtual environment is activated
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  No virtual environment detected")
        print("   Consider creating one: python -m venv venv")
        print("   Then activate it: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
    
    # Check requirements.txt
    if Path("requirements.txt").exists():
        print("✅ Requirements file found")
    else:
        print("❌ Error: requirements.txt not found")
        return False
    
    # Check if dependencies are installed
    try:
        import numpy
        import pandas
        import pydicom
        print("✅ Core dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Please install dependencies: pip install -r requirements.txt")
        return False
    
    # Check project structure
    required_dirs = ["src", "data", "notebooks", "config"]
    missing_dirs = []
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ Directory found: {dir_name}/")
        else:
            missing_dirs.append(dir_name)
            print(f"❌ Missing directory: {dir_name}/")
    
    if missing_dirs:
        print(f"\n⚠️  Missing directories: {', '.join(missing_dirs)}")
        print("   These will be created when you run the pipeline")
    
    # Check data directories
    data_dirs = ["data/raw", "data/clinical", "data/processed", "data/metadata"]
    print("\n📁 Data Directory Status:")
    
    for data_dir in data_dirs:
        if Path(data_dir).exists():
            items = list(Path(data_dir).iterdir())
            print(f"  {data_dir}/: {len(items)} items")
        else:
            print(f"  {data_dir}/: Not created yet")
    
    # Provide next steps
    print("\n🎯 Next Steps:")
    print("1. Place your PPMI DICOM files in data/raw/")
    print("2. Place your clinical data CSVs in data/clinical/")
    print("3. Update config/data_config.yaml with your file names")
    print("4. Run the pipeline: python src/main.py")
    print("5. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
    
    # Check if data is ready
    raw_path = Path("data/raw")
    clinical_path = Path("data/clinical")
    
    if raw_path.exists() and any(raw_path.iterdir()):
        print("\n✅ Raw DICOM data found")
    else:
        print("\n⚠️  No raw DICOM data found in data/raw/")
        print("   Please add your PPMI SPECT images")
    
    if clinical_path.exists() and any(clinical_path.glob("*.csv")):
        print("✅ Clinical data found")
    else:
        print("\n⚠️  No clinical data found in data/clinical/")
        print("   Please add your PPMI clinical CSV files")
    
    print("\n📚 For more information, see README.md")
    print("🔧 For configuration, see config/data_config.yaml")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
