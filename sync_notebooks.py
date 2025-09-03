#!/usr/bin/env python3
"""
Notebook Sync Helper Script

This script helps manage the auto-sync between Python files and Jupyter notebooks using Jupytext.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return result.stdout, result.stderr, True
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr, False

def sync_python_to_notebook(python_file):
    """Sync changes from Python file to notebook."""
    print(f"🔄 Syncing {python_file} to notebook...")
    cmd = f"jupytext --sync {python_file}"
    stdout, stderr, success = run_command(cmd)
    
    if success:
        print(f"✅ Successfully synced {python_file}")
    else:
        print(f"❌ Failed to sync {python_file}")
        print(f"Error: {stderr}")
    
    return success

def sync_notebook_to_python(notebook_file):
    """Sync changes from notebook to Python file."""
    print(f"🔄 Syncing {notebook_file} to Python...")
    cmd = f"jupytext --sync {notebook_file}"
    stdout, stderr, success = run_command(cmd)
    
    if success:
        print(f"✅ Successfully synced {notebook_file}")
    else:
        print(f"❌ Failed to sync {notebook_file}")
        print(f"Error: {stderr}")
    
    return success

def setup_pairing(python_file):
    """Set up pairing between Python file and notebook."""
    print(f"🔗 Setting up pairing for {python_file}...")
    cmd = f"jupytext --set-formats ipynb,py:percent {python_file}"
    stdout, stderr, success = run_command(cmd)
    
    if success:
        print(f"✅ Successfully paired {python_file}")
    else:
        print(f"❌ Failed to pair {python_file}")
        print(f"Error: {stderr}")
    
    return success

def main():
    """Main function to manage notebook syncing."""
    print("🚀 Jupytext Notebook Sync Helper")
    print("=" * 40)
    
    # Check if jupytext is available
    try:
        import jupytext
        print("✅ Jupytext is available")
    except ImportError:
        print("❌ Jupytext not found. Please install it with: pip install jupytext")
        return
    
    notebooks_dir = Path("notebooks")
    if not notebooks_dir.exists():
        print("❌ Notebooks directory not found")
        return
    
    # Find all Python files in notebooks directory
    python_files = list(notebooks_dir.glob("*.py"))
    
    if not python_files:
        print("❌ No Python files found in notebooks directory")
        return
    
    print(f"\n📁 Found {len(python_files)} Python files:")
    for py_file in python_files:
        print(f"  - {py_file.name}")
    
    print("\n🔧 Available actions:")
    print("1. Setup pairing for all files")
    print("2. Sync Python to notebooks")
    print("3. Sync notebooks to Python")
    print("4. Check pairing status")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🔗 Setting up pairing for all files...")
        for py_file in python_files:
            setup_pairing(py_file)
    
    elif choice == "2":
        print("\n🔄 Syncing Python files to notebooks...")
        for py_file in python_files:
            sync_python_to_notebook(py_file)
    
    elif choice == "3":
        print("\n🔄 Syncing notebooks to Python files...")
        notebook_files = list(notebooks_dir.glob("*.ipynb"))
        for nb_file in notebook_files:
            sync_notebook_to_python(nb_file)
    
    elif choice == "4":
        print("\n📊 Checking pairing status...")
        for py_file in python_files:
            notebook_file = py_file.with_suffix('.ipynb')
            if notebook_file.exists():
                print(f"✅ {py_file.name} ↔ {notebook_file.name}")
            else:
                print(f"❌ {py_file.name} (no notebook found)")
    
    else:
        print("❌ Invalid choice")
        return
    
    print("\n🎉 Sync operation completed!")
    print("\n💡 Tips:")
    print("- Edit Python files (.py) and they'll auto-sync to notebooks")
    print("- Use 'jupytext --sync <file>' to manually sync")
    print("- The .jupytext config file enables automatic syncing")

if __name__ == "__main__":
    main()
