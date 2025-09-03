# 🚀 Jupytext Auto-Sync Setup

## Overview
This project uses **Jupytext** to automatically sync between Python files (`.py`) and Jupyter notebooks (`.ipynb`). This enables a powerful workflow where you can:

- **Edit Python files** in your preferred editor (VS Code, PyCharm, etc.)
- **Run notebooks** in Jupyter with automatic updates
- **Version control** both formats seamlessly
- **Collaborate** with team members who prefer different formats

## 🔧 Setup Complete!

Your auto-sync is already configured! Here's what's been set up:

### ✅ Paired Files
- `notebooks/01_data_exploration.py` ↔ `notebooks/01_data_exploration.ipynb`
- `notebooks/02_3d_volume_analysis.py` ↔ `notebooks/02_3d_volume_analysis.ipynb`
- `notebooks/03_sbr_baseline_modeling.py` ↔ `notebooks/03_sbr_baseline_modeling.ipynb`

### ✅ Configuration Files
- `.jupytext` - Global configuration for auto-sync
- `sync_notebooks.py` - Helper script for managing sync operations

## 🎯 How to Use

### **Option 1: Edit Python Files (Recommended)**
1. **Edit** any `.py` file in your preferred editor
2. **Save** the file
3. **Run** the corresponding `.ipynb` in Jupyter - it will automatically update!

### **Option 2: Edit Notebooks**
1. **Edit** any `.ipynb` file in Jupyter
2. **Save** the notebook
3. **Run** `jupytext --sync <notebook>` to update the Python file

### **Option 3: Manual Sync**
```bash
# Sync Python to notebook
jupytext --sync notebooks/01_data_exploration.py

# Sync notebook to Python
jupytext --sync notebooks/01_data_exploration.ipynb

# Sync all files
jupytext --sync notebooks/
```

## 🛠️ Helper Script

Use the included sync helper script:

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Run the helper
python sync_notebooks.py
```

**Available Actions:**
1. **Setup pairing** for all files
2. **Sync Python to notebooks**
3. **Sync notebooks to Python**
4. **Check pairing status**

## 📁 File Structure

```
notebooks/
├── 01_data_exploration.py          # Edit this file
├── 01_data_exploration.ipynb       # Auto-updates from .py
├── 02_3d_volume_analysis.py        # Edit this file
├── 02_3d_volume_analysis.ipynb     # Auto-updates from .py
├── 03_sbr_baseline_modeling.py     # Edit this file
└── 03_sbr_baseline_modeling.ipynb  # Auto-updates from .py
```

## 🔄 Workflow Examples

### **Example 1: Adding a New Cell**
1. **Edit** `01_data_exploration.py`:
   ```python
   # Cell 8: New Analysis
   print("New analysis cell added!")
   ```
2. **Save** the file
3. **Open** `01_data_exploration.ipynb` in Jupyter
4. **See** the new cell automatically appears!

### **Example 2: Modifying Existing Code**
1. **Edit** any cell in the Python file
2. **Save** the file
3. **Refresh** the notebook in Jupyter
4. **Changes** are automatically applied!

### **Example 3: Adding New Functions**
1. **Add** new functions to the Python file
2. **Save** the file
3. **Use** the functions in your notebook immediately!

## 🚨 Important Notes

### **Cell Structure**
- Each cell in the Python file starts with `# Cell X: Description`
- Jupytext automatically converts these to notebook cells
- **Don't delete** the cell markers!

### **File Formats**
- **Python files**: Use `# Cell X:` format for cell separation
- **Notebooks**: Standard Jupyter notebook format
- **Both** are automatically kept in sync

### **Saving**
- **Always save** your Python files after editing
- **Notebooks** will update automatically when you open them
- **Manual sync** may be needed if you edit notebooks directly

## 🔧 Troubleshooting

### **Sync Not Working?**
```bash
# Check pairing status
python sync_notebooks.py

# Re-establish pairing
jupytext --set-formats ipynb,py:percent notebooks/your_file.py
```

### **Notebook Not Updating?**
```bash
# Force sync
jupytext --sync notebooks/your_file.py

# Check file permissions
ls -la notebooks/
```

### **Jupytext Not Found?**
```bash
# Install in virtual environment
source venv/bin/activate
pip install jupytext
```

## 🎉 Benefits

### **For You:**
- **Edit in your preferred editor** (VS Code, PyCharm, Vim, etc.)
- **Run in Jupyter** with full interactive features
- **Version control** both formats
- **No manual copying** between files

### **For Collaboration:**
- **Team members** can use their preferred format
- **Git diffs** are clean and readable
- **Merge conflicts** are minimized
- **Consistent** code structure

## 🚀 Next Steps

1. **Start editing** the Python files in your preferred editor
2. **Open notebooks** in Jupyter to run and test
3. **Use the helper script** for any sync issues
4. **Enjoy** the seamless workflow!

## 📚 Additional Resources

- [Jupytext Documentation](https://jupytext.readthedocs.io/)
- [Jupytext GitHub](https://github.com/mwouts/jupytext)
- [Jupyter Notebook Best Practices](https://jupyter-notebook.readthedocs.io/en/stable/)

---

**Happy coding! 🎯** Your auto-sync setup is ready to go!
