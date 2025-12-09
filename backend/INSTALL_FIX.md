# Fix for Python 3.12 Installation Issues

If you're getting the `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'` error, this is a Python 3.12 compatibility issue.

## Quick Fix

### Option 1: Use the Installation Script (Recommended)

```bash
cd backend
python install_sam2.py
```

This script will:
1. Upgrade pip and setuptools
2. Clone SAM2 repository
3. Install it properly

### Option 2: Manual Installation Steps

```bash
cd backend

# 1. Upgrade pip and setuptools
python -m pip install --upgrade pip setuptools wheel

# 2. Install base dependencies first
pip install -r requirements.txt

# 3. Clone SAM2 repository
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2

# 4. Install SAM2
pip install -e .

# 5. Go back to backend directory
cd ..
```

### Option 3: Use Python 3.11 or 3.10 (Most Reliable)

If you continue having issues with Python 3.12, consider using Python 3.11 or 3.10:

1. Install Python 3.11 from https://www.python.org/downloads/
2. Create a virtual environment:
   ```bash
   python3.11 -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. Then follow the normal installation steps

### Option 4: Use Pre-built Package (If Available)

Check if there's a pre-built wheel available:
```bash
pip install segment-anything-2 --no-build-isolation
```

## Verify Installation

After installation, verify SAM2 is installed:

```bash
python -c "from sam2 import build_sam2; print('SAM2 installed successfully!')"
```

If this works, you're good to go!

## Still Having Issues?

1. **Check Python version**: `python --version` (should be 3.8-3.11 for best compatibility)
2. **Update pip**: `python -m pip install --upgrade pip`
3. **Clear pip cache**: `pip cache purge`
4. **Use virtual environment**: Always recommended to avoid conflicts



