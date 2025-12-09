# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Frontend Dependencies
```bash
npm install
```

### Step 2: Install Backend Dependencies
```bash
cd backend

# Install base dependencies
pip install -r requirements.txt

# Install SAM2 from source (use the script)
python install_sam2.py
```

**If you get Python 3.12 errors**, see `backend/INSTALL_FIX.md` or use Python 3.11/3.10 instead.

### Step 3: Download Model Weights

**Option A: Use the script (Recommended)**
```bash
cd backend
python download_model.py
# Choose "large" when prompted (or tiny/small/base for faster processing)
```

**Option B: Manual download**
```bash
cd backend
mkdir checkpoints
cd checkpoints
# Download large model (recommended)
curl -L -o sam2_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.pt
cd ../..
```

### Step 4: Start the Servers

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
```
Wait for "SAM2 model loaded successfully!" message.

**Terminal 2 - Frontend:**
```bash
npm run dev
```

### Step 5: Open in Browser
Visit: `http://localhost:5173`

Upload a prescription image and see the segmentation results!

## 📋 Model Sizes

- **tiny** (~40MB) - Fastest, less accurate
- **small** (~100MB) - Good balance  
- **base** (~200MB) - Better accuracy
- **large** (~1GB) - Best accuracy ⭐ Recommended

## ⚠️ Troubleshooting

**Model not found?**
- Check `backend/checkpoints/` folder exists
- Verify the `.pt` file is there
- Check filename matches `main.py`

**Backend won't start?**
- Make sure Python 3.8+ is installed
- Check all dependencies: `pip install -r requirements.txt`
- Install SAM2 from source if needed (see Step 2)

**Frontend can't connect?**
- Ensure backend is running on port 8000
- Check `http://localhost:8000/health` in browser
- Verify CORS settings in `backend/main.py`

## 📚 Full Documentation

See `SETUP.md` for detailed instructions and troubleshooting.

