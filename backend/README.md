# Backend API for Prescription OCR with SAM2 Segmentation

This backend provides image segmentation using Meta's Segment Anything Model 2 (SAM2) to differentiate between handwritten and printed text in prescription images.

## Setup Instructions

### 1. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Important:** SAM2 must be installed from source. Use the installation script:

```bash
python install_sam2.py
```

Or manually:

```bash
# Upgrade pip first
python -m pip install --upgrade pip setuptools wheel

# Install base dependencies
pip install -r requirements.txt

# Install SAM2 from source
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
cd ..
```

**Note for Python 3.12 users:** If you encounter `pkgutil.ImpImporter` errors, see `INSTALL_FIX.md` for solutions. Python 3.11 or 3.10 is recommended for best compatibility.

### 2. Download SAM2 Model Weights

You need to download the SAM2 model checkpoint. Choose one of the following models:

- **sam2_hiera_tiny.pt** (Smallest, fastest) - ~40MB
- **sam2_hiera_small.pt** (Small) - ~100MB
- **sam2_hiera_base.pt** (Base) - ~200MB
- **sam2_hiera_large.pt** (Large, most accurate) - ~1GB

#### Option A: Download from Hugging Face (Recommended)

1. Go to: https://huggingface.co/facebook/sam2-hiera/tree/main
2. Download the checkpoint file you want (e.g., `sam2_hiera_large.pt`)
3. Create a `checkpoints` folder in the backend directory:
   ```bash
   mkdir checkpoints
   ```
4. Move the downloaded file to `backend/checkpoints/`

#### Option B: Download from Official Source

1. Visit: https://github.com/facebookresearch/segment-anything-2
2. Follow their instructions to download model weights
3. Place the `.pt` file in `backend/checkpoints/`

#### Option C: Use Python Script

Create and run this script:

```python
import urllib.request
import os

os.makedirs("checkpoints", exist_ok=True)

# Download sam2_hiera_large.pt
url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.pt"
urllib.request.urlretrieve(url, "checkpoints/sam2_hiera_large.pt")
print("Download complete!")
```

### 3. Update Model Configuration

Edit `backend/main.py` and change the model configuration if you downloaded a different model:

```python
model_cfg = "sam2_hiera_large.yaml"  # Change to match your model
checkpoint_path = "checkpoints/sam2_hiera_large.pt"  # Change to match your model
```

Available configurations:
- `sam2_hiera_tiny.yaml`
- `sam2_hiera_small.yaml`
- `sam2_hiera_base.yaml`
- `sam2_hiera_large.yaml`

### 4. Run the Server

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 5. API Endpoints

- `GET /` - API information
- `GET /health` - Check if model is loaded
- `POST /api/segment` - Upload image and get segmentation results

### 6. Testing the API

You can test the API using curl:

```bash
curl -X POST "http://localhost:8000/api/segment" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```

## Notes

- The first request may take longer as the model loads
- GPU is recommended for faster processing (CUDA)
- The classification between handwritten and printed text uses heuristics and can be improved with a dedicated ML model
- For production, consider using a more sophisticated text classification model

## Troubleshooting

1. **Model not found error**: Make sure the checkpoint file is in `backend/checkpoints/` with the correct name
2. **CUDA out of memory**: Use a smaller model (tiny or small) or reduce image size
3. **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

