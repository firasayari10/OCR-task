# Setup Guide for Prescription OCR with SAM2 Segmentation

This guide will help you set up the complete system with SAM2 (Segment Anything Model 2) for differentiating handwritten and printed text in prescription images.

## Prerequisites

- **Python 3.8+** (for backend)
- **Node.js 16+** (for frontend)
- **CUDA-capable GPU** (recommended, but CPU will work)
- **8GB+ RAM** (16GB+ recommended)

## Step 1: Frontend Setup

```bash
# Install frontend dependencies
npm install
```

## Step 2: Backend Setup

### 2.1 Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Note:** If you encounter issues installing `segment-anything-2`, you may need to install it from source:

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
cd ..
```

### 2.2 Download SAM2 Model Weights

You need to download the SAM2 model checkpoint. Choose based on your needs:

- **tiny** (~40MB) - Fastest, least accurate
- **small** (~100MB) - Good balance
- **base** (~200MB) - Better accuracy
- **large** (~1GB) - Best accuracy (recommended)

#### Option A: Use the Download Script (Easiest)

```bash
cd backend
python download_model.py
```

Then follow the prompts to select your model size.

#### Option B: Manual Download

1. Visit: https://github.com/facebookresearch/segment-anything-2
2. Download the checkpoint file you want
3. Create `checkpoints` folder: `mkdir checkpoints`
4. Place the `.pt` file in `backend/checkpoints/`

#### Option C: Direct Download URLs

```bash
# Create checkpoints directory
mkdir -p backend/checkpoints
cd backend/checkpoints

# Download large model (recommended)
curl -L -o sam2_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.pt

# Or download smaller models:
# curl -L -o sam2_hiera_base.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_base.pt
# curl -L -o sam2_hiera_small.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_small.pt
# curl -L -o sam2_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_tiny.pt
```

### 2.3 Update Model Configuration (if needed)

If you downloaded a model other than `large`, edit `backend/main.py`:

```python
# Change these lines:
model_cfg = "sam2_hiera_large.yaml"  # Change to match your model
checkpoint_path = "checkpoints/sam2_hiera_large.pt"  # Change to match your model
```

Available options:
- `sam2_hiera_tiny.yaml` / `sam2_hiera_tiny.pt`
- `sam2_hiera_small.yaml` / `sam2_hiera_small.pt`
- `sam2_hiera_base.yaml` / `sam2_hiera_base.pt`
- `sam2_hiera_large.yaml` / `sam2_hiera_large.pt`

## Step 3: Run the Application

### Terminal 1: Start Backend Server

```bash
cd backend
python main.py
```

The backend will be available at `http://localhost:8000`

**First run note:** The model will load on startup, which may take 1-2 minutes.

### Terminal 2: Start Frontend

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Step 4: Test the Application

1. Open `http://localhost:5173` in your browser
2. Click "Upload Prescription" or drag and drop an image
3. Wait for segmentation (may take 10-30 seconds depending on image size and GPU)
4. View the results:
   - **Red regions** = Handwritten text
   - **Green regions** = Printed text
   - Toggle between original and segmented view
   - See statistics in the text panel

## Troubleshooting

### Backend Issues

1. **Model not found error**
   - Ensure the checkpoint file is in `backend/checkpoints/`
   - Check the filename matches what's in `main.py`

2. **CUDA out of memory**
   - Use a smaller model (tiny or small)
   - Reduce image size before uploading
   - Close other GPU applications

3. **Import errors**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Try installing segment-anything-2 from source (see Step 2.1)

4. **Port already in use**
   - Change the port in `main.py`: `uvicorn.run(app, host="0.0.0.0", port=8001)`
   - Update frontend API URL in `LandingPage.jsx`

### Frontend Issues

1. **CORS errors**
   - Ensure backend is running
   - Check CORS settings in `backend/main.py`

2. **API connection failed**
   - Verify backend is running on `http://localhost:8000`
   - Check browser console for errors
   - Test backend health: `curl http://localhost:8000/health`

## API Endpoints

- `GET /` - API information
- `GET /health` - Check model status
- `POST /api/segment` - Upload image and get segmentation

## Performance Tips

- **GPU**: Use CUDA for 10-50x faster processing
- **Model size**: Larger models are more accurate but slower
- **Image size**: Smaller images process faster
- **Batch processing**: Currently processes one image at a time

## Next Steps

- Integrate actual OCR for text extraction
- Improve handwritten vs printed classification with ML model
- Add batch processing for multiple images
- Add export functionality for segmented regions

## Support

For issues with SAM2, visit: https://github.com/facebookresearch/segment-anything-2



