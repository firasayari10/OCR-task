# TrOCR Backend Setup

This backend uses **TrOCR (Transformer-based OCR)** from Microsoft for handwritten text recognition, which is perfect for prescription OCR.

## Advantages of TrOCR

- ✅ **No model download needed** - Downloads automatically from Hugging Face
- ✅ **Specifically designed for handwritten text**
- ✅ **Works out of the box** - No manual setup required
- ✅ **Fast and accurate** for prescription text

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

This will install:
- FastAPI (web framework)
- Transformers (Hugging Face library)
- PyTorch (already installed)
- Other dependencies

### 2. Run the Server

```bash
python main.py
```

**First run:** The TrOCR model will automatically download from Hugging Face (~500MB). This happens automatically on first use.

You'll see:
```
Loading TrOCR model on cpu...
TrOCR model loaded successfully!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Test the API

The API will be available at `http://localhost:8000`

**Health check:**
```bash
curl http://localhost:8000/health
```

**Test OCR:**
```bash
curl -X POST "http://localhost:8000/api/segment" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/prescription.jpg"
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Check if model is loaded
- `POST /api/segment` - Upload image and get OCR results

## Response Format

```json
{
  "success": true,
  "extracted_text": "The extracted text from the image...",
  "image_preview": "data:image/png;base64,...",
  "text_length": 123,
  "model": "microsoft/trocr-base-handwritten"
}
```

## Model Information

- **Model:** `microsoft/trocr-base-handwritten`
- **Purpose:** Handwritten text recognition
- **Size:** ~500MB (downloads automatically)
- **Device:** Uses GPU if available, otherwise CPU

## Troubleshooting

1. **Model download fails:**
   - Check internet connection
   - The model downloads to: `~/.cache/huggingface/transformers/`
   - You can pre-download: `python -c "from transformers import AutoModelForVision2Seq; AutoModelForVision2Seq.from_pretrained('microsoft/trocr-base-handwritten')"`

2. **Out of memory:**
   - The model works on CPU but is slower
   - For faster processing, use GPU (CUDA)

3. **Import errors:**
   - Make sure transformers is installed: `pip install transformers`
   - Update transformers: `pip install --upgrade transformers`

## Next Steps

1. Start the backend: `python main.py`
2. Start the frontend: `npm run dev` (in project root)
3. Open `http://localhost:5173`
4. Upload a prescription image!



