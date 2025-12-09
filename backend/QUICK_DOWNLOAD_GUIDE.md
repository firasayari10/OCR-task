# Quick Model Download Guide

Since automated downloads are getting 403 errors, here's the **easiest way** to get the model:

## ✅ Easiest Method: Browser Download

1. **Open your web browser**

2. **Copy and paste this URL:**
   ```
   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.pt
   ```

3. **The file will start downloading** (~1GB, may take 5-15 minutes)

4. **Once downloaded:**
   - The file will be in your Downloads folder
   - Move it to: `C:\Users\Firas\Desktop\ocr\backend\checkpoints\`
   - Make sure it's named: `sam2_hiera_large.pt`

5. **Verify it's there:**
   ```bash
   cd backend
   dir checkpoints
   ```
   You should see `sam2_hiera_large.pt`

## 🚀 Alternative: Use Smaller Model for Testing

If you want to test quickly, download the **tiny** model (~40MB):

1. Visit: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_tiny.pt
2. Save to: `backend/checkpoints/sam2_hiera_tiny.pt`
3. Edit `backend/main.py` lines 43-44:
   ```python
   model_cfg = "sam2_hiera_tiny.yaml"
   checkpoint_path = "checkpoints/sam2_hiera_tiny.pt"
   ```

## 📋 After Download

Once the model file is in place:

1. **Start the backend:**
   ```bash
   cd backend
   python main.py
   ```
   You should see: "SAM2 model loaded successfully!"

2. **Start the frontend** (new terminal):
   ```bash
   npm run dev
   ```

3. **Open browser:** http://localhost:5173

## ⚠️ If Browser Download Also Fails

Try these alternative sources:

1. **GitHub Releases:** Check https://github.com/facebookresearch/segment-anything-2/releases
2. **Direct from Meta:** Visit the official SAM2 repository
3. **Use a VPN:** Sometimes CDN blocks are region-specific

## 🧪 Test Without Model

You can start the server without the model to test the frontend:

```bash
cd backend
python main.py
```

The server will start with a warning, and you can see the UI, but segmentation won't work until the model is downloaded.



