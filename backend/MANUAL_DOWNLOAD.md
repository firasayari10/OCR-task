# Manual Model Download Guide

The automatic download is getting a 403 error. Here are alternative methods to download the SAM2 model:

## Method 1: Download via Browser (Easiest)

1. Open your web browser
2. Visit one of these URLs:

   **Large model (recommended, ~1GB):**
   ```
   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.pt
   ```

   **Base model (~200MB):**
   ```
   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_base.pt
   ```

   **Small model (~100MB):**
   ```
   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_small.pt
   ```

   **Tiny model (~40MB):**
   ```
   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_tiny.pt
   ```

3. The file will download automatically
4. Move the downloaded `.pt` file to: `backend/checkpoints/`
5. Rename it to match what's in `main.py` (e.g., `sam2_hiera_large.pt`)

## Method 2: Use wget (if installed)

```bash
cd backend/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.pt
```

## Method 3: Use Python requests

Create and run this script:

```python
import requests
import os

os.makedirs("checkpoints", exist_ok=True)

url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.pt"
output_path = "checkpoints/sam2_hiera_large.pt"

print("Downloading model...")
response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(output_path, 'wb') as f:
    downloaded = 0
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.1f}%", end='', flush=True)

print(f"\n✓ Downloaded to {output_path}")
```

## Method 4: Alternative Source (Hugging Face)

You can also try downloading from Hugging Face:

1. Visit: https://huggingface.co/facebook/sam2-hiera/tree/main
2. Download the checkpoint file you need
3. Place it in `backend/checkpoints/`

## After Download

Once you have the model file in `backend/checkpoints/`, verify it exists:

```bash
cd backend
dir checkpoints
```

Then you can start the server:

```bash
python main.py
```



