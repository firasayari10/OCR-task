"""
Download SAM2 model using Python requests (works better than urllib on Windows)
"""
import requests
import os
from pathlib import Path

MODELS = {
    "tiny": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_tiny.pt",
        "name": "sam2_hiera_tiny.pt",
        "size": "~40MB"
    },
    "small": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_small.pt",
        "name": "sam2_hiera_small.pt",
        "size": "~100MB"
    },
    "base": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_base.pt",
        "name": "sam2_hiera_base.pt",
        "size": "~200MB"
    },
    "large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.pt",
        "name": "sam2_hiera_large.pt",
        "size": "~1GB"
    }
}

def download_model(model_size="large"):
    """Download SAM2 model checkpoint using requests"""
    
    if model_size not in MODELS:
        print(f"Error: Invalid model size. Choose from: {', '.join(MODELS.keys())}")
        return False
    
    model_info = MODELS[model_size]
    checkpoint_dir = Path("checkpoints")
    checkpoint_path = checkpoint_dir / model_info["name"]
    
    # Create checkpoints directory if it doesn't exist
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Check if model already exists
    if checkpoint_path.exists():
        print(f"Model {model_info['name']} already exists at {checkpoint_path}")
        response = input("Do you want to download it again? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return True
    
    print(f"Downloading {model_info['name']} ({model_info['size']})...")
    print(f"URL: {model_info['url']}")
    print("This may take a while depending on your internet connection...")
    print()
    
    try:
        # Use a session with headers to avoid 403 errors
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        response = session.get(model_info["url"], stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(checkpoint_path, 'wb') as f:
            downloaded = 0
            chunk_size = 8192
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}MB / {mb_total:.1f}MB)", end='', flush=True)
        
        print(f"\n✓ Successfully downloaded {model_info['name']} to {checkpoint_path}")
        print(f"\nNext steps:")
        print(f"1. The model is ready to use")
        print(f"2. Run: python main.py")
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"\n✗ HTTP Error: {e}")
        print(f"Status code: {response.status_code}")
        print("\nThe download URL might be temporarily unavailable.")
        print("Try these alternatives:")
        print(f"1. Download manually from: {model_info['url']}")
        print("2. Visit: https://github.com/facebookresearch/segment-anything-2")
        print("3. Check: https://huggingface.co/facebook/sam2-hiera")
        return False
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nAlternative download methods:")
        print(f"1. Visit: {model_info['url']} in your browser")
        print(f"2. Download manually and place in {checkpoint_path}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_size = sys.argv[1].lower()
    else:
        print("SAM2 Model Downloader (using requests)")
        print("=" * 50)
        print("Available models:")
        for size, info in MODELS.items():
            print(f"  {size:6} - {info['name']:25} ({info['size']})")
        print()
        model_size = input("Enter model size to download (tiny/small/base/large) [large]: ").strip().lower()
        if not model_size:
            model_size = "large"
    
    # Install requests if not available
    try:
        import requests
    except ImportError:
        print("Installing requests library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests
    
    download_model(model_size)



