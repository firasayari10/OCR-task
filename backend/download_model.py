"""
Script to download SAM2 model weights
Run this script to download the model checkpoint
"""

import urllib.request
import os
import sys

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
    """Download SAM2 model checkpoint"""
    
    if model_size not in MODELS:
        print(f"Error: Invalid model size. Choose from: {', '.join(MODELS.keys())}")
        return False
    
    model_info = MODELS[model_size]
    checkpoint_dir = "checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, model_info["name"])
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(checkpoint_path):
        print(f"Model {model_info['name']} already exists at {checkpoint_path}")
        response = input("Do you want to download it again? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return True
    
    print(f"Downloading {model_info['name']} ({model_info['size']})...")
    print(f"URL: {model_info['url']}")
    print("This may take a while depending on your internet connection...")
    
    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rProgress: {percent}%", end='', flush=True)
        
        urllib.request.urlretrieve(
            model_info["url"],
            checkpoint_path,
            reporthook=progress_hook
        )
        
        print(f"\n✓ Successfully downloaded {model_info['name']} to {checkpoint_path}")
        print(f"\nNext steps:")
        print(f"1. Update main.py to use '{model_size}' model if needed")
        print(f"2. Run: python main.py")
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nAlternative download methods:")
        print(f"1. Visit: {model_info['url']}")
        print(f"2. Download manually and place in {checkpoint_path}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_size = sys.argv[1].lower()
    else:
        print("SAM2 Model Downloader")
        print("=" * 50)
        print("Available models:")
        for size, info in MODELS.items():
            print(f"  {size:6} - {info['name']:25} ({info['size']})")
        print()
        model_size = input("Enter model size to download (tiny/small/base/large) [large]: ").strip().lower()
        if not model_size:
            model_size = "large"
    
    download_model(model_size)



