"""
Download SAM2 model from Hugging Face (alternative to Facebook CDN)
"""
import os
from pathlib import Path

def download_from_huggingface():
    """Instructions for downloading from Hugging Face"""
    print("=" * 60)
    print("Download SAM2 Model from Hugging Face")
    print("=" * 60)
    print()
    print("Method 1: Using Hugging Face CLI (Recommended)")
    print("-" * 60)
    print("1. Install Hugging Face CLI:")
    print("   pip install huggingface_hub")
    print()
    print("2. Download the model:")
    print("   python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='facebook/sam2-hiera', filename='sam2_hiera_large.pt', local_dir='checkpoints')\"")
    print()
    print("Method 2: Manual Download from Hugging Face")
    print("-" * 60)
    print("1. Visit: https://huggingface.co/facebook/sam2-hiera/tree/main")
    print("2. Click on 'sam2_hiera_large.pt' (or the model you want)")
    print("3. Click the download button")
    print("4. Save the file to: backend/checkpoints/sam2_hiera_large.pt")
    print()
    print("Method 3: Direct Browser Download")
    print("-" * 60)
    print("Try these URLs in your browser:")
    print()
    print("Large model:")
    print("  https://huggingface.co/facebook/sam2-hiera/resolve/main/sam2_hiera_large.pt")
    print()
    print("Base model:")
    print("  https://huggingface.co/facebook/sam2-hiera/resolve/main/sam2_hiera_base.pt")
    print()
    print("Small model:")
    print("  https://huggingface.co/facebook/sam2-hiera/resolve/main/sam2_hiera_small.pt")
    print()
    print("Tiny model:")
    print("  https://huggingface.co/facebook/sam2-hiera/resolve/main/sam2_hiera_tiny.pt")
    print()
    print("=" * 60)

if __name__ == "__main__":
    try:
        from huggingface_hub import hf_hub_download
        
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        print("Downloading sam2_hiera_large.pt from Hugging Face...")
        print("This may take a while...")
        
        downloaded_path = hf_hub_download(
            repo_id="facebook/sam2-hiera",
            filename="sam2_hiera_large.pt",
            local_dir=str(checkpoint_dir),
            local_dir_use_symlinks=False
        )
        
        print(f"\n✓ Successfully downloaded to: {downloaded_path}")
        print("\nThe model is ready to use!")
        
    except ImportError:
        print("huggingface_hub not installed. Showing manual instructions...")
        print()
        download_from_huggingface()
        print()
        print("To install huggingface_hub and download automatically:")
        print("  pip install huggingface_hub")
        print("  python download_from_huggingface.py")
    except Exception as e:
        print(f"Error: {e}")
        print()
        download_from_huggingface()



