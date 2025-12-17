"""
Host your custom models on HuggingFace Hub for production.

Steps:
1. Create HuggingFace account
2. Upload your models
3. Use them in production

This is FREE and recommended for transformer models!
"""

# ============================================
# Upload SAM2 to HuggingFace Hub
# ============================================

from huggingface_hub import HfApi, create_repo
import os

def upload_sam2_to_hub(model_path: str, repo_name: str):
    """
    Upload SAM2 checkpoint to HuggingFace Hub.
    
    Args:
        model_path: Path to sam2_hiera_large.pt
        repo_name: Your HF repo (e.g., "your-username/sam2-ocr")
    """
    # Login first: huggingface-cli login
    api = HfApi()
    
    # Create repo
    try:
        create_repo(repo_name, repo_type="model", private=False)
        print(f"Created repo: {repo_name}")
    except Exception as e:
        print(f"Repo might exist: {e}")
    
    # Upload model file
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="sam2_hiera_large.pt",
        repo_id=repo_name,
        repo_type="model"
    )
    print(f"✓ Uploaded to: https://huggingface.co/{repo_name}")


# ============================================
# Download from HuggingFace Hub in Production
# ============================================

from huggingface_hub import hf_hub_download

def download_sam2_from_hub(repo_name: str, cache_dir: str = "./models") -> str:
    """
    Download SAM2 from HuggingFace Hub.
    
    Args:
        repo_name: Your HF repo (e.g., "your-username/sam2-ocr")
        cache_dir: Where to cache the model
    
    Returns:
        Path to downloaded model
    """
    model_path = hf_hub_download(
        repo_id=repo_name,
        filename="sam2_hiera_large.pt",
        cache_dir=cache_dir
    )
    print(f"Downloaded SAM2 to: {model_path}")
    return model_path


# ============================================
# Example: Production Setup
# ============================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Upload: python huggingface_hosting.py upload")
        print("  Download: python huggingface_hosting.py download")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "upload":
        # Upload your SAM2 model
        upload_sam2_to_hub(
            model_path="checkpoints/sam2_hiera_large.pt",
            repo_name="firasayari10/sam2-medical-ocr"  # Change to your username
        )
    
    elif command == "download":
        # Download in production
        model_path = download_sam2_from_hub(
            repo_name="firasayari10/sam2-medical-ocr"
        )
        print(f"Model ready at: {model_path}")
