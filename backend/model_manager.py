"""
Production Model Manager
Downloads and manages models from cloud storage for production deployment.
"""

import os
import logging
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model downloads and caching for production."""
    
    def __init__(self, cache_dir: str = "./model_cache"):
        """
        Initialize model manager.
        
        Args:
            cache_dir: Local directory to cache downloaded models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry - map model names to download URLs
        self.model_registry = {
            "sam2_hiera_large": {
                "url": "https://your-storage.blob.core.windows.net/models/sam2_hiera_large.pt",
                "filename": "sam2_hiera_large.pt",
                "size_mb": 856,
                "required": True
            },
            "all-MiniLM-L6-v2": {
                "hf_model": "sentence-transformers/all-MiniLM-L6-v2",
                "size_mb": 87,
                "required": True
            },
            "trocr-large-handwritten": {
                "hf_model": "microsoft/trocr-large-handwritten",
                "size_mb": 4253,
                "required": False  # Optional
            }
        }
    
    def download_file(self, url: str, destination: Path, desc: str = "Downloading"):
        """Download file with progress bar."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        logger.info(f"Downloaded {destination}")
    
    def ensure_model(self, model_name: str, force_download: bool = False) -> Path:
        """
        Ensure model is available, download if needed.
        
        Args:
            model_name: Name of model from registry
            force_download: Force re-download even if cached
        
        Returns:
            Path to model file/directory
        """
        if model_name not in self.model_registry:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_info = self.model_registry[model_name]
        
        # Handle HuggingFace models
        if "hf_model" in model_info:
            logger.info(f"HuggingFace model {model_name} will be auto-downloaded by transformers")
            return None  # Let transformers handle it
        
        # Handle custom models (like SAM2)
        model_path = self.cache_dir / model_info["filename"]
        
        if model_path.exists() and not force_download:
            logger.info(f"Model {model_name} already cached at {model_path}")
            return model_path
        
        logger.info(f"Downloading {model_name} ({model_info['size_mb']} MB)...")
        self.download_file(
            model_info["url"],
            model_path,
            desc=f"Downloading {model_name}"
        )
        
        return model_path
    
    def ensure_all_required_models(self):
        """Download all required models for production."""
        logger.info("Ensuring all required models are available...")
        
        for model_name, model_info in self.model_registry.items():
            if model_info.get("required", True):
                try:
                    self.ensure_model(model_name)
                except Exception as e:
                    logger.error(f"Failed to download {model_name}: {e}")
                    raise
        
        logger.info("All required models are ready!")
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get path to cached model."""
        if model_name not in self.model_registry:
            return None
        
        model_info = self.model_registry[model_name]
        
        if "hf_model" in model_info:
            return None  # HF models don't have local paths
        
        return self.cache_dir / model_info["filename"]


# Singleton instance
_model_manager = None

def get_model_manager(cache_dir: str = "./model_cache") -> ModelManager:
    """Get or create ModelManager singleton."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(cache_dir)
    return _model_manager


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    manager = get_model_manager()
    
    # Download all required models
    manager.ensure_all_required_models()
    
    # Get path to SAM2 model
    sam2_path = manager.get_model_path("sam2_hiera_large")
    print(f"SAM2 model at: {sam2_path}")
