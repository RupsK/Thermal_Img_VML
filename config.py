"""
Configuration settings for Thermal Image AI Analyzer
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (for backward compatibility)
load_dotenv()

# Try to import app_secrets, fallback to environment variables
try:
    from app_secrets import HUGGINGFACE_TOKEN, USE_GPU, LOW_MEMORY_MODE
    print("✅ Using app_secrets.py configuration")
except ImportError as e:
    print(f"⚠️ app_secrets.py not found, using environment variables. Error: {e}")
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN', '')
    USE_GPU = os.getenv('USE_GPU', 'false').lower() == 'true'
    LOW_MEMORY_MODE = os.getenv('LOW_MEMORY_MODE', 'false').lower() == 'true'

class Config:
    """Configuration class for the Thermal Image AI Analyzer"""
    
    # Hugging Face Token - load from secrets or environment variable
    HF_TOKEN = HUGGINGFACE_TOKEN
    
    # Model configurations
    MODELS = {
        "BLIP Base": "Salesforce/blip-image-captioning-base",
        "BLIP Large": "Salesforce/blip-image-captioning-large", 
        "GIT Base": "microsoft/git-base-coco",
        "LLaVA-Next": "llava-hf/llava-1.5-7b-hf",
        "SmolVLM": "microsoft/DialoGPT-medium"
    }
    
    # App settings
    APP_TITLE = "Thermal Image AI Analyzer"
    APP_ICON = "🔥"
    LAYOUT = "wide"
    
    # Processing settings
    DEFAULT_IMAGE_SIZE = (224, 224)
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
    SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
    
    # Model settings
    DEFAULT_MODEL = "BLIP Base"
    # Device configuration - use secrets or environment variables
    DEVICE = "cpu" if os.getenv('FORCE_CPU', 'true').lower() == 'true' else ("cuda" if USE_GPU else "cpu")
    
    # Memory management
    LOW_MEMORY_MODE = LOW_MEMORY_MODE
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present"""
        if not cls.HF_TOKEN:
            raise ValueError(
                "HUGGINGFACE_TOKEN environment variable is required. "
                "Please set it in your .env file or environment variables."
            )
        return True
    
    @classmethod
    def get_model_config(cls, model_name):
        """Get configuration for a specific model"""
        if model_name not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        return cls.MODELS[model_name]
