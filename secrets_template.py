"""
Secrets configuration template for Thermal Image AI Analyzer

Copy this file to secrets.py and fill in your actual values.
This file is safe to commit to GitHub as it only contains placeholder values.
"""

# Hugging Face Configuration
# Get your token from: https://huggingface.co/settings/tokens
HUGGINGFACE_TOKEN = "your_huggingface_token_here"

# Alternative token sources (uncomment and use if needed)
# OPENAI_API_KEY = "your_openai_api_key_here"
# ANTHROPIC_API_KEY = "your_anthropic_api_key_here"

# Hardware Configuration
USE_GPU = True
LOW_MEMORY_MODE = False

# App Configuration
STREAMLIT_SERVER_PORT = 8501
STREAMLIT_SERVER_ADDRESS = "0.0.0.0"

# Model Configuration
DEFAULT_MODEL = "BLIP Base"
ENSEMBLE_ENABLED = False

# Processing Configuration
MAX_FILE_SIZE = 209715200
BATCH_SIZE = 1
MAX_WORKERS = 4

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "thermal_analyzer.log"

# Security Configuration
ENABLE_TOKEN_OVERRIDE = False
REQUIRE_AUTHENTICATION = False
