#!/usr/bin/env python3
"""
Token Setup Script for Thermal Image AI Analyzer

This script helps you set up your Hugging Face token for the application.
"""

import os
import sys

def setup_tokens():
    """Interactive token setup"""
    print("🔧 Thermal Image AI Analyzer - Token Setup")
    print("=" * 50)
    
    # Check if secrets.py already exists
    if os.path.exists('secrets.py'):
        print("⚠️  secrets.py already exists!")
        response = input("Do you want to overwrite it? (y/N): ").lower().strip()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    print("\n📋 To get your Hugging Face token:")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Click 'New token'")
    print("3. Give it a name (e.g., 'Thermal Analyzer')")
    print("4. Select 'Read' role")
    print("5. Copy the generated token")
    print()
    
    # Get token from user
    token = input("Enter your Hugging Face token: ").strip()
    
    if not token:
        print("❌ No token provided. Setup cancelled.")
        return
    
    if token == "your_actual_huggingface_token_here":
        print("❌ Please enter your actual token, not the placeholder.")
        return
    
    # Create secrets.py content
    secrets_content = f'''"""
Secrets configuration for Thermal Image AI Analyzer

This file contains sensitive configuration values.
DO NOT commit this file to version control.
"""

# Hugging Face Configuration
# Get your token from: https://huggingface.co/settings/tokens
HUGGINGFACE_TOKEN = "{token}"

# Alternative token sources (uncomment and use if needed)
# OPENAI_API_KEY = "your_actual_openai_api_key_here"
# ANTHROPIC_API_KEY = "your_actual_anthropic_api_key_here"

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
'''
    
    # Write secrets.py
    try:
        with open('secrets.py', 'w') as f:
            f.write(secrets_content)
        print("✅ secrets.py created successfully!")
        print("🔒 This file is automatically ignored by git for security.")
        
        # Test the configuration
        print("\n🧪 Testing configuration...")
        try:
            from config import Config
            Config.validate_config()
            print("✅ Configuration is valid!")
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            
    except Exception as e:
        print(f"❌ Error creating secrets.py: {e}")
        return
    
    print("\n🎉 Setup complete!")
    print("You can now run the application with: streamlit run streamlit_app.py")

if __name__ == "__main__":
    setup_tokens()
