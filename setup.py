#!/usr/bin/env python3
"""
Setup script for Thermal Image AI Analyzer
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file from template"""
    template_path = Path("env_template.txt")
    env_path = Path(".env")
    
    if env_path.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return False
    
    if not template_path.exists():
        print("❌ env_template.txt not found!")
        return False
    
    # Copy template to .env
    with open(template_path, 'r') as f:
        content = f.read()
    
    with open(env_path, 'w') as f:
        f.write(content)
    
    print("✅ Created .env file from template")
    return True

def get_huggingface_token():
    """Get Hugging Face token from user"""
    print("\n🔑 Hugging Face Token Setup")
    print("=" * 50)
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'Read' permissions")
    print("3. Copy the token (it starts with 'hf_')")
    print()
    
    token = input("Enter your Hugging Face token: ").strip()
    
    if not token.startswith('hf_'):
        print("❌ Invalid token format! Token should start with 'hf_'")
        return None
    
    return token

def update_env_file(token):
    """Update .env file with the provided token"""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("❌ .env file not found! Run setup again.")
        return False
    
    # Read current content
    with open(env_path, 'r') as f:
        content = f.read()
    
    # Replace placeholder with actual token
    content = content.replace('your_huggingface_token_here', token)
    
    # Write back
    with open(env_path, 'w') as f:
        f.write(content)
    
    print("✅ Updated .env file with your token")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'streamlit', 'torch', 'transformers', 'PIL', 
        'numpy', 'cv2', 'matplotlib', 'seaborn', 
        'pandas', 'huggingface_hub', 'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def main():
    """Main setup function"""
    print("🚀 Thermal Image AI Analyzer Setup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        return
    
    # Create .env file
    if not create_env_file():
        return
    
    # Get token from user
    token = get_huggingface_token()
    if not token:
        return
    
    # Update .env file
    if not update_env_file(token):
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the app: streamlit run streamlit_app.py")
    print("2. Or use: start_app.bat (Windows)")
    print("\nFor deployment options, see: DEPLOYMENT.md")

if __name__ == "__main__":
    main()
