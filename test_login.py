#!/usr/bin/env python3
"""
Test script to check Hugging Face login functionality
"""

from config import Config
from huggingface_hub import login, whoami

def test_login():
    """Test both login and whoami functions"""
    print("🔑 Testing Hugging Face Token")
    print("=" * 40)
    
    token = Config.HF_TOKEN
    print(f"Token: {token[:10]}...")
    
    # Test whoami first
    try:
        user_info = whoami(token=token)
        print("✅ whoami() successful")
        print(f"User: {user_info}")
    except Exception as e:
        print(f"❌ whoami() failed: {e}")
    
    # Test login
    try:
        login(token=token)
        print("✅ login() successful")
    except Exception as e:
        print(f"❌ login() failed: {e}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")

if __name__ == "__main__":
    test_login()
