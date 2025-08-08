#!/usr/bin/env python3
"""
Test script to verify Hugging Face token
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_token():
    """Test if Hugging Face token is valid"""
    token = os.getenv('HUGGINGFACE_TOKEN', '')
    
    if not token:
        print("❌ HUGGINGFACE_TOKEN not found in environment variables")
        print("Please set your token in .env file or environment variables")
        return False
    
    if not token.startswith('hf_'):
        print("❌ Invalid token format. Token should start with 'hf_'")
        return False
    
    try:
        from huggingface_hub import login, whoami
        login(token=token)
        user_info = whoami(token=token)
        print(f"✅ Token is valid! Logged in as: {user_info}")
        return True
    except Exception as e:
        print(f"❌ Token validation failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔑 Testing Hugging Face Token...")
    test_token()
