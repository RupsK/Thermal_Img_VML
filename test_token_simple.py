#!/usr/bin/env python3
"""
Simple token test script for debugging
"""

import os
import sys

def test_token_simple():
    """Simple token test"""
    print("🔑 Simple Token Test")
    print("=" * 30)
    
    # Try to import config
    try:
        from config import Config
        token = Config.HF_TOKEN
        print(f"✅ Token loaded from config: {token[:10]}..." if token else "❌ No token found")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    # Check if token exists
    if not token:
        print("❌ No token found in configuration")
        return False
    
    # Check token format
    if not token.startswith('hf_'):
        print("❌ Token format invalid - should start with 'hf_'")
        return False
    
    print(f"✅ Token format looks correct: {token[:10]}...")
    
    # Try to test the token
    try:
        from huggingface_hub import login, whoami
        
        print("🔄 Testing token with login()...")
        login(token=token)
        print("✅ login() successful")
        
        print("🔄 Testing token with whoami()...")
        user_info = whoami(token=token)
        print(f"✅ whoami() successful - User: {user_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Token test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_token_simple()
    if success:
        print("\n🎉 Token is working correctly!")
    else:
        print("\n❌ Token has issues. Please check your configuration.")
        sys.exit(1)
