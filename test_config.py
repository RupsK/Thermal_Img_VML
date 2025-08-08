#!/usr/bin/env python3
"""
Test script to verify the configuration system works correctly.
"""

import os
import sys

def test_configuration():
    """Test the configuration system"""
    print("🧪 Testing Configuration System")
    print("=" * 40)
    
    # Test 1: Check if secrets.py exists
    print("1. Checking secrets.py file...")
    if os.path.exists('secrets.py'):
        print("   ✅ secrets.py found")
    else:
        print("   ⚠️  secrets.py not found (will use environment variables)")
    
    # Test 2: Check if .env exists
    print("2. Checking .env file...")
    if os.path.exists('.env'):
        print("   ✅ .env file found")
    else:
        print("   ⚠️  .env file not found")
    
    # Test 3: Test config import
    print("3. Testing config import...")
    try:
        from config import Config
        print("   ✅ Config imported successfully")
    except Exception as e:
        print(f"   ❌ Config import failed: {e}")
        return False
    
    # Test 4: Test token validation
    print("4. Testing token validation...")
    try:
        Config.validate_config()
        print("   ✅ Token validation passed")
    except ValueError as e:
        print(f"   ❌ Token validation failed: {e}")
        print("   💡 Run 'python setup_tokens.py' to configure your token")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False
    
    # Test 5: Test model configuration
    print("5. Testing model configuration...")
    try:
        models = Config.MODELS
        print(f"   ✅ Found {len(models)} models")
        for name, model_id in models.items():
            print(f"      - {name}: {model_id}")
    except Exception as e:
        print(f"   ❌ Model configuration error: {e}")
        return False
    
    # Test 6: Test device configuration
    print("6. Testing device configuration...")
    try:
        device = Config.DEVICE
        print(f"   ✅ Device configured: {device}")
    except Exception as e:
        print(f"   ❌ Device configuration error: {e}")
        return False
    
    print("\n🎉 All tests passed! Configuration is working correctly.")
    return True

if __name__ == "__main__":
    success = test_configuration()
    if not success:
        print("\n❌ Configuration test failed. Please check the setup.")
        sys.exit(1)
    else:
        print("\n✅ Configuration is ready for use!")
