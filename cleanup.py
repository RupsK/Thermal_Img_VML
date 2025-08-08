"""
Cleanup script for Thermal Image AI Analyzer
"""

import os

def cleanup():
    """Remove temporary files"""
    files_to_remove = [
        "create_test_images.py",
        "cleanup.py"
    ]
    
    print("🧹 Cleaning up temporary files...")
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"✅ Removed {file}")
        else:
            print(f"⚠️  {file} not found")
    
    print("🎉 Cleanup completed!")

if __name__ == "__main__":
    cleanup()
