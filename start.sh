#!/bin/bash

# Thermal Image AI Analyzer - Start Script

echo "🔥 Starting Thermal Image AI Analyzer..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "📦 Conda found, checking environment..."
    
    # Check if environment exists
    if conda env list | grep -q "thermal_img"; then
        echo "✅ Activating existing thermal_img environment..."
        conda activate thermal_img
    else
        echo "🔄 Creating new thermal_img environment..."
        conda create -n thermal_img python=3.9 -y
        conda activate thermal_img
    fi
else
    echo "⚠️  Conda not found, using system Python..."
fi

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Failed to install dependencies. Please check your Python environment."
    exit 1
fi

# Start the Streamlit application
echo "🚀 Starting Streamlit application..."
echo "🌐 Application will be available at: http://localhost:8501"
echo "📱 Press Ctrl+C to stop the application"

streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
