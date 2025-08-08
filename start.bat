@echo off
chcp 65001 >nul

echo 🔥 Starting Thermal Image AI Analyzer...

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% equ 0 (
    echo 📦 Conda found, checking environment...
    
    REM Check if environment exists
    conda env list | findstr "thermal_img" >nul
    if %errorlevel% equ 0 (
        echo ✅ Activating existing thermal_img environment...
        call conda activate thermal_img
    ) else (
        echo 🔄 Creating new thermal_img environment...
        call conda create -n thermal_img python=3.9 -y
        call conda activate thermal_img
    )
) else (
    echo ⚠️  Conda not found, using system Python...
)

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Check if installation was successful
if %errorlevel% equ 0 (
    echo ✅ Dependencies installed successfully!
) else (
    echo ❌ Failed to install dependencies. Please check your Python environment.
    pause
    exit /b 1
)

REM Start the Streamlit application
echo 🚀 Starting Streamlit application...
echo 🌐 Application will be available at: http://localhost:8501
echo 📱 Press Ctrl+C to stop the application

streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

pause
