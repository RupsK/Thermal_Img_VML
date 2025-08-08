# Test Images Directory

This directory contains sample thermal images for testing the Thermal Image AI Analyzer application.

## Available Test Images

- `download.jpg` - Sample thermal image for testing
- `thermal_sample1.jpg` - Industrial thermal image
- `thermal_sample2.jpg` - Medical thermal image
- `thermal_sample3.jpg` - Security thermal image

## Adding Your Own Test Images

1. Place your thermal images in this directory
2. Supported formats: JPG, JPEG, PNG, BMP, TIFF, TIF
3. Update the `test_images` list in `streamlit_app.py` if you add new images

## Image Requirements

- **Format:** Thermal images (grayscale or color-mapped)
- **Size:** Recommended 512x512 to 2048x2048 pixels
- **File Size:** Under 200MB per image
- **Quality:** Clear thermal patterns and temperature variations

## Usage

These images are automatically loaded in the application's "Select from Test Images" dropdown menu for easy testing and demonstration purposes.
