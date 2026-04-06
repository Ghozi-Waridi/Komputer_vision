#!/bin/bash

# 🧠 Brain MRI Tumor Classification - Streamlit Quick Start
# Script untuk install dependencies dan launch Streamlit app

echo "==============================================="
echo "🧠 Brain MRI Tumor Classification"
echo "📊 Streamlit Dashboard - Quick Start"
echo "==============================================="
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "✓ Python version: $python_version"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements_streamlit.txt -q

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Error installing dependencies"
    exit 1
fi
echo ""

# Check if models exist
echo "🔍 Checking for saved models..."
if [ -d "models" ] && [ "$(ls -A models/ | grep '_nn_' | wc -l)" -gt 0 ]; then
    echo "✓ Found $(ls models/ | grep '_nn_' | wc -l) model(s)"
else
    echo "⚠️  Warning: No models found in models/ folder"
    echo "   Please run 'main copy.ipynb' first to generate models"
fi
echo ""

# Launch Streamlit
echo "🚀 Launching Streamlit app..."
echo "   Opening: http://localhost:8501"
echo ""
echo "💡 Tips:"
echo "   - Press Ctrl+C to stop the app"
echo "   - Press 'R' in terminal to rerun"
echo "   - Press 'C' to clear cache"
echo ""

streamlit run streamlit_app.py
