#!/bin/bash
# Setup script for NLP Text Classification Dashboard

echo "🎯 NLP Text Classification Dashboard - Setup"
echo "=============================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Compare version (requires 3.11+)
required_version="3.11"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Error: Python 3.11 or higher is required"
    echo "Current version: $python_version"
    exit 1
fi
echo "✅ Python version is compatible"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "✅ Virtual environment recreated"
    fi
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet
echo "✅ pip upgraded"
echo ""

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Download NLTK data
echo "📥 Downloading NLTK data..."
python3 << END
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Downloading punkt...")
nltk.download('punkt', quiet=True)
print("Downloading brown...")
nltk.download('brown', quiet=True)
print("✅ NLTK data downloaded")
END
echo ""

# Create output directory
echo "📁 Creating output directory..."
mkdir -p nlp_results
echo "✅ Output directory created"
echo ""

# Setup complete
echo "=============================================="
echo "✅ Setup complete!"
echo ""
echo "🚀 To start the application:"
echo "   1. Activate virtual environment: source venv/bin/activate"
echo "   2. Run the app: streamlit run streamlit_nlp_app.py"
echo ""
echo "📖 For more information, see README.md"
echo "=============================================="
