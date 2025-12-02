#!/bin/bash

# Setup script for Streamlit Cloud
# Downloads spaCy model and TextBlob corpora before app starts

echo "Starting setup script..."

# Download spaCy English model
echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Download TextBlob corpora
echo "Downloading TextBlob corpora..."
python -m textblob.download_corpora

echo "Setup complete!"
