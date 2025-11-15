#!/bin/bash
# SIE-X Installation Script

echo "🚀 Installing SIE-X..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_lg
python -m spacy download xx_ent_wiki_sm

# Setup Node.js SDK
cd sdk/nodejs
npm install
npm run build
cd ../..

# Setup Go SDK
cd sdk/go
go mod download
cd ../..

# Create necessary directories
mkdir -p models
mkdir -p data
mkdir -p logs
mkdir -p outputs

echo "✅ Installation complete!"
echo "📖 Run 'uvicorn sie_x.api.server:app --reload' to start the API"
