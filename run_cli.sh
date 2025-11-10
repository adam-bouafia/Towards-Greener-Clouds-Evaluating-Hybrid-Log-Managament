#!/bin/bash
# Wrapper script to run CLI with proper Python environment

# Change to script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "✅ Using virtual environment (.venv)"
    .venv/bin/python cli.py
else
    echo "⚠️  No virtual environment found."
    echo ""
    echo "Would you like to create one? [y/N]"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🔨 Creating virtual environment..."
        python3 -m venv .venv
        
        echo "📦 Installing dependencies..."
        .venv/bin/pip install --upgrade pip
        .venv/bin/pip install -r requirements.txt
        
        echo ""
        echo "✅ Setup complete! Starting CLI..."
        echo ""
        .venv/bin/python cli.py
    else
        echo "❌ Cancelled. Create venv manually with: python3 -m venv .venv"
        exit 1
    fi
fi
