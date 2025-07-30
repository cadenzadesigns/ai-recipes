#!/bin/bash
# Quick initialization script for the AI Recipe Extractor

echo "🚀 Initializing AI Recipe Extractor..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Sync dependencies
echo "📦 Installing dependencies..."
uv sync

# Check for .env file
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your OpenAI API key"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "✨ Setup complete! To get started:"
echo "   1. Edit .env and add your OpenAI API key"
echo "   2. Run: uv run ai-recipes --help"
echo ""