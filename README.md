# AI Recipe Extractor

Extract recipes from images, web pages, and PDFs using AI. This tool uses OpenAI's vision and language models to intelligently parse recipe information from various sources and save them in an easy-to-copy text format.

## Features

- 📸 Extract recipes from images (single or multiple images per recipe)
- 📋 Extract recipes from clipboard images (just copy and run!)
- 🌐 Scrape recipes from web pages with intelligent parsing
- 📄 Extract recipes from PDF files
- 🤖 AI-powered extraction using OpenAI's GPT models
- 📝 Clean, formatted text output for easy copying
- 🔄 Batch processing support
- 📋 Structured data validation with Pydantic

## Prerequisites

Install [uv](https://github.com/astral-sh/uv) - a fast Python package manager:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

## Quick Start

If you're starting from scratch:
```bash
# Create a new directory and initialize a uv project
mkdir ai-recipes
cd ai-recipes
uv init

# Add the project dependencies
uv add openai pydantic pillow requests beautifulsoup4 pypdf2 click python-dotenv

# Optional: Add platform-specific clipboard support
# On macOS:
uv add --optional macos

# Copy the source code from this repository
# ... then run:
uv sync
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-recipes.git
cd ai-recipes
```

2. Run the initialization script:
```bash
./init.sh
```

Or manually initialize:
```bash
uv sync
```

This will automatically:
- Create a virtual environment in `.venv`
- Install all dependencies from `pyproject.toml`
- Create a `uv.lock` file for reproducible installs

3. Set up your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

Use `uv run` to execute commands in the project environment:

### Extract from Images

Single image:
```bash
uv run ai-recipes image cookbook_page.jpg
```

Multiple images (treated as one recipe):
```bash
uv run ai-recipes image page1.jpg page2.jpg --batch
```

### Extract from Clipboard

Copy an image to your clipboard and run:
```bash
uv run ai-recipes clipboard
```

**Platform-specific setup:**
- **macOS**: Install `pngpaste` for better clipboard support: `brew install pngpaste`
- **Linux**: Install `xclip` (X11) or `wl-paste` (Wayland)
- **Windows**: Works out of the box

### Extract from Web Pages

```bash
uv run ai-recipes web https://example.com/recipe/chocolate-cake
```

### Extract from PDFs

Extract from all pages:
```bash
uv run ai-recipes pdf cookbook.pdf
```

Extract from specific pages:
```bash
uv run ai-recipes pdf cookbook.pdf --pages "1,3,5-7"
```

### Batch Processing

Create a file with one input per line (images, URLs, or PDFs):
```
recipes.txt:
https://example.com/recipe1
/path/to/image1.jpg
/path/to/cookbook.pdf
https://example.com/recipe2
```

Process all inputs:
```bash
uv run ai-recipes batch recipes.txt
```

### Options

- `--output-dir, -o`: Specify output directory (default: `recipes/`)
- `--batch, -b`: Save multiple recipes in a single file
- `--source, -s`: Add source information to recipes
- `--model`: Specify OpenAI model (default: `gpt-4o`)

## Output Format

Recipes are saved as formatted text files with the following structure:

```
RECIPE: Chocolate Chip Cookies
================================

DESCRIPTION:
Classic homemade chocolate chip cookies that are crispy on the outside and chewy on the inside.

INGREDIENTS:
  • 2 1/4 cups all-purpose flour
  • 1 tsp baking soda
  • 1 tsp salt
  • 1 cup butter, softened
  • 3/4 cup granulated sugar
  • 3/4 cup packed brown sugar
  • 2 large eggs
  • 2 tsp vanilla extract
  • 2 cups chocolate chips

DIRECTIONS:
  1. Preheat oven to 375°F (190°C).
  2. In a small bowl, combine flour, baking soda and salt.
  3. In a large bowl, beat butter and sugars until creamy.
  4. Add eggs and vanilla; beat well.
  5. Gradually beat in flour mixture.
  6. Stir in chocolate chips.
  7. Drop by rounded tablespoons onto ungreased cookie sheets.
  8. Bake 9-11 minutes or until golden brown.

NOTES:
For chewier cookies, slightly underbake them. Store in an airtight container.

SOURCE: Grandma's Recipe Book
```

## Development

### Development Setup

Install development dependencies:
```bash
uv add --dev pytest pytest-cov black ruff mypy types-requests
```

### Project Structure

```
ai-recipes/
├── src/
│   ├── models.py         # Pydantic recipe model
│   ├── extractors/       # Content extraction modules
│   │   ├── image.py      # Image processing
│   │   ├── web.py        # Web scraping
│   │   └── pdf.py        # PDF processing
│   ├── llm_client.py     # OpenAI API integration
│   ├── formatter.py      # Output formatting
│   └── cli.py           # Command-line interface
├── pyproject.toml       # Project configuration and dependencies
└── README.md
```

### Running Tests and Tools

```bash
# Run tests
uv run pytest

# Format code
uv run black src/

# Lint code
uv run ruff src/

# Type checking
uv run mypy src/
```

### Adding Dependencies

```bash
# Add production dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Remove dependency
uv remove package-name
```

## Requirements

- Python 3.8+ (uv will automatically download the correct Python version)
- OpenAI API key
- Internet connection for web scraping and API calls

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.