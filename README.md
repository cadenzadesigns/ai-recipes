# AI Recipe Extractor

Extract recipes from images, web pages, and PDFs using AI. This tool uses OpenAI's vision and language models to intelligently parse recipe information from various sources and save them in an easy-to-copy text format.

## Features

- 📸 Extract recipes from images (single or multiple images per recipe, including HEIC/HEIF)
- 📋 Extract recipes from clipboard images (just copy and run!)
- 🌐 Scrape recipes from web pages with intelligent parsing
- 📄 Extract recipes from PDF files
- 🤖 AI-powered extraction using OpenAI's GPT models
- 📝 Dual output: human-readable text files AND JSON for integrations
- 🔄 Batch processing support
- 🍳 Paprika Recipe Manager integration
- 🔀 Convert existing text recipes to JSON format
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

# Optional: Add HEIC support for iPhone photos
uv add pillow-heif

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

3. Set up your API keys:
```bash
cp .env.example .env
# Edit .env and add:
# - Your OpenAI API key (required)
# - Your Paprika credentials (optional, for upload feature)
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

### Convert Text Recipes to JSON

Convert all existing text recipes:
```bash
uv run ai-recipes convert
```

Convert a single text file:
```bash
uv run ai-recipes convert -s recipes/txt/chocolate_cake.txt
```

### Upload to Paprika (Experimental)

**⚠️ Known Issues with Paprika API Integration**

We've implemented Paprika upload functionality, but there are significant limitations:

1. **Authentication works** - We can authenticate with the v2 API using macOS client headers
2. **Uploads succeed** - The API accepts recipes and returns success
3. **Recipes don't sync** - Uploaded recipes don't appear in the recipe list or Paprika apps

This appears to be because:
- The Paprika API is unofficial and undocumented
- Uploaded recipes may go to a staging area that requires server-side processing
- The sync mechanism between API uploads and user apps is unclear

**Current Status**: The upload command exists but recipes won't appear in your Paprika apps.

```bash
# Setup (for future use when sync issues are resolved):
# Add to .env:
# PAPRIKA_EMAIL=your-email@example.com
# PAPRIKA_PASSWORD=your-paprika-password

# Upload attempt (recipes won't sync to apps):
uv run ai-recipes paprika recipes/json/chocolate_cake.json
```

**Recommended Workaround**: 
- Use the generated `.txt` files with Paprika's built-in import features
- Copy recipe text and paste into Paprika's manual entry
- Use Paprika's browser extension for web recipes
- Email recipes to your Paprika import email address

### Options

- `--output-dir, -o`: Specify output directory (default: `recipes/`)
- `--batch, -b`: Save multiple recipes in a single file
- `--source, -s`: Add source information to recipes
- `--model`: Specify OpenAI model (default: `gpt-4o`)

## Output Structure

Recipes are saved in two formats:

```
recipes/
├── txt/     # Human-readable text files
└── json/    # JSON files for Paprika and other integrations
```

### Text Format

The text files are formatted for easy reading and copying:

```
Chocolate Chip Cookies
================================

DESCRIPTION:
Classic homemade chocolate chip cookies that are crispy on the outside and chewy on the inside.

SERVINGS: 24 cookies
TOTAL TIME: 25 minutes

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
  • For chewier cookies, slightly underbake them
  • Store in an airtight container for up to 5 days
  • Can be frozen for up to 3 months

SOURCE: Grandma's Recipe Book
```

### JSON Format

The JSON files contain structured data perfect for importing into recipe management apps like Paprika.

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
│   │   ├── image.py      # Image processing (including HEIC)
│   │   ├── clipboard.py  # Clipboard image extraction
│   │   ├── web.py        # Web scraping
│   │   └── pdf.py        # PDF processing
│   ├── llm_client.py     # OpenAI API integration
│   ├── paprika_client.py # Paprika Recipe Manager API
│   ├── converter.py      # Text to JSON converter
│   ├── formatter.py      # Output formatting
│   └── cli.py           # Command-line interface
├── recipes/              # Output directory (gitignored)
│   ├── txt/             # Human-readable text recipes
│   └── json/            # JSON format for integrations
├── pyproject.toml       # Project configuration and dependencies
├── Makefile            # Development commands
└── README.md
```

### Running Tests and Tools

```bash
# Run tests
uv run pytest

# Format code
uv run black src/

# Lint and auto-fix code
make ruff

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

## Recipe Data Model

The tool extracts the following information:
- **Name**: Recipe title
- **Description**: Exact text from the source (preserved verbatim)
- **Servings**: Yield information (e.g., "4 servings", "12 cookies")
- **Total Time**: Total cook time (only if explicitly stated in source)
- **Ingredients**: List with quantities
- **Directions**: Step-by-step instructions
- **Notes**: Tips, variations, and storage instructions (as bullet points)
- **Source**: Where the recipe came from

## Requirements

- Python 3.8+ (uv will automatically download the correct Python version)
- OpenAI API key
- Internet connection for web scraping and API calls
- Optional: Paprika Recipe Manager account for upload feature

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.