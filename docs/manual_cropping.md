# Manual Image Cropping Feature

## Overview

The AI Recipe Extractor now supports manual image cropping, allowing users to draw bounding boxes around recipe images they want to extract. This feature replaces the automatic ML-based detection which was unreliable.

## Usage

### Command Line Interface

The manual cropping feature is available in all CLI modes by adding the `--manual-crop` flag:

#### Single/Multiple Images
```bash
# Single image with manual cropping
uv run ai-recipes image recipe.jpg --manual-crop

# Multiple images as one recipe
uv run ai-recipes image page1.jpg page2.jpg page3.jpg --batch --manual-crop

# Multiple recipes with manifest
uv run ai-recipes image *.jpg --manifest recipes.json --manual-crop
```

#### PDF Files
```bash
# Extract from PDF with manual cropping
uv run ai-recipes pdf cookbook.pdf --manual-crop

# Specific pages only
uv run ai-recipes pdf cookbook.pdf --pages "1-5,7,9" --manual-crop
```

#### Web URLs
```bash
# Extract from web page with manual cropping
uv run ai-recipes web https://example.com/recipe --manual-crop
```

#### Batch Processing
```bash
# Process multiple inputs with manual cropping
uv run ai-recipes batch input_list.txt --manual-crop
```

## How It Works

1. **Recipe Extraction**: First, the text content is extracted and processed by the LLM to create the recipe.

2. **Image Selection**: After the recipe is saved, the manual cropping interface opens:
   - For each image, you'll be asked how many recipe images to extract (0 to skip)
   - An OpenCV window will display the image

3. **Drawing Bounding Boxes**:
   - Click and drag to draw rectangles around recipe images
   - Each rectangle will be labeled (Image 1, Image 2, etc.)
   - The first image is automatically saved as `main.jpg`

4. **Keyboard Controls**:
   - `c` - Clear all rectangles
   - `z` - Undo last rectangle
   - `SPACE` or `ENTER` - Accept and continue
   - `ESC` - Cancel (no images will be saved)

5. **Image Organization**:
   - Cropped images are saved in `recipes/[recipe_name]/images/`
   - Original source images are preserved in `images/originals/`
   - A `metadata.json` file tracks all extracted images

## Requirements

- Display environment (X11, Windows, or macOS GUI)
- For headless servers, use X11 forwarding or skip manual cropping

## Web Application

The Streamlit web application currently uses automatic image extraction. Manual cropping in web browsers would require a different approach (e.g., using JavaScript-based cropping libraries) and is not implemented in this version.

## Fallback Behavior

If manual cropping is requested but cannot be performed (e.g., no display available), the tool will:
1. Still extract the recipe text
2. Skip image extraction with a warning message
3. Save the recipe without images

## Examples

### Cookbook with Multiple Pages
```bash
# Extract a recipe from pages 5-8 of a cookbook PDF
uv run ai-recipes pdf cookbook.pdf --pages "5-8" --manual-crop

# You'll see:
# - Page 5: "How many recipe images to extract? (0 to skip): 1"
#   Draw a box around the main dish photo
# - Page 6: "How many recipe images to extract? (0 to skip): 0"
#   Skip the ingredients list page
# - Page 7: "How many recipe images to extract? (0 to skip): 2"
#   Draw boxes around two step-by-step photos
# - Page 8: "How many recipe images to extract? (0 to skip): 0"
#   Skip the text-only instructions
```

### Multi-Image Recipe
```bash
# Process a recipe spread across 3 photos
uv run ai-recipes image IMG_001.jpg IMG_002.jpg IMG_003.jpg --batch --manual-crop

# The tool will:
# 1. Combine all 3 images to extract the complete recipe
# 2. Then show each image for manual cropping
# 3. You can extract the hero shot from IMG_001 and step photos from IMG_003
```

## Tips

1. **Image Quality**: Draw tight bounding boxes around the actual food images, excluding borders and text
2. **Multiple Regions**: You can draw multiple rectangles on a single page to extract multiple images
3. **Skip Text Pages**: Enter "0" for pages that contain only text or no relevant images
4. **Main Photo**: The first cropped image is automatically designated as the main recipe photo