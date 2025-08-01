# Manual Image Cropping in Streamlit Web App

## Overview

The Streamlit web application now supports manual image cropping across all extraction modes, providing a user-friendly web interface for selecting recipe images without requiring OpenCV or command-line interaction.

## Features

### 1. Single Image Mode
- Upload a single image containing a recipe
- After recipe extraction, manually define crop regions for recipe images
- Use coordinate inputs to precisely select image areas
- Visual preview shows selected regions in green

### 2. Batch Processing Mode
- **Individual Recipes**: Each image processed separately with its own cropping session
- **Combined Recipe**: All images processed as one recipe, then crop from all pages
- Pending crops are queued and processed after recipe extraction
- Progress tracking shows which recipe is being cropped

### 3. PDF Extraction
- PDF pages are automatically converted to images
- Each page can be cropped individually
- Select which pages contain recipe images (skip text-only pages)
- Supports partial page selection

### 4. Web URL Extraction
- Images are automatically downloaded from the web page
- Filter out small icons and thumbnails
- Manually select which images to keep for the recipe
- Up to 10 images downloaded per page

## How to Use

### Basic Workflow

1. **Upload/Enter Source**
   - Choose your input method (image, PDF, URL)
   - Enable "✂️ Manual Crop" checkbox (enabled by default)

2. **Extract Recipe**
   - Click "🚀 Extract Recipe" to process the text
   - Recipe is saved immediately

3. **Crop Images**
   - For each image/page, select number of regions to extract (0 to skip)
   - Define crop regions using coordinate inputs
   - Click "Add Region" to add each crop area
   - Visual preview shows all defined regions

4. **Navigation**
   - Use "◀ Previous" and "Next ▶" buttons to navigate between images
   - "Save & Continue" to move to next image
   - "Finish Cropping" on the last image to save all crops

### Coordinate System

- **X1, Y1**: Top-left corner of the crop region
- **X2, Y2**: Bottom-right corner of the crop region
- Coordinates are in pixels relative to the displayed image
- The interface shows current image dimensions for reference

### Batch Processing with Manual Crop

When processing multiple images:

1. All recipes are extracted first
2. After extraction completes, manual cropping UI appears
3. Process each recipe's images in sequence
4. Progress is maintained across all pending crops

### Tips

1. **Quick Selection**: For full images, keep default coordinates (full image bounds)
2. **Multiple Regions**: Add multiple crop regions per page for step-by-step photos
3. **Skip Pages**: Set "0" regions for text-only or irrelevant pages
4. **Main Image**: The first cropped image is automatically set as the main recipe photo

## Technical Details

### Session State Management
- Crop regions are stored in Streamlit session state
- Progress is maintained even if you navigate between tabs
- Pending crops are queued for processing after recipe extraction

### Image Organization
```
recipes/[recipe_name]/
├── [recipe_name].txt
├── [recipe_name].json
└── images/
    ├── main.jpg          # First cropped image
    ├── image_1.jpg       # Additional images
    ├── image_2.jpg
    └── originals/        # Original source images
        ├── page_001.png  # For PDFs
        └── web_image_001.jpg  # For URLs
```

### Limitations

1. **No Drawing Interface**: Unlike the CLI version, the web app uses coordinate inputs instead of mouse drawing
2. **Session-Based**: Crop regions are lost if the session expires
3. **Sequential Processing**: Images must be cropped in order
4. **No Undo**: Use "Remove" buttons to delete individual regions

## Example Scenarios

### Multi-Page Cookbook PDF
1. Upload PDF and select page range (e.g., pages 5-8)
2. Extract recipe (e.g., "Chocolate Cake")
3. Page 5: Define 1 region for hero photo
4. Page 6: Skip (0 regions) - ingredients list
5. Page 7: Define 2 regions for process photos
6. Page 8: Skip (0 regions) - text instructions

### Web Recipe with Multiple Images
1. Enter recipe URL
2. Extract recipe text
3. Review downloaded images
4. Skip small thumbnails and ads
5. Select main recipe photo and any process shots
6. Finish cropping to save

### Batch Processing Multiple Recipes
1. Upload 10 images (5 recipes, 2 pages each)
2. Choose "Individual Recipes" mode
3. System extracts all 5 recipes
4. Manual cropping UI appears
5. Process each recipe's 2 pages in sequence
6. Total: 10 cropping sessions for 5 recipes

## Benefits Over Automatic Extraction

- **Accuracy**: User selects exact images needed
- **Control**: Skip irrelevant decorative elements
- **Quality**: Choose the best representation of the dish
- **Flexibility**: Adapt to any layout or format