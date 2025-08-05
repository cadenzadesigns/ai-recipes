# Recipe Viewer

## Overview

The AI Recipe Extractor now includes a built-in recipe viewer that allows you to browse and view all extracted recipes in a beautiful, Streamlit-native interface.

## Features

### Recipe Collection Page
- **Grid Layout**: All recipes displayed in an attractive card grid
- **Recipe Cards**: Each card shows:
  - Recipe image (if available)
  - Recipe name
  - Description preview
  - Servings and total time
- **Search Functionality**: Real-time search to filter recipes by name or description
- **Native Navigation**: Click "View Recipe" to see full details

### Individual Recipe Pages
- **Full Recipe Details**:
  - Recipe name and alternate names
  - Description
  - Servings and total time
  - Complete ingredient list (with support for recipe components)
  - Step-by-step directions
  - Notes and tips
  - Source attribution
- **Recipe Images**:
  - Main recipe photo displayed prominently
  - Step-by-step photos in a grid layout
  - Image captions and descriptions
- **Export Options**: Download recipes as TXT or JSON files
- **Paprika Compatibility**: Recipe pages are formatted to be easily imported by Paprika Recipe Manager's in-app browser

## How It Works

1. **Automatic HTML Generation**: When recipes are extracted and saved, HTML pages are automatically generated for potential static serving
2. **Streamlit Pages**: The viewer uses Streamlit's native multipage functionality:
   - `pages/recipe_collection.py` - Recipe collection browser
   - `pages/recipe_viewer.py` - Individual recipe viewer
3. **Session State Navigation**: Recipes are passed between pages using Streamlit's session state

## Usage

1. Extract recipes using any of the available methods (image, web, PDF)
2. Access the recipe collection:
   - Click the "Browse Recipe Collection →" button in the sidebar
3. Search for recipes or browse the collection
4. Click "View Recipe" on any recipe card to see full details
5. Use the back button or sidebar to return to browsing

## Implementation Details

### New Files
- `app/recipe_html_generator.py`: Generates HTML pages for recipes (for static serving if needed)
- `app/recipe_utils.py`: Utility functions for recipe management
- `pages/recipe_collection.py`: Streamlit page for browsing recipes
- `pages/recipe_viewer.py`: Streamlit page for viewing individual recipes

### Modified Files
- `app/main.py`: 
  - Added HTML generation to recipe saving process
  - Added recipe collection button to sidebar
  - Integrated recipe utilities

## Technical Notes

- Uses Streamlit's native multipage functionality (st.Page, st.switch_page)
- Recipe data is passed between pages using st.session_state
- HTML files are still generated for potential future static serving
- Images are loaded directly from the recipe directories
- Search functionality is implemented using Streamlit's text_input widget