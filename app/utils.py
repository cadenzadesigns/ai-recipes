"""Utility functions for the Streamlit web app."""

import base64
import io
import json
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image

from src.models import Recipe


def manage_session_state() -> None:
    """Initialize and manage Streamlit session state variables."""
    # Initialize session state variables if they don't exist
    if "recipes" not in st.session_state:
        st.session_state.recipes = []

    if "selected_recipe" not in st.session_state:
        st.session_state.selected_recipe = None

    if "recipe_dirs" not in st.session_state:
        st.session_state.recipe_dirs = {}

    if "upload_history" not in st.session_state:
        st.session_state.upload_history = []

    if "current_view" not in st.session_state:
        st.session_state.current_view = "gallery"

    if "processing_status" not in st.session_state:
        st.session_state.processing_status = {}


def clear_session_state() -> None:
    """Clear all session state data."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    manage_session_state()


def add_recipe_to_session(recipe: Recipe, recipe_dir: Optional[str] = None) -> None:
    """
    Add a recipe to the session state.

    Args:
        recipe: Recipe object to add
        recipe_dir: Path to recipe directory
    """
    if "recipes" not in st.session_state:
        st.session_state.recipes = []

    if "recipe_dirs" not in st.session_state:
        st.session_state.recipe_dirs = {}

    # Check if recipe already exists (by name)
    existing_names = [r.name for r in st.session_state.recipes]
    if recipe.name not in existing_names:
        st.session_state.recipes.append(recipe)
        if recipe_dir:
            st.session_state.recipe_dirs[recipe.name] = recipe_dir


def load_recipes_from_directory(
    recipes_dir: str = "recipes",
) -> List[Tuple[Recipe, str]]:
    """
    Load all recipes from the recipes directory.

    Args:
        recipes_dir: Base directory containing recipe folders

    Returns:
        List of (Recipe, recipe_dir) tuples
    """
    recipes_path = Path(recipes_dir)
    if not recipes_path.exists():
        return []

    recipes = []

    for recipe_folder in recipes_path.iterdir():
        if not recipe_folder.is_dir():
            continue

        # Look for JSON file in the recipe folder
        json_files = list(recipe_folder.glob("*.json"))
        if not json_files:
            continue

        json_file = json_files[0]  # Take the first JSON file

        try:
            with open(json_file, encoding="utf-8") as f:
                recipe_data = json.load(f)

            recipe = Recipe(**recipe_data)
            recipes.append((recipe, str(recipe_folder)))

        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Error loading recipe from {json_file}: {e}")
            continue

    return recipes


def process_uploaded_file(uploaded_file, file_type: str) -> Dict[str, Any]:
    """
    Process an uploaded file and return processing status.

    Args:
        uploaded_file: Streamlit uploaded file object
        file_type: Type of file ('image', 'pdf', 'text')

    Returns:
        Dictionary with processing status and results
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Store processing status
        status = {
            "filename": uploaded_file.name,
            "file_type": file_type,
            "status": "processing",
            "tmp_path": tmp_path,
            "uploaded_at": datetime.now(),
            "error": None,
            "recipe": None,
            "recipe_dir": None,
        }

        return status

    except Exception as e:
        return {
            "filename": uploaded_file.name,
            "file_type": file_type,
            "status": "error",
            "error": str(e),
            "uploaded_at": datetime.now(),
        }


def generate_thumbnails(
    image_path: str, sizes: List[Tuple[int, int]] = None
) -> Dict[str, str]:
    """
    Generate thumbnails for an image.

    Args:
        image_path: Path to the source image
        sizes: List of (width, height) tuples for thumbnail sizes

    Returns:
        Dictionary mapping size names to base64 encoded thumbnails
    """
    if sizes is None:
        sizes = [
            (150, 150),  # small
            (300, 300),  # medium
            (600, 600),  # large
        ]

    thumbnails = {}

    try:
        with Image.open(image_path) as img:
            # Convert RGBA to RGB if necessary
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(
                    img, mask=img.split()[-1] if img.mode == "RGBA" else None
                )
                img = background

            for i, (width, height) in enumerate(sizes):
                # Create thumbnail maintaining aspect ratio
                img_copy = img.copy()
                img_copy.thumbnail((width, height), Image.Resampling.LANCZOS)

                # Convert to base64
                buffer = io.BytesIO()
                img_copy.save(buffer, format="JPEG", quality=85)
                img_b64 = base64.b64encode(buffer.getvalue()).decode()

                size_name = ["small", "medium", "large"][i] if i < 3 else f"size_{i}"
                thumbnails[size_name] = f"data:image/jpeg;base64,{img_b64}"

    except Exception as e:
        st.error(f"Error generating thumbnails for {image_path}: {e}")

    return thumbnails


def create_batch_zip(
    recipes: List[Recipe],
    recipe_dirs: Optional[List[str]] = None,
    include_images: bool = True,
) -> bytes:
    """
    Create a ZIP file containing multiple recipes and their images.

    Args:
        recipes: List of Recipe objects
        recipe_dirs: List of recipe directory paths
        include_images: Whether to include recipe images in the ZIP

    Returns:
        ZIP file content as bytes
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Add recipe files
        for i, recipe in enumerate(recipes):
            recipe_dir = (
                recipe_dirs[i] if recipe_dirs and i < len(recipe_dirs) else None
            )
            safe_name = _get_safe_filename(recipe.name)

            # Add text file
            text_content = recipe.to_text()
            zip_file.writestr(f"{safe_name}/{safe_name}.txt", text_content)

            # Add JSON file
            json_content = json.dumps(recipe.model_dump(), indent=2)
            zip_file.writestr(f"{safe_name}/{safe_name}.json", json_content)

            # Add images if available and requested
            if include_images and recipe_dir and recipe.images:
                recipe_path = Path(recipe_dir)
                images_dir = recipe_path / "images"

                if images_dir.exists():
                    # Add images with metadata
                    for img in recipe.images:
                        img_path = images_dir / img.filename
                        if img_path.exists():
                            with open(img_path, "rb") as img_file:
                                zip_file.writestr(
                                    f"{safe_name}/images/{img.filename}",
                                    img_file.read(),
                                )

                    # Add metadata.json if it exists
                    metadata_path = images_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path) as meta_file:
                            zip_file.writestr(
                                f"{safe_name}/images/metadata.json", meta_file.read()
                            )

        # Add collection index
        index_content = create_recipe_index(recipes)
        zip_file.writestr("recipe_index.txt", index_content)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def create_recipe_index(recipes: List[Recipe]) -> str:
    """
    Create a text index of all recipes.

    Args:
        recipes: List of Recipe objects

    Returns:
        Formatted index as string
    """
    lines = []
    lines.append("Recipe Collection Index")
    lines.append("=" * 50)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total recipes: {len(recipes)}")
    lines.append("")

    for i, recipe in enumerate(recipes, 1):
        lines.append(f"{i}. {recipe.name}")
        if recipe.description:
            desc = (
                recipe.description[:100] + "..."
                if len(recipe.description) > 100
                else recipe.description
            )
            lines.append(f"   {desc}")

        metadata = []
        if recipe.servings:
            metadata.append(f"Serves: {recipe.servings}")
        if recipe.total_time:
            metadata.append(f"Time: {recipe.total_time}")
        if recipe.source:
            metadata.append(f"Source: {recipe.source}")

        if metadata:
            lines.append(f"   {' • '.join(metadata)}")

        lines.append("")

    return "\n".join(lines)


def validate_recipe_data(recipe_dict: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate recipe data dictionary.

    Args:
        recipe_dict: Dictionary containing recipe data

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Try to create Recipe object to validate
        Recipe(**recipe_dict)
        return True, None
    except Exception as e:
        return False, str(e)


def format_error_message(error: Exception, context: str = "") -> str:
    """
    Format error messages for display in Streamlit.

    Args:
        error: Exception object
        context: Additional context about where the error occurred

    Returns:
        Formatted error message
    """
    error_type = type(error).__name__
    error_msg = str(error)

    if context:
        return f"**{error_type}** in {context}: {error_msg}"
    else:
        return f"**{error_type}**: {error_msg}"


def get_file_size_str(file_size: int) -> str:
    """
    Convert file size in bytes to human-readable string.

    Args:
        file_size: Size in bytes

    Returns:
        Formatted file size string
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if file_size < 1024.0:
            return f"{file_size:.1f} {unit}"
        file_size /= 1024.0
    return f"{file_size:.1f} TB"


def cleanup_temp_files(temp_paths: List[str]) -> None:
    """
    Clean up temporary files.

    Args:
        temp_paths: List of temporary file paths to delete
    """
    for path in temp_paths:
        try:
            if Path(path).exists():
                Path(path).unlink()
        except Exception as e:
            st.warning(f"Could not delete temporary file {path}: {e}")


def save_upload_history(
    filename: str, file_type: str, status: str, recipe_name: Optional[str] = None
) -> None:
    """
    Save upload history to session state.

    Args:
        filename: Original filename
        file_type: Type of file uploaded
        status: Processing status
        recipe_name: Name of extracted recipe (if successful)
    """
    if "upload_history" not in st.session_state:
        st.session_state.upload_history = []

    history_entry = {
        "timestamp": datetime.now(),
        "filename": filename,
        "file_type": file_type,
        "status": status,
        "recipe_name": recipe_name,
    }

    st.session_state.upload_history.append(history_entry)

    # Keep only last 50 entries
    if len(st.session_state.upload_history) > 50:
        st.session_state.upload_history = st.session_state.upload_history[-50:]


def display_upload_history() -> None:
    """Display upload history in Streamlit."""
    if "upload_history" not in st.session_state or not st.session_state.upload_history:
        st.info("No upload history available.")
        return

    st.markdown("### Upload History")

    for entry in reversed(st.session_state.upload_history[-10:]):  # Show last 10
        timestamp = entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

        with col1:
            st.write(f"**{entry['filename']}**")

        with col2:
            st.write(entry["file_type"].title())

        with col3:
            if entry["status"] == "success":
                st.success("✓")
            elif entry["status"] == "error":
                st.error("✗")
            else:
                st.info("⏳")

        with col4:
            if entry.get("recipe_name"):
                st.write(entry["recipe_name"])
            else:
                st.write(timestamp)


def _get_safe_filename(filename: str) -> str:
    """Convert a string to a safe filename."""
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    safe_name = "".join(c if c in safe_chars else "_" for c in filename)
    # Remove multiple underscores and strip
    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")
    return safe_name.strip("_")


def export_recipes_to_formats(
    recipes: List[Recipe], recipe_dirs: Optional[List[str]] = None
) -> Dict[str, bytes]:
    """
    Export recipes to various formats.

    Args:
        recipes: List of Recipe objects
        recipe_dirs: Optional list of recipe directory paths

    Returns:
        Dictionary mapping format names to file contents
    """
    exports = {}

    # Combined text format
    text_lines = []
    text_lines.append("Recipe Collection")
    text_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    text_lines.append(f"Total recipes: {len(recipes)}")
    text_lines.append("=" * 70)
    text_lines.append("")

    for i, recipe in enumerate(recipes, 1):
        if i > 1:
            text_lines.append("\n" + "=" * 70 + "\n")
        text_lines.append(f"[Recipe {i} of {len(recipes)}]")
        text_lines.append(recipe.to_text())

    exports["combined_text"] = "\n".join(text_lines).encode("utf-8")

    # Combined JSON format
    json_data = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "total_recipes": len(recipes),
            "format_version": "1.0",
        },
        "recipes": [recipe.model_dump() for recipe in recipes],
    }
    exports["combined_json"] = json.dumps(json_data, indent=2).encode("utf-8")

    # ZIP with all files and images
    exports["complete_zip"] = create_batch_zip(
        recipes, recipe_dirs, include_images=True
    )

    return exports


def get_recipe_statistics(recipes: List[Recipe]) -> Dict[str, Any]:
    """
    Calculate statistics for a collection of recipes.

    Args:
        recipes: List of Recipe objects

    Returns:
        Dictionary containing various statistics
    """
    if not recipes:
        return {}

    stats = {
        "total_recipes": len(recipes),
        "total_ingredients": sum(len(recipe.ingredients) for recipe in recipes),
        "total_steps": sum(len(recipe.directions) for recipe in recipes),
        "avg_ingredients": sum(len(recipe.ingredients) for recipe in recipes)
        / len(recipes),
        "avg_steps": sum(len(recipe.directions) for recipe in recipes) / len(recipes),
        "recipes_with_images": sum(1 for recipe in recipes if recipe.images),
        "recipes_with_notes": sum(1 for recipe in recipes if recipe.notes),
        "recipes_with_source": sum(1 for recipe in recipes if recipe.source),
    }

    # Most common ingredients
    ingredient_counts = {}
    for recipe in recipes:
        for ingredient in recipe.ingredients:
            name = ingredient.item.name.lower().strip()
            ingredient_counts[name] = ingredient_counts.get(name, 0) + 1

    stats["common_ingredients"] = sorted(
        ingredient_counts.items(), key=lambda x: x[1], reverse=True
    )[:10]

    # Recipe sources
    source_counts = {}
    for recipe in recipes:
        if recipe.source:
            source_counts[recipe.source] = source_counts.get(recipe.source, 0) + 1

    stats["sources"] = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)

    return stats
