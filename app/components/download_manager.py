"""
Download management component for the AI Recipe Extractor Streamlit app.

This module handles all download operations including single recipe downloads,
batch downloads, ZIP archive creation, and integration with the existing
recipe formatting system.
"""

import json

# Import from the existing src modules
import sys
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import streamlit as st

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.config import (
    DOWNLOAD_FORMATS,
    MAX_ZIP_SIZE,
    ZIP_COMPRESSION_LEVEL,
    format_file_size,
    get_safe_filename,
)
from src.formatter import RecipeFormatter
from src.models import Recipe, RecipeImage


class DownloadManager:
    """Manages all download operations for the Streamlit web app."""

    def __init__(self):
        self.formatter = RecipeFormatter()

    def create_single_recipe_download(
        self,
        recipe: Recipe,
        format_type: str = "txt",
        include_images: bool = False,
        custom_filename: Optional[str] = None,
    ) -> Tuple[bytes, str, str]:
        """
        Create a download for a single recipe.

        Args:
            recipe: The recipe to download
            format_type: Format type ('txt', 'json', 'zip')
            include_images: Whether to include images (only for ZIP format)
            custom_filename: Custom filename (optional)

        Returns:
            Tuple of (file_content, filename, mime_type)
        """
        if format_type not in ["txt", "json", "zip"]:
            raise ValueError(f"Unsupported format: {format_type}")

        # Generate filename
        if custom_filename:
            base_name = get_safe_filename(custom_filename)
        else:
            base_name = get_safe_filename(recipe.name)

        if format_type == "txt":
            content = recipe.to_text().encode("utf-8")
            filename = f"{base_name}.txt"
            mime_type = DOWNLOAD_FORMATS["txt"]["mime"]

        elif format_type == "json":
            content = json.dumps(recipe.model_dump(), indent=2).encode("utf-8")
            filename = f"{base_name}.json"
            mime_type = DOWNLOAD_FORMATS["json"]["mime"]

        elif format_type == "zip":
            content, filename = self.create_recipe_with_images_zip(recipe, base_name)
            mime_type = DOWNLOAD_FORMATS["zip"]["mime"]

        return content, filename, mime_type

    def create_batch_recipe_download(
        self,
        recipes: List[Recipe],
        format_type: str = "txt",
        include_images: bool = False,
        custom_filename: Optional[str] = None,
    ) -> Tuple[bytes, str, str]:
        """
        Create a download for multiple recipes.

        Args:
            recipes: List of recipes to download
            format_type: Format type ('txt', 'json', 'zip')
            include_images: Whether to include images (only for ZIP format)
            custom_filename: Custom filename (optional)

        Returns:
            Tuple of (file_content, filename, mime_type)
        """
        if not recipes:
            raise ValueError("No recipes provided for batch download")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if custom_filename:
            base_name = get_safe_filename(custom_filename)
        else:
            base_name = f"recipes_batch_{timestamp}"

        if format_type == "txt":
            content = self._create_batch_text_content(recipes)
            filename = f"{base_name}.txt"
            mime_type = DOWNLOAD_FORMATS["txt"]["mime"]

        elif format_type == "json":
            content = self._create_batch_json_content(recipes)
            filename = f"{base_name}.json"
            mime_type = DOWNLOAD_FORMATS["json"]["mime"]

        elif format_type == "zip":
            content, filename = self._create_batch_zip_content(
                recipes, base_name, include_images
            )
            mime_type = DOWNLOAD_FORMATS["zip"]["mime"]

        else:
            raise ValueError(f"Unsupported format: {format_type}")

        return content, filename, mime_type

    def create_recipe_with_images_zip(
        self, recipe: Recipe, base_name: Optional[str] = None
    ) -> Tuple[bytes, str]:
        """
        Create a ZIP archive containing a recipe and its images.

        Args:
            recipe: The recipe to include
            base_name: Base name for the ZIP file

        Returns:
            Tuple of (zip_content, filename)
        """
        if not base_name:
            base_name = get_safe_filename(recipe.name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{base_name}_{timestamp}.zip"

        # Create ZIP in memory
        zip_buffer = BytesIO()

        with zipfile.ZipFile(
            zip_buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=ZIP_COMPRESSION_LEVEL
        ) as zip_file:

            # Add recipe text file
            recipe_text = recipe.to_text()
            zip_file.writestr(f"{base_name}.txt", recipe_text)

            # Add recipe JSON file
            recipe_json = json.dumps(recipe.model_dump(), indent=2)
            zip_file.writestr(f"{base_name}.json", recipe_json)

            # Add images if present
            if recipe.images:
                self._add_images_to_zip(zip_file, recipe.images, base_name)

        zip_buffer.seek(0)
        zip_content = zip_buffer.getvalue()

        # Check size limit
        if len(zip_content) > MAX_ZIP_SIZE:
            raise ValueError(
                f"ZIP file too large: {format_file_size(len(zip_content))}"
            )

        return zip_content, zip_filename

    def _create_batch_text_content(self, recipes: List[Recipe]) -> bytes:
        """Create combined text content for multiple recipes."""
        content_lines = []

        # Header
        content_lines.append("Recipe Collection")
        content_lines.append("=" * 70)
        content_lines.append(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        content_lines.append(f"Total recipes: {len(recipes)}")
        content_lines.append("=" * 70)
        content_lines.append("")

        # Add each recipe
        for i, recipe in enumerate(recipes, 1):
            if i > 1:
                content_lines.append("")
                content_lines.append("=" * 70)
                content_lines.append("")

            content_lines.append(f"[Recipe {i} of {len(recipes)}]")
            content_lines.append("")
            content_lines.append(recipe.to_text())

        return "\n".join(content_lines).encode("utf-8")

    def _create_batch_json_content(self, recipes: List[Recipe]) -> bytes:
        """Create JSON array content for multiple recipes."""
        recipe_data = {
            "generated_on": datetime.now().isoformat(),
            "total_recipes": len(recipes),
            "recipes": [recipe.model_dump() for recipe in recipes],
        }

        return json.dumps(recipe_data, indent=2).encode("utf-8")

    def _create_batch_zip_content(
        self, recipes: List[Recipe], base_name: str, include_images: bool = False
    ) -> Tuple[bytes, str]:
        """Create ZIP archive for multiple recipes."""
        zip_filename = f"{base_name}.zip"
        zip_buffer = BytesIO()

        with zipfile.ZipFile(
            zip_buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=ZIP_COMPRESSION_LEVEL
        ) as zip_file:

            # Add batch text file
            batch_text = self._create_batch_text_content(recipes).decode("utf-8")
            zip_file.writestr(f"{base_name}.txt", batch_text)

            # Add batch JSON file
            batch_json = self._create_batch_json_content(recipes).decode("utf-8")
            zip_file.writestr(f"{base_name}.json", batch_json)

            # Add individual recipe files
            for i, recipe in enumerate(recipes, 1):
                recipe_name = get_safe_filename(recipe.name)
                recipe_dir = f"recipes/{recipe_name}"

                # Individual recipe text
                zip_file.writestr(f"{recipe_dir}/{recipe_name}.txt", recipe.to_text())

                # Individual recipe JSON
                zip_file.writestr(
                    f"{recipe_dir}/{recipe_name}.json",
                    json.dumps(recipe.model_dump(), indent=2),
                )

                # Add images if requested and available
                if include_images and recipe.images:
                    self._add_images_to_zip(zip_file, recipe.images, recipe_dir)

        zip_buffer.seek(0)
        zip_content = zip_buffer.getvalue()

        # Check size limit
        if len(zip_content) > MAX_ZIP_SIZE:
            raise ValueError(
                f"ZIP file too large: {format_file_size(len(zip_content))}"
            )

        return zip_content, zip_filename

    def _add_images_to_zip(
        self, zip_file: zipfile.ZipFile, images: List[RecipeImage], base_path: str
    ) -> None:
        """Add recipe images to ZIP archive."""
        if not images:
            return

        images_dir = f"{base_path}/images"

        # Create metadata file for images
        image_metadata = {
            "images": [
                {
                    "filename": img.filename,
                    "description": img.description,
                    "is_main": img.is_main,
                    "is_step": img.is_step,
                }
                for img in images
            ]
        }

        zip_file.writestr(
            f"{images_dir}/metadata.json", json.dumps(image_metadata, indent=2)
        )

        # Note: In a real implementation, you would need access to the actual
        # image files. This is a placeholder for the image inclusion logic.
        # The actual images would need to be read from the file system or
        # stored in the Recipe object as bytes.

        for image in images:
            # Placeholder - in real implementation, would read actual image file
            zip_file.writestr(
                f"{images_dir}/{image.filename}",
                f"[Image placeholder: {image.filename}]\n{image.description}",
            )

    def format_filename(self, text: str, extension: str = "") -> str:
        """
        Format a safe filename from text.

        Args:
            text: Input text to convert to filename
            extension: File extension (with or without dot)

        Returns:
            Safe filename string
        """
        safe_name = get_safe_filename(text)

        if extension:
            if not extension.startswith("."):
                extension = f".{extension}"
            return f"{safe_name}{extension}"

        return safe_name

    def get_download_link(
        self,
        recipe: Recipe,
        format_type: str = "txt",
        label: Optional[str] = None,
        filename: Optional[str] = None,
        key: Optional[str] = None,
    ) -> None:
        """
        Create and display a Streamlit download button.

        Args:
            recipe: Recipe to download
            format_type: Download format ('txt', 'json', 'zip')
            label: Button label (optional)
            filename: Custom filename (optional)
            key: Streamlit component key (optional)
        """
        try:
            content, download_filename, mime_type = self.create_single_recipe_download(
                recipe, format_type, filename
            )

            if not label:
                label = DOWNLOAD_FORMATS[format_type]["label"]

            st.download_button(
                label=label,
                data=content,
                file_name=download_filename,
                mime=mime_type,
                key=key,
            )

        except Exception as e:
            st.error(f"Failed to create download: {str(e)}")

    def get_batch_download_link(
        self,
        recipes: List[Recipe],
        format_type: str = "txt",
        label: Optional[str] = None,
        filename: Optional[str] = None,
        include_images: bool = False,
        key: Optional[str] = None,
    ) -> None:
        """
        Create and display a Streamlit download button for batch recipes.

        Args:
            recipes: List of recipes to download
            format_type: Download format ('txt', 'json', 'zip')
            label: Button label (optional)
            filename: Custom filename (optional)
            include_images: Include images in download (ZIP only)
            key: Streamlit component key (optional)
        """
        try:
            content, download_filename, mime_type = self.create_batch_recipe_download(
                recipes, format_type, include_images, filename
            )

            if not label:
                if format_type == "zip":
                    label = f"📦 Download {len(recipes)} Recipes as ZIP"
                else:
                    label = f"{DOWNLOAD_FORMATS[format_type]['label']} ({len(recipes)} recipes)"

            st.download_button(
                label=label,
                data=content,
                file_name=download_filename,
                mime=mime_type,
                key=key,
            )

        except Exception as e:
            st.error(f"Failed to create batch download: {str(e)}")

    def create_download_section(
        self,
        recipe: Recipe,
        show_zip: bool = True,
        section_title: str = "💾 Download Options",
    ) -> None:
        """
        Create a complete download section with multiple format options.

        Args:
            recipe: Recipe to create downloads for
            show_zip: Whether to show ZIP download option
            section_title: Title for the download section
        """
        st.markdown(f"#### {section_title}")

        cols = st.columns(3 if show_zip else 2)

        with cols[0]:
            self.get_download_link(
                recipe, "txt", key=f"download_txt_{hash(recipe.name)}"
            )

        with cols[1]:
            self.get_download_link(
                recipe, "json", key=f"download_json_{hash(recipe.name)}"
            )

        if show_zip and len(cols) > 2:
            with cols[2]:
                self.get_download_link(
                    recipe,
                    "zip",
                    label="📦 Download with Images",
                    key=f"download_zip_{hash(recipe.name)}",
                )

    def create_batch_download_section(
        self,
        recipes: List[Recipe],
        show_individual: bool = True,
        section_title: str = "💾 Bulk Download Options",
    ) -> None:
        """
        Create a complete batch download section.

        Args:
            recipes: List of recipes for batch download
            show_individual: Whether to show individual recipe downloads
            section_title: Title for the download section
        """
        if not recipes:
            st.warning("No recipes available for download.")
            return

        st.markdown(f"#### {section_title}")

        # Batch downloads
        st.markdown("**Download All Recipes:**")

        cols = st.columns(3)

        with cols[0]:
            self.get_batch_download_link(recipes, "txt", key="batch_download_txt")

        with cols[1]:
            self.get_batch_download_link(recipes, "json", key="batch_download_json")

        with cols[2]:
            self.get_batch_download_link(
                recipes, "zip", include_images=True, key="batch_download_zip"
            )

        # Individual downloads
        if show_individual and len(recipes) > 1:
            st.markdown("---")
            st.markdown("**Individual Recipe Downloads:**")

            for i, recipe in enumerate(recipes):
                with st.expander(f"📄 {recipe.name}"):
                    self.create_download_section(
                        recipe, show_zip=False, section_title="Download this recipe:"
                    )

    def get_download_progress(
        self, current: int, total: int, operation: str = "Processing"
    ) -> None:
        """
        Display download/processing progress.

        Args:
            current: Current item being processed
            total: Total items to process
            operation: Description of the operation
        """
        progress = current / total if total > 0 else 0

        st.progress(progress)
        st.text(f"{operation} {current} of {total}...")

    def validate_download_request(
        self, recipes: Union[Recipe, List[Recipe]], format_type: str
    ) -> bool:
        """
        Validate a download request.

        Args:
            recipes: Recipe or list of recipes
            format_type: Requested format type

        Returns:
            True if valid, False otherwise
        """
        # Check format type
        if format_type not in DOWNLOAD_FORMATS:
            st.error(f"Unsupported download format: {format_type}")
            return False

        # Check if recipes exist
        if isinstance(recipes, list):
            if not recipes:
                st.error("No recipes available for download.")
                return False
        elif not recipes:
            st.error("No recipe available for download.")
            return False

        return True

    def estimate_download_size(
        self,
        recipes: Union[Recipe, List[Recipe]],
        format_type: str,
        include_images: bool = False,
    ) -> str:
        """
        Estimate download size for given recipes and format.

        Args:
            recipes: Recipe or list of recipes
            format_type: Download format
            include_images: Whether images are included

        Returns:
            Formatted size estimate string
        """
        if isinstance(recipes, Recipe):
            recipes = [recipes]

        # Base size estimation (rough)
        text_size_per_recipe = 2048  # ~2KB per recipe in text
        json_size_per_recipe = 3072  # ~3KB per recipe in JSON

        if format_type == "txt":
            estimated_size = len(recipes) * text_size_per_recipe
        elif format_type == "json":
            estimated_size = len(recipes) * json_size_per_recipe
        elif format_type == "zip":
            estimated_size = len(recipes) * (
                text_size_per_recipe + json_size_per_recipe
            )
            # Add image size estimate if including images
            if include_images:
                avg_image_size = 500 * 1024  # 500KB average per image
                total_images = sum(len(r.images) if r.images else 0 for r in recipes)
                estimated_size += total_images * avg_image_size
        else:
            estimated_size = 1024  # 1KB default

        return format_file_size(estimated_size)


# Convenience function for easy import
def get_download_manager() -> DownloadManager:
    """Get a configured DownloadManager instance."""
    return DownloadManager()
