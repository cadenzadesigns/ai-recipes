"""Recipe image extraction with manual cropping option."""

from typing import Any, Dict, List

import click

from ..models import RecipeImage
from .manual_cropper import ManualImageCropper


class ManualRecipeImageExtractor:
    """Extract recipe images using manual cropping interface."""

    def __init__(self):
        self.cropper = ManualImageCropper()

    def extract_recipe_images(
        self,
        image_paths: List[str],
        recipe_name: str,
        recipe_dir: str,
        progress_callback=None,
        interactive: bool = True,
    ) -> Dict[str, Any]:
        """Extract recipe images using manual cropping.

        Args:
            image_paths: List of paths to images to process
            recipe_name: Name of the recipe
            recipe_dir: Directory where recipe is saved
            progress_callback: Optional callback for progress updates
            interactive: Whether to use interactive cropping (default True)
        """

        if not interactive:
            # If not interactive, skip image extraction
            return {
                "recipe_name": recipe_name,
                "manual_crop": False,
                "source_images": 0,
                "extracted_images": [],
            }

        # Check if we're in a headless environment
        import os

        if os.environ.get("DISPLAY") is None and os.name != "nt":
            click.echo("Warning: No display detected. Skipping manual image cropping.")
            click.echo(
                "To enable manual cropping, run with a display or use X11 forwarding."
            )
            return {
                "recipe_name": recipe_name,
                "manual_crop": False,
                "source_images": 0,
                "extracted_images": [],
            }

        try:
            # Get crop regions from user
            crop_regions = self.cropper.crop_multiple_images(image_paths, recipe_name)

            if not crop_regions:
                click.echo("No images selected for extraction.")
                return {
                    "recipe_name": recipe_name,
                    "manual_crop": True,
                    "source_images": len(image_paths),
                    "extracted_images": [],
                }

            # Save cropped images
            metadata = self.cropper.save_cropped_images(
                crop_regions, recipe_name, recipe_dir
            )

            click.echo(
                f"✓ Manually extracted {len(metadata['extracted_images'])} images"
            )

            return metadata

        except Exception as e:
            click.echo(f"Error during manual cropping: {str(e)}")
            # Return empty metadata on error
            return {
                "recipe_name": recipe_name,
                "manual_crop": True,
                "source_images": len(image_paths),
                "extracted_images": [],
                "error": str(e),
            }

    def convert_to_recipe_images(self, metadata: Dict[str, Any]) -> List[RecipeImage]:
        """Convert metadata to RecipeImage objects."""
        return [
            RecipeImage(
                filename=img["filename"],
                description=img["description"],
                is_main=img["is_main"],
                is_step=img["is_step"],
            )
            for img in metadata.get("extracted_images", [])
        ]
