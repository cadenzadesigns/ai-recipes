import base64
import io
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image


class ImageExtractor:
    """Extract recipe content from images."""

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.heic', '.heif'}

    def __init__(self, max_size: tuple = (1024, 1024)):
        self.max_size = max_size

    def process_image(self, image_path: str) -> List[Dict[str, Any]]:
        """Process a single image and return content for LLM."""
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {path.suffix}")

        # Open and resize image if needed
        try:
            # For HEIC images, we might need pillow-heif
            if path.suffix.lower() in {'.heic', '.heif'}:
                try:
                    from pillow_heif import register_heif_opener
                    register_heif_opener()
                except ImportError:
                    raise ValueError(
                        "HEIC/HEIF support requires pillow-heif. Install with: uv add pillow-heif"
                    )

            img = Image.open(image_path)
            # Convert RGBA to RGB if necessary
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')

            # Resize if larger than max size
            if img.size[0] > self.max_size[0] or img.size[1] > self.max_size[1]:
                img.thumbnail(self.max_size, Image.Resampling.LANCZOS)

            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        finally:
            if 'img' in locals():
                img.close()

        return [
            {
                "type": "text",
                "text": "Please extract the recipe from this image:"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]

    def process_multiple_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple images and return combined content for LLM."""
        content = [
            {
                "type": "text",
                "text": "Please extract the complete recipe from these images. If the recipe spans multiple images, combine all the information:"
            }
        ]

        for image_path in image_paths:
            try:
                # Process each image and add to content
                single_image_content = self.process_image(image_path)
                # Only add the image part, not the text prompt
                content.append(single_image_content[1])
            except Exception as e:
                print(f"Warning: Failed to process image {image_path}: {str(e)}")
                continue

        if len(content) == 1:
            raise ValueError("No valid images could be processed")

        return content
