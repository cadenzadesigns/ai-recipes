"""Manual image cropping interface using OpenCV."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False


class ManualImageCropper:
    """Interactive manual image cropping using OpenCV GUI."""

    def __init__(self):
        self.current_image = None
        self.display_image = None
        self.original_shape = None
        self.scale_factor = 1.0
        self.rectangles = []
        self.current_rect = None
        self.drawing = False
        self.window_name = "Manual Image Cropping"

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image and convert HEIC if necessary."""

        # Try to load with OpenCV first
        img = cv2.imread(str(image_path))

        if img is None:
            # Try with PIL for HEIC/HEIF support
            pil_img = Image.open(image_path)
            # Convert to RGB if necessary
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            # Convert PIL to OpenCV format
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return img

    def resize_for_display(
        self, image: np.ndarray, max_width: int = 1200, max_height: int = 800
    ) -> Tuple[np.ndarray, float]:
        """Resize image to fit screen while maintaining aspect ratio."""
        height, width = image.shape[:2]

        # Calculate scale factor
        scale_w = max_width / width if width > max_width else 1.0
        scale_h = max_height / height if height > max_height else 1.0
        scale = min(scale_w, scale_h)

        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
            return resized, scale

        return image.copy(), 1.0

    def draw_rectangles(self):
        """Draw all rectangles on the display image."""
        self.display_image = self.current_image.copy()

        # Draw completed rectangles
        for i, rect in enumerate(self.rectangles):
            cv2.rectangle(
                self.display_image,
                (rect["x1"], rect["y1"]),
                (rect["x2"], rect["y2"]),
                (0, 255, 0),
                2,
            )

            # Add label
            label = f"Image {i+1}"
            cv2.putText(
                self.display_image,
                label,
                (rect["x1"] + 5, rect["y1"] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Draw current rectangle being drawn
        if self.current_rect:
            cv2.rectangle(
                self.display_image,
                (self.current_rect["x1"], self.current_rect["y1"]),
                (self.current_rect["x2"], self.current_rect["y2"]),
                (255, 0, 0),
                2,
            )

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing rectangles."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_rect = {"x1": x, "y1": y, "x2": x, "y2": y}

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_rect["x2"] = x
                self.current_rect["y2"] = y
                self.draw_rectangles()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                # Normalize rectangle coordinates
                x1 = min(self.current_rect["x1"], self.current_rect["x2"])
                y1 = min(self.current_rect["y1"], self.current_rect["y2"])
                x2 = max(self.current_rect["x1"], self.current_rect["x2"])
                y2 = max(self.current_rect["y1"], self.current_rect["y2"])

                # Only add if rectangle has meaningful size
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    self.rectangles.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

                self.current_rect = None
                self.draw_rectangles()

    def crop_single_image(
        self, image_path: str, num_crops: int = 1
    ) -> List[Dict[str, Any]]:
        """Interactively crop regions from a single image.

        Returns list of crop regions with coordinates scaled to original image size.
        """
        # Load and prepare image
        original_image = self.load_image(image_path)
        self.original_shape = original_image.shape[:2]

        # Resize for display
        self.current_image, self.scale_factor = self.resize_for_display(original_image)
        self.display_image = self.current_image.copy()
        self.rectangles = []

        # Create window and set up mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Instructions
        instructions = [
            f"Draw {num_crops} rectangle(s) around the recipe image(s) you want to keep.",
            "Click and drag to draw a rectangle.",
            "Press 'c' to clear all rectangles.",
            "Press 'z' to undo last rectangle.",
            "Press SPACE or ENTER when done.",
            "Press ESC to cancel.",
        ]

        print("\n".join(instructions))

        while True:
            # Display image
            cv2.imshow(self.window_name, self.display_image)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC - cancel
                cv2.destroyWindow(self.window_name)
                return []

            elif key in [13, 32]:  # Enter or Space - done
                if len(self.rectangles) > 0:
                    cv2.destroyWindow(self.window_name)
                    break
                else:
                    print("Please draw at least one rectangle before continuing.")

            elif key == ord("c"):  # Clear all
                self.rectangles = []
                self.draw_rectangles()
                cv2.imshow(self.window_name, self.display_image)

            elif key == ord("z"):  # Undo last
                if self.rectangles:
                    self.rectangles.pop()
                    self.draw_rectangles()
                    cv2.imshow(self.window_name, self.display_image)

        # Convert rectangles to original image coordinates
        crop_regions = []
        for rect in self.rectangles:
            # Scale back to original image size
            x1 = int(rect["x1"] / self.scale_factor)
            y1 = int(rect["y1"] / self.scale_factor)
            x2 = int(rect["x2"] / self.scale_factor)
            y2 = int(rect["y2"] / self.scale_factor)

            crop_regions.append(
                {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "original_image": str(image_path),
                }
            )

        return crop_regions

    def crop_multiple_images(
        self, image_paths: List[str], recipe_name: str
    ) -> List[Dict[str, Any]]:
        """Process multiple images, allowing user to select and crop from each."""
        all_crops = []

        print(f"\nProcessing {len(image_paths)} images for recipe: {recipe_name}")
        print(
            "For each image, draw rectangles around the recipe images you want to keep."
        )
        print("Press SPACE to skip an image if it contains no recipe images.\n")

        for idx, image_path in enumerate(image_paths):
            print(f"\nImage {idx + 1} of {len(image_paths)}: {Path(image_path).name}")

            # Ask user how many crops for this image
            while True:
                try:
                    response = input(
                        "How many recipe images to extract from this page? (0 to skip): "
                    )
                    num_crops = int(response)
                    if num_crops >= 0:
                        break
                    print("Please enter a non-negative number.")
                except ValueError:
                    print("Please enter a valid number.")

            if num_crops == 0:
                print("Skipping this image.")
                continue

            # Get crops for this image
            crops = self.crop_single_image(image_path, num_crops)

            # Add image index for tracking
            for crop in crops:
                crop["source_index"] = idx

            all_crops.extend(crops)

        return all_crops

    def save_cropped_images(
        self, crop_regions: List[Dict[str, Any]], recipe_name: str, recipe_dir: str
    ) -> Dict[str, Any]:
        """Save cropped images to recipe directory with metadata."""
        recipe_path = Path(recipe_dir)
        images_dir = recipe_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Create originals subdirectory
        originals_dir = images_dir / "originals"
        originals_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "recipe_name": recipe_name,
            "manual_crop": True,
            "source_images": len(set(crop["original_image"] for crop in crop_regions)),
            "extracted_images": [],
        }

        # Copy original images and track which ones we've copied
        copied_originals = set()
        for crop in crop_regions:
            original_path = Path(crop["original_image"])
            if str(original_path) not in copied_originals:
                dest_path = originals_dir / original_path.name
                shutil.copy2(str(original_path), str(dest_path))
                copied_originals.add(str(original_path))

        # Process and save crops
        for idx, crop in enumerate(crop_regions):
            # Determine filename
            if idx == 0:
                filename = "main.jpg"
                is_main = True
            else:
                filename = f"image_{idx}.jpg"
                is_main = False

            output_path = images_dir / filename

            # Load and crop image
            img = self.load_image(crop["original_image"])
            x, y = crop["x"], crop["y"]
            w, h = crop["width"], crop["height"]

            # Ensure coordinates are within bounds
            h_orig, w_orig = img.shape[:2]
            x = max(0, min(x, w_orig - 1))
            y = max(0, min(y, h_orig - 1))
            x2 = min(x + w, w_orig)
            y2 = min(y + h, h_orig)

            cropped = img[y:y2, x:x2]

            # Save cropped image
            cv2.imwrite(str(output_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])

            metadata["extracted_images"].append(
                {
                    "filename": filename,
                    "source": Path(crop["original_image"]).name,
                    "description": f"Manually cropped recipe image {idx + 1}",
                    "is_main": is_main,
                    "is_step": False,
                    "cropped": True,
                    "crop_region": {"x": x, "y": y, "width": w, "height": h},
                }
            )

        # Save metadata
        metadata_file = images_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata
