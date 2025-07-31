import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from ..llm_client import LLMClient

try:
    # import layoutparser as lp  # Not using due to detectron2 dependency
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    LAYOUTPARSER_AVAILABLE = False


class BoundingBox(BaseModel):
    """Bounding box coordinates for cropping."""

    x_min: float = Field(description="Left edge as percentage of image width (0-100)")
    y_min: float = Field(description="Top edge as percentage of image height (0-100)")
    x_max: float = Field(description="Right edge as percentage of image width (0-100)")
    y_max: float = Field(
        description="Bottom edge as percentage of image height (0-100)"
    )


class FoodImage(BaseModel):
    """Represents a food image detected on a page."""

    description: str = Field(description="Detailed description of what the image shows")
    location: str = Field(
        description="Where on the page (e.g., 'full page', 'top half', 'embedded', 'none')"
    )
    is_main_photo: bool = Field(
        default=False,
        description="Whether this is the main/hero photo of the finished dish",
    )
    is_step_photo: bool = Field(
        default=False, description="Whether this is a step-by-step process photo"
    )
    bounding_box: Optional[BoundingBox] = Field(
        default=None,
        description="Precise bounding box for the food image if cropping is needed",
    )


class ImageAnalysis(BaseModel):
    """Analysis result from LLM for image content."""

    page_type: str = Field(
        description="Type of page: 'recipe_page', 'food_photo', or 'other'"
    )
    has_recipe_text: bool = Field(
        default=False, description="Whether the page contains recipe text"
    )
    recipe_title: Optional[str] = Field(
        default=None, description="Recipe title if visible on a recipe page"
    )
    has_ingredients_list: bool = Field(
        default=False, description="Whether the page contains an ingredients list"
    )
    has_instructions: bool = Field(
        default=False, description="Whether the page contains cooking instructions"
    )
    recipe_text_location: Optional[str] = Field(
        default=None, description="Description of where recipe text is located"
    )
    food_images: List[FoodImage] = Field(
        default_factory=list, description="List of food images found on the page"
    )
    should_crop: bool = Field(
        default=False, description="Whether images should be cropped from the page"
    )
    page_description: Optional[str] = Field(
        default=None, description="Overall description of the page content"
    )


class RecipeImageExtractor:
    """Extract and crop recipe-related images from cookbook pages."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.supported_formats = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".heic",
            ".heif",
        }

    def analyze_image_content(
        self, image_path: str, recipe_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use LLM to analyze image content and identify recipe-related regions."""

        # Prepare image for LLM
        try:
            # For HEIC images
            if Path(image_path).suffix.lower() in {".heic", ".heif"}:
                try:
                    from pillow_heif import register_heif_opener

                    register_heif_opener()
                except ImportError:
                    raise ValueError("HEIC/HEIF support requires pillow-heif")

            img = Image.open(image_path)
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(
                    img, mask=img.split()[-1] if img.mode == "RGBA" else None
                )
                img = background
            elif img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # Resize for analysis
            max_size = (1024, 1024)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

            original_size = Image.open(image_path).size

        finally:
            if "img" in locals():
                img.close()

        # Analyze with LLM
        recipe_context = (
            f" The recipe being extracted is '{recipe_name}'." if recipe_name else ""
        )
        analysis_prompt = f"""Analyze this image and identify recipe-related content.{recipe_context}

Please identify:
1. Page type classification:
   - "recipe_page": Contains recipe text (title, ingredients list, instructions, cooking times, servings, etc.)
   - "food_photo": Primarily shows finished dish, ingredients, or cooking process photos
   - "other": Non-recipe content

2. For RECIPE PAGES, identify:
   - Recipe title if visible
   - Whether it contains an ingredients list
   - Whether it contains cooking instructions/directions
   - Any food photos embedded in the page
   - Page layout (text only, text with photos, etc.)

3. For FOOD PHOTOS, describe:
   - What food/dish is shown (be specific based on the recipe name if provided)
   - Whether it's a finished dish (main photo) or process photo (step photo)
   - Image quality and composition
   - Use the recipe name to provide accurate descriptions (e.g., if the recipe is "Miso Buttermilk Biscuits", describe them as biscuits, not scones)

4. Important: A page with recipe text (ingredients, instructions) should ALWAYS be classified as "recipe_page" even if it also contains photos. Only classify as "food_photo" if the page is PRIMARILY or ONLY a photograph.

5. For food images worth extracting:
   - Only identify SIGNIFICANT food images that would be useful as recipe photos
   - Ignore small decorative elements, partial images, or background images
   - The image should be clear, prominent, and show the food well
   - If the image is just a small decorative element or partial view, set is_main_photo and is_step_photo to false
   - Estimate location:
   - "full page" if the image takes up most of the page
   - "top half", "bottom half", "left half", "right half"
   - "embedded" if it's a small image within text (these are usually NOT worth extracting)
   - "none" if no food images present

6. IMPORTANT: For recipe pages with text, only mark food images as is_main_photo=true or is_step_photo=true if they are:
   - Large and prominent (not small decorative elements)
   - Clear photos of the actual food (not partial views or background elements)
   - Worth extracting as standalone recipe photos

7. For each food image that should be cropped:
   - Provide a bounding_box with coordinates as percentages (0-100) of the image dimensions
   - x_min/y_min is the top-left corner, x_max/y_max is the bottom-right corner
   - For example: a food photo in the top half would be approximately {{\"x_min\": 0, \"y_min\": 0, \"x_max\": 100, \"y_max\": 50}}
   - Be precise - identify the exact boundaries of the food photo, excluding any surrounding text or margins
   - If the food image takes up the full page, you can omit the bounding_box"""

        messages = [
            {
                "role": "system",
                "content": "You are an image analysis assistant specializing in cookbook and recipe content.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": analysis_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ]

        # Use structured output with Pydantic model
        response = self.llm_client.client.beta.chat.completions.parse(
            model=self.llm_client.model,
            messages=messages,
            temperature=0.1,
            response_format=ImageAnalysis,
        )

        # Get the parsed response directly
        analysis = response.choices[0].message.parsed

        # Convert to dict and add extra metadata
        result = analysis.model_dump()
        result["original_size"] = original_size
        result["image_path"] = image_path

        return result

    def detect_image_regions_with_layoutparser(
        self, image_path: str
    ) -> List[Dict[str, Any]]:
        """Use LayoutParser to detect image regions in document pages."""
        if not LAYOUTPARSER_AVAILABLE:
            print("LayoutParser not available, falling back to OpenCV method")
            return self.detect_content_regions(image_path)

        try:
            # Since detectron2 isn't installed, use vision-based approach
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL for HEIC
                pil_img = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Use enhanced OpenCV method to detect image regions
            return self.detect_image_regions_opencv_enhanced(image_path)

        except Exception as e:
            print(f"Image region detection failed: {e}")
            print("Falling back to basic OpenCV method")
            return self.detect_content_regions(image_path)

    def detect_image_regions_opencv_enhanced(
        self, image_path: str
    ) -> List[Dict[str, Any]]:
        """Enhanced OpenCV method to detect image regions in document pages."""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            pil_img = Image.open(image_path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        height, width = img.shape[:2]

        # Method 1: Edge-based detection for photos with clear borders
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Edge detection with Canny
        edges = cv2.Canny(filtered, 30, 100)

        # Close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter for rectangular photo-like regions
        image_regions = []
        min_area = 0.1 * width * height  # At least 10% of page
        max_area = 0.9 * width * height  # At most 90% of page (to exclude full page)

        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Look for rectangular shapes (4 vertices)
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Filter by aspect ratio
                    aspect_ratio = w / h
                    if 0.3 < aspect_ratio < 3.0:
                        # Check if region has high color variance
                        region = img[y : y + h, x : x + w]
                        if region.size > 0:
                            # Calculate color statistics
                            gray_region = gray[y : y + h, x : x + w]
                            std_dev = np.std(gray_region)

                            # Photos typically have higher standard deviation than white backgrounds
                            if std_dev > 30:
                                image_regions.append(
                                    {
                                        "x": x,
                                        "y": y,
                                        "width": w,
                                        "height": h,
                                        "confidence": min(1.0, std_dev / 80.0),
                                    }
                                )

        # If no regions found with edge detection, try color-based detection
        if not image_regions:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Create mask for non-white areas (potential photo regions)
            lower_bound = np.array([0, 0, 0])
            upper_bound = np.array([180, 255, 240])  # Exclude very bright (white) areas
            mask = cv2.inRange(hsv, lower_bound, upper_bound)

            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find contours in the mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.3 < aspect_ratio < 3.0:
                        image_regions.append(
                            {
                                "x": x,
                                "y": y,
                                "width": w,
                                "height": h,
                                "confidence": 0.7,  # Lower confidence for color-based detection
                            }
                        )

        # Merge overlapping regions
        merged_regions = self._merge_overlapping_regions_dict(image_regions)

        # If still no regions found but image has significant color content,
        # return a slightly cropped version of the full image
        if not merged_regions:
            # Check if the image has significant content (not just white page)
            mean_val = np.mean(gray)
            if mean_val < 240:  # Not a blank white page
                # Return slightly inset region to remove borders
                margin = int(0.05 * min(width, height))  # 5% margin
                merged_regions = [
                    {
                        "x": margin,
                        "y": margin,
                        "width": width - 2 * margin,
                        "height": height - 2 * margin,
                        "confidence": 0.5,
                    }
                ]

        return merged_regions

    def detect_content_regions(self, image_path: str) -> List[Dict[str, Any]]:
        """Fallback: Use OpenCV to detect distinct content regions in the image."""

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            # Try with PIL for HEIC
            pil_img = Image.open(image_path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and get bounding boxes for significant regions
        regions = []
        min_area = 0.05 * img.shape[0] * img.shape[1]  # At least 5% of image

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append(
                    {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "confidence": 0.5,  # Lower confidence for CV method
                    }
                )

        # Merge overlapping regions
        merged_regions = self._merge_overlapping_regions_dict(regions)

        return merged_regions

    def _merge_overlapping_regions_dict(
        self, regions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge overlapping bounding boxes (dict version)."""
        if not regions:
            return []

        # Sort by x coordinate
        regions = sorted(regions, key=lambda r: r["x"])

        merged = []
        current = regions[0]

        for next_region in regions[1:]:
            x1, y1, w1, h1 = (
                current["x"],
                current["y"],
                current["width"],
                current["height"],
            )
            x2, y2, w2, h2 = (
                next_region["x"],
                next_region["y"],
                next_region["width"],
                next_region["height"],
            )

            # Check if regions overlap
            if x1 <= x2 <= x1 + w1 and y1 <= y2 <= y1 + h1:
                # Merge
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                current = {
                    "x": new_x,
                    "y": new_y,
                    "width": new_w,
                    "height": new_h,
                    "confidence": max(current["confidence"], next_region["confidence"]),
                }
            else:
                merged.append(current)
                current = next_region

        merged.append(current)
        return merged

    def crop_and_save_image(
        self, image_path: str, crop_region: Tuple[int, int, int, int], output_path: str
    ) -> str:
        """Crop the image and save it."""

        img = Image.open(image_path)

        x, y, w, h = crop_region
        cropped = img.crop((x, y, x + w, y + h))

        # Save with good quality
        cropped.save(output_path, quality=95, optimize=True)

        return output_path

    def extract_recipe_images(
        self, image_paths: List[str], recipe_name: str, recipe_dir: str
    ) -> Dict[str, Any]:
        """Extract and organize recipe images from multiple input images."""

        # The recipe directory already exists from save_recipe
        recipe_path = Path(recipe_dir)
        images_dir = recipe_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "recipe_name": recipe_name,
            "source_images": len(image_paths),
            "extracted_images": [],
        }

        # Track how many photos we've seen to avoid duplicate filenames
        main_photo_count = 0
        step_photo_count = 0
        other_photo_count = 0

        for idx, image_path in enumerate(image_paths):
            try:
                # Analyze image with recipe name context
                analysis = self.analyze_image_content(image_path, recipe_name)

                # Skip non-food/recipe content
                if analysis["page_type"] == "other":
                    continue

                # Get food images from analysis
                food_images = analysis.get("food_images", [])

                # Handle different page types
                if analysis["page_type"] == "food_photo" and not food_images:
                    # For pure food photo pages without specific regions identified
                    food_images = [
                        {
                            "description": analysis.get(
                                "page_description", "Recipe photo"
                            ),
                            "location": "full page",
                            "is_main_photo": True,
                            "is_step_photo": False,
                        }
                    ]
                elif analysis["page_type"] == "recipe_page":
                    # For recipe pages, only save them if they have significant embedded food images
                    # Skip pure text pages or pages with only minor decorative images
                    significant_images = [
                        img
                        for img in food_images
                        if img.get("location") not in ["embedded", "none"]
                        and (img.get("is_main_photo") or img.get("is_step_photo"))
                    ]
                    if not significant_images:
                        # Skip this page - it's just recipe text without significant food images
                        continue
                    food_images = significant_images

                if not food_images:
                    continue

                # Get the original image size for cropping calculations
                original_img = Image.open(image_path)
                img_width, img_height = original_img.size
                original_img.close()

                # Detect image regions using LayoutParser or OpenCV
                detected_regions = self.detect_image_regions_with_layoutparser(
                    image_path
                )

                if detected_regions:
                    print(
                        f"\nDetected {len(detected_regions)} image regions in {Path(image_path).name}"
                    )
                    for i, region in enumerate(detected_regions):
                        print(
                            f"  Region {i}: x={region['x']}, y={region['y']}, w={region['width']}, h={region['height']}, confidence={region.get('confidence', 0):.2f}"
                        )

                # Process food images
                for img_idx, food_image in enumerate(food_images):
                    if food_image.get("is_main_photo"):
                        if main_photo_count == 0:
                            filename = "main.jpg"
                        else:
                            filename = f"main_{main_photo_count + 1}.jpg"
                        main_photo_count += 1
                    elif food_image.get("is_step_photo"):
                        step_photo_count += 1
                        filename = f"step_{step_photo_count}.jpg"
                    else:
                        other_photo_count += 1
                        filename = f"image_{other_photo_count}.jpg"

                    output_file = images_dir / filename

                    # Try to find a matching detected region for this food image
                    was_cropped = False

                    if detected_regions:
                        # Find the best matching region
                        best_region = None
                        if len(detected_regions) == 1:
                            # If only one region detected, use it
                            best_region = detected_regions[0]
                        else:
                            # Multiple regions - could match based on location description
                            # For now, take the largest region
                            best_region = max(
                                detected_regions, key=lambda r: r["width"] * r["height"]
                            )

                        if best_region:
                            # Use the detected region for cropping
                            crop_region = (
                                best_region["x"],
                                best_region["y"],
                                best_region["width"],
                                best_region["height"],
                            )
                            self.crop_and_save_image(
                                image_path, crop_region, str(output_file)
                            )
                            was_cropped = True
                            print(
                                f"  -> Cropped image to region: x={crop_region[0]}, y={crop_region[1]}, w={crop_region[2]}, h={crop_region[3]}"
                            )

                    if not was_cropped:
                        # Copy full image
                        img = Image.open(image_path)
                        img.save(str(output_file), quality=95, optimize=True)
                        img.close()
                        print("  -> Saved full image (no cropping needed)")

                    metadata["extracted_images"].append(
                        {
                            "filename": filename,
                            "source": Path(image_path).name,
                            "description": food_image.get("description", ""),
                            "is_main": food_image.get("is_main_photo", False),
                            "is_step": food_image.get("is_step_photo", False),
                            "cropped": was_cropped,
                        }
                    )

            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue

        # Save metadata
        metadata_file = images_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata
