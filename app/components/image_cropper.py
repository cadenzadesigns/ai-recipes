"""Web-based image cropping component for Streamlit."""

import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw


def get_image_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


class StreamlitImageCropper:
    """Web-based image cropping interface for Streamlit."""

    def __init__(self):
        if "crop_regions" not in st.session_state:
            st.session_state.crop_regions = {}
        if "current_image_index" not in st.session_state:
            st.session_state.current_image_index = 0

    def crop_single_image_canvas(
        self, image: Image.Image, image_key: str, max_crops: int = 5
    ) -> List[Dict]:
        """Manual cropping interface using canvas and coordinate inputs."""
        st.write("### Draw Crop Regions")

        # Initialize crop regions for this image if not exists
        if image_key not in st.session_state.crop_regions:
            st.session_state.crop_regions[image_key] = []

        # Display image with existing crop regions
        display_img = image.copy()
        draw = ImageDraw.Draw(display_img)

        # Draw existing regions
        for i, region in enumerate(st.session_state.crop_regions[image_key]):
            x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            draw.text((x1 + 5, y1 + 5), f"Image {i+1}", fill="green")

        # Display the image
        st.image(display_img, use_column_width=True)

        # Manual coordinate input
        col1, col2 = st.columns(2)

        with col1:
            st.write("#### Add New Crop Region")
            x1 = st.number_input(
                "X1 (left)",
                min_value=0,
                max_value=image.width,
                value=0,
                key=f"{image_key}_x1",
            )
            y1 = st.number_input(
                "Y1 (top)",
                min_value=0,
                max_value=image.height,
                value=0,
                key=f"{image_key}_y1",
            )
            x2 = st.number_input(
                "X2 (right)",
                min_value=0,
                max_value=image.width,
                value=image.width,
                key=f"{image_key}_x2",
            )
            y2 = st.number_input(
                "Y2 (bottom)",
                min_value=0,
                max_value=image.height,
                value=image.height,
                key=f"{image_key}_y2",
            )

            if st.button("Add Region", key=f"{image_key}_add"):
                if x2 > x1 and y2 > y1:
                    new_region = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    st.session_state.crop_regions[image_key].append(new_region)
                    st.rerun()
                else:
                    st.error("Invalid region: X2 must be > X1 and Y2 must be > Y1")

        with col2:
            st.write("#### Current Regions")
            if st.session_state.crop_regions[image_key]:
                for i, region in enumerate(st.session_state.crop_regions[image_key]):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.text(
                            f"Region {i+1}: ({region['x1']}, {region['y1']}) to ({region['x2']}, {region['y2']})"
                        )
                    with col_b:
                        if st.button("Remove", key=f"{image_key}_remove_{i}"):
                            st.session_state.crop_regions[image_key].pop(i)
                            st.rerun()
            else:
                st.info("No regions defined yet")

            if st.button("Clear All", key=f"{image_key}_clear"):
                st.session_state.crop_regions[image_key] = []
                st.rerun()

        return st.session_state.crop_regions[image_key]

    def crop_single_image_simple(
        self, image: Image.Image, image_key: str
    ) -> Optional[Dict]:
        """Simple single-region cropping using streamlit-cropper."""
        # Try to use streamlit-cropper if available
        try:
            from streamlit_cropper import st_cropper

            st.write("### Crop Recipe Image")
            st.info("Draw a rectangle around the recipe image you want to keep.")

            # Crop the image
            cropped_img = st_cropper(
                image,
                realtime_update=True,
                box_color="green",
                aspect_ratio=None,
                key=f"{image_key}_cropper",
            )

            # Get crop coordinates
            if cropped_img is not None and cropped_img.size != image.size:
                # Calculate crop box coordinates
                # This is approximate since st_cropper doesn't return exact coordinates
                return {
                    "x1": 0,
                    "y1": 0,
                    "x2": cropped_img.width,
                    "y2": cropped_img.height,
                    "cropped_img": cropped_img,
                }
            return None

        except ImportError:
            st.warning(
                "streamlit-cropper not installed. Using coordinate input instead."
            )
            return None

    def crop_multiple_images(
        self, images: List[Tuple[str, Image.Image]], recipe_name: str
    ) -> Dict[str, List[Dict]]:
        """Process multiple images for cropping."""
        st.write(f"## Manual Image Cropping for: {recipe_name}")

        total_images = len(images)

        # Image navigation
        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            if st.button(
                "◀ Previous", disabled=st.session_state.current_image_index == 0
            ):
                st.session_state.current_image_index -= 1
                st.rerun()

        with col2:
            st.write(
                f"### Image {st.session_state.current_image_index + 1} of {total_images}"
            )
            current_path, current_image = images[st.session_state.current_image_index]
            st.write(f"**File:** {Path(current_path).name}")

        with col3:
            if st.button(
                "Next ▶",
                disabled=st.session_state.current_image_index >= total_images - 1,
            ):
                st.session_state.current_image_index += 1
                st.rerun()

        # Image size info
        st.write(f"**Size:** {current_image.width} × {current_image.height} pixels")

        # Number of crops for this image
        image_key = f"img_{st.session_state.current_image_index}"

        num_crops = st.selectbox(
            "How many recipe images to extract from this page?",
            options=[0, 1, 2, 3, 4, 5],
            key=f"{image_key}_num_crops",
            help="Select 0 to skip this image",
        )

        if num_crops > 0:
            # Use canvas-based cropping
            crop_regions = self.crop_single_image_canvas(
                current_image, image_key, num_crops
            )

            if len(crop_regions) < num_crops:
                st.warning(
                    f"Please define {num_crops - len(crop_regions)} more region(s)"
                )
        else:
            st.info("This image will be skipped")
            if image_key in st.session_state.crop_regions:
                st.session_state.crop_regions[image_key] = []

        # Progress indicator
        st.progress((st.session_state.current_image_index + 1) / total_images)

        # Action buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Reset Current Image", type="secondary"):
                st.session_state.crop_regions[image_key] = []
                st.rerun()

        with col2:
            if st.session_state.current_image_index < total_images - 1:
                if st.button("Save & Continue", type="primary"):
                    st.session_state.current_image_index += 1
                    st.rerun()

        with col3:
            if st.session_state.current_image_index == total_images - 1:
                if st.button("Finish Cropping", type="primary"):
                    # Compile all crop regions
                    all_crops = {}
                    for i, (path, img) in enumerate(images):
                        img_key = f"img_{i}"
                        if (
                            img_key in st.session_state.crop_regions
                            and st.session_state.crop_regions[img_key]
                        ):
                            all_crops[path] = st.session_state.crop_regions[img_key]
                    return all_crops

        return None

    def save_cropped_images(
        self,
        images: Dict[str, Image.Image],
        crop_regions: Dict[str, List[Dict]],
        recipe_name: str,
        output_dir: Path,
    ) -> Dict:
        """Save cropped images to output directory."""
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Create originals subdirectory
        originals_dir = images_dir / "originals"
        originals_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "recipe_name": recipe_name,
            "manual_crop": True,
            "source_images": len(images),
            "extracted_images": [],
        }

        image_counter = 0

        for image_path, regions in crop_regions.items():
            # Save original
            original_name = Path(image_path).name
            original_img = images[image_path]
            original_img.save(originals_dir / original_name)

            # Process each crop region
            for region in regions:
                # Determine filename
                if image_counter == 0:
                    filename = "main.jpg"
                    is_main = True
                else:
                    filename = f"image_{image_counter}.jpg"
                    is_main = False

                # Crop and save
                cropped = original_img.crop(
                    (region["x1"], region["y1"], region["x2"], region["y2"])
                )
                cropped.save(images_dir / filename, quality=95)

                metadata["extracted_images"].append(
                    {
                        "filename": filename,
                        "source": original_name,
                        "description": f"Manually cropped recipe image {image_counter + 1}",
                        "is_main": is_main,
                        "is_step": False,
                        "cropped": True,
                        "crop_region": {
                            "x": region["x1"],
                            "y": region["y1"],
                            "width": region["x2"] - region["x1"],
                            "height": region["y2"] - region["y1"],
                        },
                    }
                )

                image_counter += 1

        # Save metadata
        with open(images_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata
