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
        """Interactive cropping interface using streamlit-cropper with thumbnails."""
        from streamlit_cropper import st_cropper
        
        st.write("### Select Recipe Images")
        
        # Description
        st.info("🖱️ Click and drag to draw rectangles around recipe images. You can crop multiple regions from the same page.")

        # Initialize crop regions for this image if not exists
        if image_key not in st.session_state.crop_regions:
            st.session_state.crop_regions[image_key] = []

        # Initialize current crop index if not exists
        crop_key = f"{image_key}_crop_index"
        if crop_key not in st.session_state:
            st.session_state[crop_key] = 0

        current_crop_index = st.session_state[crop_key]
        
        # Show current progress
        if st.session_state.crop_regions[image_key]:
            st.success(f"✅ {len(st.session_state.crop_regions[image_key])} image(s) selected")
        
        # Create thumbnail for cropping while keeping original for final crop
        original_image = image
        thumbnail_max_size = 800  # Max width or height for thumbnail
        
        # Calculate thumbnail size maintaining aspect ratio
        if image.width > thumbnail_max_size or image.height > thumbnail_max_size:
            if image.width > image.height:
                new_width = thumbnail_max_size
                new_height = int((image.height * thumbnail_max_size) / image.width)
            else:
                new_height = thumbnail_max_size
                new_width = int((image.width * thumbnail_max_size) / image.height)
            
            thumbnail = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            scale_x = image.width / new_width
            scale_y = image.height / new_height
        else:
            thumbnail = image
            scale_x = 1.0
            scale_y = 1.0
        
        # Show image dimensions
        st.caption(f"📏 Original: {original_image.width}×{original_image.height}px | Thumbnail: {thumbnail.width}×{thumbnail.height}px")
        
        # Main cropping interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"#### Crop Region {current_crop_index + 1}")
            
            # Interactive cropper on thumbnail
            cropped_img = st_cropper(
                thumbnail,
                realtime_update=False,  # Only update when crop is changed, not on every mouse move
                box_color="green",
                aspect_ratio=None,  # Free aspect ratio
                key=f"{image_key}_cropper_{current_crop_index}",
                return_type="both"  # Return both image and coordinates
            )
            
            # Image type selection
            col_a, col_b = st.columns(2)
            with col_a:
                is_main = st.radio(
                    "Image type:",
                    ["Main recipe image", "Step/side image"],
                    key=f"{image_key}_type_{current_crop_index}",
                    index=0 if len(st.session_state.crop_regions[image_key]) == 0 else 1
                )
            
            with col_b:
                description = st.text_input(
                    "Description (optional):",
                    placeholder="e.g., 'Final dish', 'Step 3'",
                    key=f"{image_key}_desc_{current_crop_index}"
                )

        with col2:
            st.write("#### Actions")
            
            # Add cropped region
            if st.button("✅ Save This Crop", key=f"{image_key}_save", type="primary"):
                # Check if we have a valid crop region
                crop_valid = False
                rect = None
                cropped_thumbnail = None
                
                if cropped_img is not None:
                    # Handle tuple format (image, rect)
                    if isinstance(cropped_img, tuple) and len(cropped_img) == 2:
                        cropped_thumbnail, rect = cropped_img
                        if rect and rect.get('width', 0) > 0 and rect.get('height', 0) > 0:
                            crop_valid = True
                    # Handle object format with attributes
                    elif hasattr(cropped_img, 'img') and cropped_img.img is not None:
                        cropped_thumbnail = cropped_img.img
                        rect = cropped_img.rect
                        crop_valid = True
                    elif hasattr(cropped_img, 'rect') and cropped_img.rect:
                        rect = cropped_img.rect
                        if rect.get('width', 0) > 0 and rect.get('height', 0) > 0:
                            crop_valid = True
                
                if crop_valid:
                    # Scale coordinates back to original image size
                    scaled_coords = {
                        "x1": int(rect['left'] * scale_x),
                        "y1": int(rect['top'] * scale_y),
                        "x2": int((rect['left'] + rect['width']) * scale_x),
                        "y2": int((rect['top'] + rect['height']) * scale_y)
                    }
                    
                    # Ensure coordinates are within image bounds
                    scaled_coords['x1'] = max(0, min(scaled_coords['x1'], original_image.width))
                    scaled_coords['y1'] = max(0, min(scaled_coords['y1'], original_image.height))
                    scaled_coords['x2'] = max(0, min(scaled_coords['x2'], original_image.width))
                    scaled_coords['y2'] = max(0, min(scaled_coords['y2'], original_image.height))
                    
                    # Crop from the original full-size image using scaled coordinates
                    original_cropped = original_image.crop((
                        scaled_coords['x1'], 
                        scaled_coords['y1'], 
                        scaled_coords['x2'], 
                        scaled_coords['y2']
                    ))
                    
                    crop_data = {
                        "coordinates": scaled_coords,
                        "cropped_image": original_cropped,  # Use crop from original image
                        "is_main": is_main == "Main recipe image",
                        "description": description or ("Main recipe image" if is_main == "Main recipe image" else "Recipe step image")
                    }
                    
                    st.session_state.crop_regions[image_key].append(crop_data)
                    
                    # Increment crop index for next crop
                    st.session_state[crop_key] = current_crop_index + 1
                    
                    st.success("✅ Crop saved!")
                    st.rerun()
                else:
                    st.error("❌ Please draw a crop region first - click and drag on the image to select an area")
            
            # Show current crops
            st.write("#### Saved Crops")
            if st.session_state.crop_regions[image_key]:
                for i, crop in enumerate(st.session_state.crop_regions[image_key]):
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        icon = "🏠" if crop.get("is_main") else "📝"
                        st.text(f"{icon} Crop {i+1}")
                        if crop.get("description"):
                            st.caption(crop["description"])
                    with col_b:
                        if st.button("🗑️", key=f"{image_key}_remove_{i}", help="Remove this crop"):
                            st.session_state.crop_regions[image_key].pop(i)
                            # Reset crop index if we removed crops
                            if st.session_state[crop_key] > len(st.session_state.crop_regions[image_key]):
                                st.session_state[crop_key] = len(st.session_state.crop_regions[image_key])
                            st.rerun()
            else:
                st.info("No crops saved yet")
            
            # Clear all button
            if st.session_state.crop_regions[image_key] and st.button("🗑️ Clear All", key=f"{image_key}_clear_all"):
                st.session_state.crop_regions[image_key] = []
                st.session_state[crop_key] = 0
                st.rerun()
            
            # Finish button
            if st.session_state.crop_regions[image_key]:
                # Create a callback to set the done flag
                def mark_page_done():
                    st.session_state[f"{image_key}_page_complete"] = True
                
                if st.button("✅ Done with Page", 
                           key=f"{image_key}_done", 
                           type="secondary",
                           on_click=mark_page_done):
                    st.success(f"✅ Finished cropping {len(st.session_state.crop_regions[image_key])} images from this page!")

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
