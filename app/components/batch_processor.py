import io
import json

# Import your existing components
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List

import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


from formatter import RecipeFormatter

from extractors.image import ImageExtractor
from extractors.recipe_image_extractor import RecipeImageExtractor
from llm_client import LLMClient
from models import Recipe, RecipeImage


class BatchProcessingResult:
    """Container for batch processing results."""

    def __init__(self):
        self.successful_recipes: List[Recipe] = []
        self.failed_extractions: List[Dict] = []
        self.processing_logs: List[str] = []
        self.total_processed: int = 0
        self.success_count: int = 0
        self.failure_count: int = 0


def batch_processor_ui():
    """Main UI component for batch processing multiple images."""

    st.header("🍳 Batch Recipe Processor")
    st.write(
        "Upload multiple images to extract recipes in batch. Process them individually or combine into one multi-page recipe."
    )

    # File uploader for multiple images
    uploaded_files = st.file_uploader(
        "Upload Recipe Images",
        type=["png", "jpg", "jpeg", "gif", "bmp", "webp", "heic", "heif"],
        accept_multiple_files=True,
        help="Select multiple images containing recipes. Supported formats: PNG, JPG, JPEG, GIF, BMP, WebP, HEIC, HEIF",
    )

    if not uploaded_files:
        st.info("👆 Upload multiple images to get started")
        return

    # Display uploaded images in a grid
    display_image_grid(uploaded_files)

    # Processing options
    st.subheader("Processing Options")

    col1, col2 = st.columns(2)

    with col1:
        processing_mode = st.radio(
            "Processing Mode",
            options=["individual", "combined"],
            format_func=lambda x: {
                "individual": "🔄 Individual Recipes",
                "combined": "📚 Combined Recipe",
            }[x],
            help={
                "individual": "Process each image as a separate recipe (default behavior)",
                "combined": "Process all images as one multi-page recipe",
            }[st.session_state.get("processing_mode", "individual")],
        )

    with col2:
        enable_image_extraction = st.checkbox(
            "🖼️ Extract Recipe Images",
            value=True,
            help="Automatically detect and extract recipe images from cookbook pages",
        )

    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        max_size = st.slider(
            "Image Processing Size (pixels)",
            min_value=512,
            max_value=2048,
            value=1024,
            step=128,
            help="Maximum size for image processing. Larger sizes provide better quality but use more tokens.",
        )

        output_format = st.selectbox(
            "Output Format",
            options=["both", "json", "text"],
            format_func=lambda x: {
                "both": "📄 Both JSON & Text",
                "json": "🔧 JSON Only",
                "text": "📝 Text Only",
            }[x],
            help="Choose the output format for extracted recipes",
        )

    # Process button
    if st.button("🚀 Process Recipes", type="primary", use_container_width=True):
        if processing_mode == "individual":
            results = process_images_individually(
                uploaded_files, enable_image_extraction, max_size, output_format
            )
        else:
            results = process_images_combined(
                uploaded_files, enable_image_extraction, max_size, output_format
            )

        # Display results
        display_batch_results(results, output_format)


def display_image_grid(uploaded_files: List) -> None:
    """Display uploaded images in a responsive grid layout."""

    st.subheader(f"📸 Uploaded Images ({len(uploaded_files)})")

    # Create columns for grid layout
    cols_per_row = 4
    rows = [
        uploaded_files[i : i + cols_per_row]
        for i in range(0, len(uploaded_files), cols_per_row)
    ]

    for row in rows:
        cols = st.columns(len(row))
        for col, file in zip(cols, row):
            with col:
                try:
                    # Display thumbnail
                    image = Image.open(file)
                    # Reset file pointer for later use
                    file.seek(0)

                    st.image(image, caption=file.name, use_column_width=True)

                    # Show file info
                    st.caption(f"📏 {image.size[0]}×{image.size[1]}")

                except Exception as e:
                    st.error(f"❌ Error loading {file.name}: {str(e)}")


def process_images_individually(
    uploaded_files: List,
    enable_image_extraction: bool,
    max_size: int,
    output_format: str,
) -> BatchProcessingResult:
    """Process each image as a separate recipe."""

    results = BatchProcessingResult()
    results.total_processed = len(uploaded_files)

    # Initialize components
    try:
        llm_client = LLMClient()
        image_extractor = ImageExtractor(max_size=(max_size, max_size))
        recipe_image_extractor = (
            RecipeImageExtractor(llm_client) if enable_image_extraction else None
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize components: {str(e)}")
        return results

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Process each image
    for idx, uploaded_file in enumerate(uploaded_files):
        progress = (idx + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(
            f"Processing {uploaded_file.name} ({idx + 1}/{len(uploaded_files)})"
        )

        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded_file.name).suffix
            ) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Reset file pointer
            uploaded_file.seek(0)

            # Extract recipe from image
            content = image_extractor.process_image(tmp_file_path)
            recipe = llm_client.extract_recipe(content, source=uploaded_file.name)

            # Save recipe to /recipes directory
            formatter = RecipeFormatter()
            recipe_dir = formatter.save_recipe(recipe)

            # Handle image extraction if enabled
            if enable_image_extraction and recipe_image_extractor:
                show_processing_progress(f"Extracting images for {recipe.name}")

                # Extract images to the saved recipe directory
                image_metadata = recipe_image_extractor.extract_recipe_images(
                    [tmp_file_path], recipe.name, recipe_dir
                )

                # Add extracted images to recipe
                if image_metadata.get("extracted_images"):
                    recipe.images = []
                    for img_info in image_metadata["extracted_images"]:
                        recipe.images.append(
                            RecipeImage(
                                filename=img_info["filename"],
                                description=img_info.get("description", ""),
                                is_main=img_info.get("is_main", False),
                                is_step=img_info.get("is_step", False),
                            )
                        )

                    # Update the saved files with image references
                    formatter.update_recipe_files(recipe, recipe_dir)

            results.successful_recipes.append(recipe)
            results.success_count += 1
            results.processing_logs.append(
                f"✅ Successfully processed {uploaded_file.name} → {recipe.name}"
            )

            # Clean up temp file
            Path(tmp_file_path).unlink(missing_ok=True)

        except Exception as e:
            error_msg = f"❌ Failed to process {uploaded_file.name}: {str(e)}"
            results.failed_extractions.append(
                {"filename": uploaded_file.name, "error": str(e)}
            )
            results.failure_count += 1
            results.processing_logs.append(error_msg)

            # Clean up temp file
            if "tmp_file_path" in locals():
                Path(tmp_file_path).unlink(missing_ok=True)

    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")

    return results


def process_images_combined(
    uploaded_files: List,
    enable_image_extraction: bool,
    max_size: int,
    output_format: str,
) -> BatchProcessingResult:
    """Process all images as one multi-page recipe."""

    results = BatchProcessingResult()
    results.total_processed = len(uploaded_files)

    # Initialize components
    try:
        llm_client = LLMClient()
        image_extractor = ImageExtractor(max_size=(max_size, max_size))
        recipe_image_extractor = (
            RecipeImageExtractor(llm_client) if enable_image_extraction else None
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize components: {str(e)}")
        return results

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Preparing images for combined processing...")
        progress_bar.progress(0.1)

        # Save all uploaded files temporarily
        temp_file_paths = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded_file.name).suffix
            ) as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_file_paths.append(tmp_file.name)
            uploaded_file.seek(0)

        progress_bar.progress(0.3)
        status_text.text("Processing combined recipe from all images...")

        # Process all images together
        content = image_extractor.process_multiple_images(temp_file_paths)

        progress_bar.progress(0.6)
        status_text.text("Extracting recipe information...")

        # Extract single recipe from combined content
        combined_source = f"Combined from {len(uploaded_files)} images: {', '.join([f.name for f in uploaded_files])}"
        recipe = llm_client.extract_recipe(content, source=combined_source)

        progress_bar.progress(0.8)

        # Save recipe to /recipes directory
        formatter = RecipeFormatter()
        recipe_dir = formatter.save_recipe(recipe)

        # Handle image extraction if enabled
        if enable_image_extraction and recipe_image_extractor:
            status_text.text(f"Extracting images for {recipe.name}...")

            # Extract images to the saved recipe directory
            image_metadata = recipe_image_extractor.extract_recipe_images(
                temp_file_paths, recipe.name, recipe_dir
            )

            # Add extracted images to recipe
            if image_metadata.get("extracted_images"):
                recipe.images = []
                for img_info in image_metadata["extracted_images"]:
                    recipe.images.append(
                        RecipeImage(
                            filename=img_info["filename"],
                            description=img_info.get("description", ""),
                            is_main=img_info.get("is_main", False),
                            is_step=img_info.get("is_step", False),
                        )
                    )

                # Update the saved files with image references
                formatter.update_recipe_files(recipe, recipe_dir)

        results.successful_recipes.append(recipe)
        results.success_count = 1
        results.processing_logs.append(
            f"✅ Successfully created combined recipe: {recipe.name}"
        )

        # Clean up temp files
        for temp_path in temp_file_paths:
            Path(temp_path).unlink(missing_ok=True)

    except Exception as e:
        error_msg = f"❌ Failed to process combined recipe: {str(e)}"
        results.failed_extractions.append(
            {"filename": "Combined processing", "error": str(e)}
        )
        results.failure_count = 1
        results.processing_logs.append(error_msg)

        # Clean up temp files
        for temp_path in temp_file_paths:
            Path(temp_path).unlink(missing_ok=True)

    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")

    return results


def show_processing_progress(message: str):
    """Display a processing message."""
    if "processing_messages" not in st.session_state:
        st.session_state.processing_messages = []

    st.session_state.processing_messages.append(message)

    # Display in a container
    with st.container():
        st.info(message)


def display_batch_results(results: BatchProcessingResult, output_format: str):
    """Display the results of batch processing in an organized manner."""

    st.header("📊 Processing Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Processed", results.total_processed)
    with col2:
        st.metric("Successful", results.success_count, delta=results.success_count)
    with col3:
        st.metric(
            "Failed",
            results.failure_count,
            delta=-results.failure_count if results.failure_count > 0 else 0,
        )
    with col4:
        success_rate = (
            (results.success_count / results.total_processed * 100)
            if results.total_processed > 0
            else 0
        )
        st.metric("Success Rate", f"{success_rate:.1f}%")

    # Processing logs
    if results.processing_logs:
        with st.expander("📋 Processing Log"):
            for log_entry in results.processing_logs:
                if log_entry.startswith("✅"):
                    st.success(log_entry)
                elif log_entry.startswith("❌"):
                    st.error(log_entry)
                else:
                    st.info(log_entry)

    # Successful recipes
    if results.successful_recipes:
        st.subheader("🎉 Successfully Extracted Recipes")

        # Recipe tabs or accordion
        if len(results.successful_recipes) == 1:
            display_single_recipe_result(results.successful_recipes[0], output_format)
        else:
            tabs = st.tabs(
                [f"📝 {recipe.name}" for recipe in results.successful_recipes]
            )
            for tab, recipe in zip(tabs, results.successful_recipes):
                with tab:
                    display_single_recipe_result(recipe, output_format)

        # Batch download options
        display_batch_download_options(results.successful_recipes, output_format)

    # Failed extractions
    if results.failed_extractions:
        st.subheader("⚠️ Failed Extractions")

        for failure in results.failed_extractions:
            with st.expander(f"❌ {failure['filename']}"):
                st.error(f"Error: {failure['error']}")

                if st.button(
                    f"🔄 Retry {failure['filename']}",
                    key=f"retry_{failure['filename']}",
                ):
                    st.info("Retry functionality would be implemented here")
                    # In a full implementation, you would re-process this specific image


def display_single_recipe_result(recipe: Recipe, output_format: str):
    """Display a single recipe result with download options."""

    # Recipe preview
    st.write(f"**{recipe.name}**")
    if recipe.description:
        st.write(recipe.description)

    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        if recipe.servings:
            st.info(f"🍽️ {recipe.servings}")
    with col2:
        if recipe.total_time:
            st.info(f"⏱️ {recipe.total_time}")
    with col3:
        if recipe.source:
            st.info(f"📖 {recipe.source}")

    # Images info
    if recipe.images:
        st.write(f"🖼️ **Images:** {len(recipe.images)} extracted")
        for img in recipe.images:
            prefix = "🏆" if img.is_main else "📷" if img.is_step else "🖼️"
            st.caption(f"{prefix} {img.filename}: {img.description}")

    # Ingredients and directions preview
    with st.expander("👀 Preview Recipe Content"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Ingredients:**")
            for ingredient in recipe.ingredients[:5]:  # Show first 5
                st.write(f"• {ingredient.to_string()}")
            if len(recipe.ingredients) > 5:
                st.write(f"... and {len(recipe.ingredients) - 5} more")

        with col2:
            st.write("**Directions:**")
            for i, direction in enumerate(recipe.directions[:3], 1):  # Show first 3
                st.write(
                    f"{i}. {direction[:100]}{'...' if len(direction) > 100 else ''}"
                )
            if len(recipe.directions) > 3:
                st.write(f"... and {len(recipe.directions) - 3} more steps")

    # Individual download buttons
    col1, col2, col3 = st.columns(3)

    if output_format in ["text", "both"]:
        with col1:
            text_content = recipe.to_text()
            st.download_button(
                "📄 Download Text",
                data=text_content,
                file_name=f"{recipe.name.replace(' ', '_').lower()}.txt",
                mime="text/plain",
                key=f"download_text_{recipe.name}",
            )

    if output_format in ["json", "both"]:
        with col2:
            json_content = json.dumps(recipe.model_dump(), indent=2)
            st.download_button(
                "🔧 Download JSON",
                data=json_content,
                file_name=f"{recipe.name.replace(' ', '_').lower()}.json",
                mime="application/json",
                key=f"download_json_{recipe.name}",
            )

    with col3:
        # Combined download (both formats)
        if output_format == "both":
            zip_data = create_recipe_zip(recipe)
            st.download_button(
                "📦 Download Both",
                data=zip_data,
                file_name=f"{recipe.name.replace(' ', '_').lower()}.zip",
                mime="application/zip",
                key=f"download_zip_{recipe.name}",
            )


def display_batch_download_options(recipes: List[Recipe], output_format: str):
    """Display options for downloading all recipes in batch."""

    st.subheader("📦 Batch Download")

    col1, col2, col3 = st.columns(3)

    # All recipes as text files
    if output_format in ["text", "both"]:
        with col1:
            all_text_zip = create_batch_text_zip(recipes)
            st.download_button(
                f"📄 Download All Text ({len(recipes)} files)",
                data=all_text_zip,
                file_name="batch_recipes_text.zip",
                mime="application/zip",
            )

    # All recipes as JSON files
    if output_format in ["json", "both"]:
        with col2:
            all_json_zip = create_batch_json_zip(recipes)
            st.download_button(
                f"🔧 Download All JSON ({len(recipes)} files)",
                data=all_json_zip,
                file_name="batch_recipes_json.zip",
                mime="application/zip",
            )

    # Combined batch (all formats)
    with col3:
        if output_format == "both":
            combined_zip = create_batch_combined_zip(recipes)
            st.download_button(
                f"📦 Download Everything ({len(recipes)} recipes)",
                data=combined_zip,
                file_name="batch_recipes_complete.zip",
                mime="application/zip",
            )


def create_recipe_zip(recipe: Recipe) -> bytes:
    """Create a ZIP file containing both text and JSON versions of a recipe."""

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Add text version
        text_content = recipe.to_text()
        zip_file.writestr(f"{recipe.name.replace(' ', '_').lower()}.txt", text_content)

        # Add JSON version
        json_content = json.dumps(recipe.model_dump(), indent=2)
        zip_file.writestr(f"{recipe.name.replace(' ', '_').lower()}.json", json_content)

    zip_buffer.seek(0)
    return zip_buffer.read()


def create_batch_text_zip(recipes: List[Recipe]) -> bytes:
    """Create a ZIP file containing text versions of all recipes."""

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for recipe in recipes:
            text_content = recipe.to_text()
            filename = f"{recipe.name.replace(' ', '_').lower()}.txt"
            zip_file.writestr(filename, text_content)

    zip_buffer.seek(0)
    return zip_buffer.read()


def create_batch_json_zip(recipes: List[Recipe]) -> bytes:
    """Create a ZIP file containing JSON versions of all recipes."""

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for recipe in recipes:
            json_content = json.dumps(recipe.model_dump(), indent=2)
            filename = f"{recipe.name.replace(' ', '_').lower()}.json"
            zip_file.writestr(filename, json_content)

    zip_buffer.seek(0)
    return zip_buffer.read()


def create_batch_combined_zip(recipes: List[Recipe]) -> bytes:
    """Create a ZIP file containing both text and JSON versions of all recipes."""

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for recipe in recipes:
            base_name = recipe.name.replace(" ", "_").lower()

            # Add text version
            text_content = recipe.to_text()
            zip_file.writestr(f"{base_name}.txt", text_content)

            # Add JSON version
            json_content = json.dumps(recipe.model_dump(), indent=2)
            zip_file.writestr(f"{base_name}.json", json_content)

    zip_buffer.seek(0)
    return zip_buffer.read()


# Utility functions for retry functionality
def retry_failed_extraction(filename: str, uploaded_files: List):
    """Retry extraction for a specific failed file."""
    # This would be implemented to allow retrying individual failed extractions
    # For now, this is a placeholder for the full implementation
    pass


def get_processing_statistics(results: BatchProcessingResult) -> Dict:
    """Get detailed processing statistics."""

    stats = {
        "total_images": results.total_processed,
        "successful_recipes": results.success_count,
        "failed_extractions": results.failure_count,
        "success_rate": (
            (results.success_count / results.total_processed * 100)
            if results.total_processed > 0
            else 0
        ),
        "recipes_with_images": sum(
            1 for recipe in results.successful_recipes if recipe.images
        ),
        "total_extracted_images": sum(
            len(recipe.images) for recipe in results.successful_recipes if recipe.images
        ),
        "average_ingredients_per_recipe": (
            sum(len(recipe.ingredients) for recipe in results.successful_recipes)
            / len(results.successful_recipes)
            if results.successful_recipes
            else 0
        ),
        "average_steps_per_recipe": (
            sum(len(recipe.directions) for recipe in results.successful_recipes)
            / len(results.successful_recipes)
            if results.successful_recipes
            else 0
        ),
    }

    return stats
