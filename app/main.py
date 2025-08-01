import json
import os

# Import from the existing src modules
import sys
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# Register HEIF/HEIC support if available
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

sys.path.append(str(Path(__file__).parent.parent))

from src.extractors.image import ImageExtractor
from src.extractors.pdf import PDFExtractor
from src.extractors.recipe_image_extractor import RecipeImageExtractor
from src.extractors.web import WebExtractor
from src.formatter import RecipeFormatter
from src.llm_client import LLMClient
from src.models import Recipe, RecipeImage

# Load environment variables from .env file in the project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class StreamlitRecipeApp:
    """Main Streamlit application for AI Recipe Extraction."""

    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.setup_api_key()

    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="AI Recipe Extractor",
            page_icon="🍳",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if "extracted_recipes" not in st.session_state:
            st.session_state.extracted_recipes = []
        if "api_key_valid" not in st.session_state:
            st.session_state.api_key_valid = False
        if "current_recipe" not in st.session_state:
            st.session_state.current_recipe = None

    def setup_api_key(self):
        """Handle OpenAI API key configuration."""
        st.sidebar.header("🔑 OpenAI Configuration")

        # Check if API key is in environment
        env_api_key = os.getenv("OPENAI_API_KEY")

        if env_api_key:
            st.sidebar.success("✅ OpenAI API key found in environment")
            st.session_state.api_key_valid = True
            self.api_key = env_api_key
        else:
            st.sidebar.warning("⚠️ No OpenAI API key found in environment")

            # Input field for API key
            api_key_input = st.sidebar.text_input(
                "Enter your OpenAI API Key:",
                type="password",
                help="Your API key will not be stored permanently",
            )

            if api_key_input:
                self.api_key = api_key_input
                st.session_state.api_key_valid = True
                st.sidebar.success("✅ API key provided")
            else:
                st.session_state.api_key_valid = False
                st.sidebar.error("❌ Please provide an OpenAI API key to continue")

        # Show instructions if no valid API key
        if not st.session_state.api_key_valid:
            st.error(
                """
            🔑 **OpenAI API Key Required**

            To use this application, you need an OpenAI API key. You can:
            1. Set the `OPENAI_API_KEY` environment variable
            2. Enter your API key in the sidebar

            Get your API key from: https://platform.openai.com/api-keys
            """
            )
            st.stop()

    def show_header(self):
        """Display the main application header."""
        st.title("🍳 AI Recipe Extractor")
        st.markdown(
            """
        Extract structured recipes from images, web pages, and PDF files using AI.
        Choose a tab below to get started!
        """
        )

    def single_image_tab(self):
        """Handle single image extraction."""
        st.header("📸 Single Image Extraction")
        st.markdown("Upload a single image containing a recipe to extract it.")

        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "gif", "bmp", "webp", "heic", "heif"],
            help="Supported formats: JPG, PNG, GIF, BMP, WebP, HEIC, HEIF",
        )

        if uploaded_file is not None:
            # Display the uploaded image
            uploaded_file.seek(0)  # Reset file pointer
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

            col1, col2 = st.columns([1, 3])
            with col1:
                extract_button = st.button(
                    "🚀 Extract Recipe", type="primary", key="single_extract_button"
                )

            if extract_button:
                with st.spinner("🤖 Extracting recipe using AI..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=Path(uploaded_file.name).suffix
                        ) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        try:
                            # Process the image
                            extractor = ImageExtractor()
                            content = extractor.process_image(tmp_file_path)

                            # Extract recipe using LLM
                            llm_client = LLMClient(api_key=self.api_key)
                            recipe = llm_client.extract_recipe(
                                content, source=uploaded_file.name
                            )

                            st.session_state.current_recipe = recipe

                            # Save recipe to /recipes directory
                            formatter = RecipeFormatter()
                            recipe_dir = formatter.save_recipe(recipe)

                            # Extract recipe images if any
                            recipe_image_extractor = RecipeImageExtractor(llm_client)
                            image_metadata = (
                                recipe_image_extractor.extract_recipe_images(
                                    [tmp_file_path], recipe.name, recipe_dir
                                )
                            )

                            # Update recipe with image references
                            recipe.images = [
                                RecipeImage(
                                    filename=img["filename"],
                                    description=img.get("description", ""),
                                    is_main=img.get("is_main", False),
                                    is_step=img.get("is_step", False),
                                )
                                for img in image_metadata.get("extracted_images", [])
                            ]

                            # Update the saved files with image references
                            formatter.update_recipe_files(recipe, recipe_dir)

                            st.success(
                                f"✅ Recipe extracted and saved to: {recipe_dir}"
                            )
                            st.session_state.recipe_dir = recipe_dir

                        finally:
                            # Clean up temporary file
                            os.unlink(tmp_file_path)

                    except Exception as e:
                        st.error(f"❌ Error extracting recipe: {str(e)}")

        # Display extracted recipe if available
        if st.session_state.current_recipe:
            self.display_recipe(st.session_state.current_recipe)

    def batch_images_tab(self):
        """Handle batch image extraction."""
        st.header("📚 Batch Image Processing")

        # Clear explanation of the three modes
        st.info(
            """
        **Three processing modes available:**

        🔄 **Multiple Recipes Mode** - Each image contains a different recipe
        - Example: You have 5 photos from different cookbook pages, each showing a complete recipe
        - Result: 5 separate recipe files

        📄 **Single Multi-Page Recipe Mode** - All images are parts of ONE recipe
        - Example: One recipe spans across 3 pages/photos
        - Result: 1 combined recipe file

        📚 **Multiple Multi-Page Recipes Mode** - Group images into multiple recipes
        - Example: Recipe 1 uses images 1-2, Recipe 2 uses images 3-5, Recipe 3 uses image 6
        - Result: 3 separate recipe files from your groupings
        """
        )

        uploaded_files = st.file_uploader(
            "Choose image files",
            type=["jpg", "jpeg", "png", "gif", "bmp", "webp", "heic", "heif"],
            accept_multiple_files=True,
            help="Upload multiple images containing recipes",
        )

        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)} file(s) uploaded")

            # Show preview of uploaded images
            with st.expander("👁️ Preview uploaded images", expanded=True):
                # Sort files by name for consistent ordering
                sorted_files = sorted(
                    enumerate(uploaded_files), key=lambda x: x[1].name
                )
                cols = st.columns(min(len(sorted_files), 4))
                for idx, (original_idx, file) in enumerate(sorted_files):
                    with cols[idx % 4]:
                        try:
                            # Reset file pointer and read the image
                            file.seek(0)
                            img = Image.open(file)
                            st.image(
                                img,
                                caption=f"{original_idx+1}. {file.name}",
                                use_container_width=True,
                            )
                            file.seek(0)  # Reset for later use
                        except Exception as e:
                            st.error(f"Cannot preview {file.name}: {str(e)}")

            if len(uploaded_files) > 4:
                st.write(f"... and {len(uploaded_files) - 4} more files")

            # Processing mode selection
            st.subheader("Choose Processing Mode")
            processing_mode = st.radio(
                "How should these images be processed?",
                ["single_recipe", "multiple_recipes", "grouped_recipes"],
                index=0,  # Default to single_recipe (multiple pages per one recipe)
                format_func=lambda x: {
                    "multiple_recipes": "🔄 Multiple Recipes - Each image is a separate recipe",
                    "single_recipe": "📄 Single Multi-Page Recipe - All images are parts of ONE recipe",
                    "grouped_recipes": "📚 Multiple Multi-Page Recipes - Group images into multiple recipes",
                }[x],
                help="Select how to process your uploaded images",
            )

            # Show relevant example based on selection
            if processing_mode == "multiple_recipes":
                st.info(
                    "✅ Each image will be processed as a separate recipe. Perfect for photographing multiple recipe cards or different pages from a cookbook."
                )
            elif processing_mode == "single_recipe":
                st.info(
                    "✅ All images will be combined into one recipe. Perfect when a single recipe spans multiple pages/photos."
                )
            else:  # grouped_recipes
                st.info(
                    "✅ Group your images into multiple recipes. Perfect when you have several multi-page recipes to process at once."
                )

                # Image grouping interface
                st.subheader("Group Your Images")

                # Initialize session state for groups if not exists
                if "recipe_groups" not in st.session_state:
                    st.session_state.recipe_groups = []

                # Add new group button
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("➕ Add Recipe Group", type="secondary"):
                        st.session_state.recipe_groups.append([])

                # Display existing groups
                if st.session_state.recipe_groups:
                    # First, show unassigned images
                    all_assigned = []
                    for group in st.session_state.recipe_groups:
                        all_assigned.extend(group)
                    assigned_set = set(all_assigned)

                    unassigned = [
                        i for i in range(len(uploaded_files)) if i not in assigned_set
                    ]

                    if unassigned:
                        st.write("**Unassigned Images:**")
                        cols = st.columns(min(len(unassigned), 4))
                        for idx, img_idx in enumerate(unassigned):
                            with cols[idx % 4]:
                                st.write(
                                    f"{img_idx + 1}. {uploaded_files[img_idx].name}"
                                )

                    # Display each group
                    groups_to_remove = []

                    # Use a form to handle multiselect updates better
                    for group_idx in range(len(st.session_state.recipe_groups)):
                        with st.expander(
                            f"Recipe Group {group_idx + 1}", expanded=True
                        ):
                            # Calculate images assigned to OTHER groups
                            other_assigned = set()
                            for other_idx in range(len(st.session_state.recipe_groups)):
                                if other_idx != group_idx:
                                    other_assigned.update(
                                        st.session_state.recipe_groups[other_idx]
                                    )

                            # Current group selections
                            current_selections = st.session_state.recipe_groups[
                                group_idx
                            ]

                            # Available = all images NOT in other groups
                            available_indices = [
                                i
                                for i in range(len(uploaded_files))
                                if i not in other_assigned
                            ]

                            # Sort available indices by filename
                            available_indices.sort(key=lambda i: uploaded_files[i].name)

                            # Show available images with thumbnails
                            st.write("Select images for this recipe:")

                            # Show current selections
                            if current_selections:
                                selected_names = [
                                    f"{i+1}. {uploaded_files[i].name}"
                                    for i in current_selections
                                ]
                                st.info(f"Selected: {', '.join(selected_names)}")

                            # Create columns for image selection
                            num_cols = 4
                            rows_needed = (
                                len(available_indices) + num_cols - 1
                            ) // num_cols

                            for row in range(rows_needed):
                                cols = st.columns(num_cols)
                                for col_idx in range(num_cols):
                                    img_list_idx = row * num_cols + col_idx
                                    if img_list_idx < len(available_indices):
                                        img_idx = available_indices[img_list_idx]

                                        with cols[col_idx]:
                                            try:
                                                file = uploaded_files[img_idx]
                                                file.seek(0)
                                                img = Image.open(file)

                                                # Create a container for the image and selection state
                                                is_selected = (
                                                    img_idx in current_selections
                                                )

                                                # Show image with visual selection indicator
                                                if is_selected:
                                                    st.success(
                                                        f"✓ {img_idx + 1}. {file.name}"
                                                    )
                                                else:
                                                    st.caption(
                                                        f"{img_idx + 1}. {file.name}"
                                                    )

                                                st.image(
                                                    img,
                                                    use_container_width=True,
                                                )
                                                file.seek(0)

                                                # Toggle button
                                                button_label = (
                                                    "✓ Selected"
                                                    if is_selected
                                                    else "Select"
                                                )
                                                button_type = (
                                                    "secondary"
                                                    if is_selected
                                                    else "primary"
                                                )

                                                if st.button(
                                                    button_label,
                                                    key=f"btn_group_{group_idx}_img_{img_idx}",
                                                    type=button_type,
                                                    use_container_width=True,
                                                ):
                                                    # Toggle selection
                                                    if is_selected:
                                                        st.session_state.recipe_groups[
                                                            group_idx
                                                        ].remove(img_idx)
                                                    else:
                                                        st.session_state.recipe_groups[
                                                            group_idx
                                                        ].append(img_idx)
                                                    st.rerun()

                                            except Exception as e:
                                                st.error(f"Cannot preview: {str(e)}")

                            # Remove group button
                            if st.button(
                                f"🗑️ Remove Group {group_idx + 1}",
                                key=f"remove_{group_idx}",
                            ):
                                groups_to_remove.append(group_idx)

                    # Remove marked groups
                    for idx in reversed(groups_to_remove):
                        st.session_state.recipe_groups.pop(idx)
                        st.rerun()

                    # Validate groups
                    valid_groups = [g for g in st.session_state.recipe_groups if g]
                    if not valid_groups:
                        st.warning("Please add at least one recipe group with images.")

                    # Add recipe group button at the bottom too
                    st.markdown("---")
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button(
                            "➕ Add Recipe Group",
                            type="secondary",
                            key="add_recipe_bottom",
                        ):
                            st.session_state.recipe_groups.append([])
                            st.rerun()
                else:
                    st.info(
                        "Click '➕ Add Recipe Group' to start grouping your images."
                    )

            col1, col2 = st.columns([1, 3])
            with col1:
                if processing_mode == "multiple_recipes":
                    button_text = "🚀 Extract Recipes"
                elif processing_mode == "single_recipe":
                    button_text = "🚀 Combine & Extract"
                else:  # grouped_recipes
                    button_text = "🚀 Extract Grouped Recipes"

                # Disable button if in grouped mode and no valid groups
                disabled = False
                if processing_mode == "grouped_recipes":
                    valid_groups = [
                        g for g in st.session_state.get("recipe_groups", []) if g
                    ]
                    disabled = not valid_groups

                extract_button = st.button(
                    button_text, type="primary", disabled=disabled
                )

            if extract_button:
                progress_bar = st.progress(0)
                status_text = st.empty()

                extracted_recipes = []

                if processing_mode == "single_recipe":
                    # Process all images as one combined recipe
                    status_text.text("Combining all images into one recipe...")

                    try:
                        # Save all uploaded files temporarily
                        temp_paths = []
                        status_text.text("Preparing images...")
                        for idx, uploaded_file in enumerate(uploaded_files):
                            progress_bar.progress(
                                (idx / len(uploaded_files)) * 0.2
                            )  # 0-20% for file prep
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=Path(uploaded_file.name).suffix
                            ) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                temp_paths.append(tmp_file.name)

                        try:
                            # Define progress callback for image processing
                            def image_progress_callback(current, total, message):
                                # Image processing takes 20-50% of the progress
                                progress = 0.2 + (current / total) * 0.3
                                progress_bar.progress(progress)
                                status_text.text(message)

                            # Process all images together
                            extractor = ImageExtractor()
                            content = extractor.process_multiple_images(
                                temp_paths, progress_callback=image_progress_callback
                            )

                            # Extract recipe using LLM (50-70%)
                            status_text.text(
                                "Extracting recipe from combined images..."
                            )
                            progress_bar.progress(0.5)

                            llm_client = LLMClient(api_key=self.api_key)
                            source_names = ", ".join([f.name for f in uploaded_files])
                            recipe = llm_client.extract_recipe(
                                content, source=f"Combined from: {source_names}"
                            )

                            progress_bar.progress(0.7)

                            # Save combined recipe to /recipes directory (70-75%)
                            status_text.text("Saving recipe...")
                            formatter = RecipeFormatter()
                            recipe_dir = formatter.save_recipe(recipe)
                            progress_bar.progress(0.75)

                            # Extract recipe images if multiple pages (75-95%)
                            if len(temp_paths) > 1:
                                status_text.text(
                                    "Extracting and organizing recipe images..."
                                )
                                progress_bar.progress(0.8)

                                # Define progress callback for image extraction
                                def extraction_progress_callback(
                                    current, total, message
                                ):
                                    # Image extraction takes 80-95% of the progress
                                    progress = 0.8 + (current / total) * 0.15
                                    progress_bar.progress(progress)
                                    status_text.text(message)

                                recipe_image_extractor = RecipeImageExtractor(
                                    llm_client
                                )
                                image_metadata = (
                                    recipe_image_extractor.extract_recipe_images(
                                        temp_paths,
                                        recipe.name,
                                        recipe_dir,
                                        progress_callback=extraction_progress_callback,
                                    )
                                )
                                progress_bar.progress(0.95)

                                # Update recipe with image references
                                recipe.images = [
                                    RecipeImage(
                                        filename=img["filename"],
                                        description=img.get("description", ""),
                                        is_main=img.get("is_main", False),
                                        is_step=img.get("is_step", False),
                                    )
                                    for img in image_metadata.get(
                                        "extracted_images", []
                                    )
                                ]

                                # Update the saved files with image references
                                formatter.update_recipe_files(recipe, recipe_dir)

                            extracted_recipes.append(recipe)

                        finally:
                            # Clean up temporary files
                            for temp_path in temp_paths:
                                os.unlink(temp_path)

                    except Exception as e:
                        st.error(f"❌ Failed to extract combined recipe: {str(e)}")

                    progress_bar.progress(1.0)

                elif processing_mode == "grouped_recipes":
                    # Process grouped recipes
                    valid_groups = [g for g in st.session_state.recipe_groups if g]
                    total_groups = len(valid_groups)

                    for group_idx, image_indices in enumerate(valid_groups):
                        status_text.text(
                            f"Processing recipe group {group_idx + 1} of {total_groups}..."
                        )
                        base_progress = group_idx / total_groups

                        try:
                            # Get the files for this group
                            group_files = [uploaded_files[idx] for idx in image_indices]

                            # Save files temporarily
                            temp_paths = []
                            for file_idx, uploaded_file in enumerate(group_files):
                                progress = base_progress + (
                                    file_idx / len(group_files)
                                ) * (0.3 / total_groups)
                                progress_bar.progress(progress)

                                with tempfile.NamedTemporaryFile(
                                    delete=False, suffix=Path(uploaded_file.name).suffix
                                ) as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    temp_paths.append(tmp_file.name)

                            try:
                                # Process images
                                extractor = ImageExtractor()
                                if len(temp_paths) > 1:
                                    content = extractor.process_multiple_images(
                                        temp_paths
                                    )
                                else:
                                    content = extractor.process_image(temp_paths[0])

                                # Extract recipe
                                progress = base_progress + 0.5 / total_groups
                                progress_bar.progress(progress)
                                status_text.text(
                                    f"Extracting recipe {group_idx + 1} of {total_groups}..."
                                )

                                llm_client = LLMClient(api_key=self.api_key)
                                source_names = ", ".join([f.name for f in group_files])
                                recipe = llm_client.extract_recipe(
                                    content,
                                    source=f"Group {group_idx + 1}: {source_names}",
                                )

                                # Save recipe
                                formatter = RecipeFormatter()
                                recipe_dir = formatter.save_recipe(recipe)

                                # Extract images
                                if len(temp_paths) > 1:
                                    recipe_image_extractor = RecipeImageExtractor(
                                        llm_client
                                    )
                                    image_metadata = (
                                        recipe_image_extractor.extract_recipe_images(
                                            temp_paths, recipe.name, recipe_dir
                                        )
                                    )

                                    # Update recipe with image references
                                    recipe.images = [
                                        RecipeImage(
                                            filename=img["filename"],
                                            description=img.get("description", ""),
                                            is_main=img.get("is_main", False),
                                            is_step=img.get("is_step", False),
                                        )
                                        for img in image_metadata.get(
                                            "extracted_images", []
                                        )
                                    ]

                                    # Update saved files
                                    formatter.update_recipe_files(recipe, recipe_dir)

                                extracted_recipes.append(recipe)

                            finally:
                                # Clean up temp files
                                for temp_path in temp_paths:
                                    os.unlink(temp_path)

                        except Exception as e:
                            st.error(
                                f"❌ Failed to extract recipe group {group_idx + 1}: {str(e)}"
                            )

                    progress_bar.progress(1.0)

                else:
                    # Process each image as a separate recipe
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {uploaded_file.name}...")
                        progress_bar.progress((i + 1) / len(uploaded_files))

                        try:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=Path(uploaded_file.name).suffix
                            ) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name

                            try:
                                # Process the image
                                extractor = ImageExtractor()
                                content = extractor.process_image(tmp_file_path)

                                # Extract recipe using LLM
                                llm_client = LLMClient(api_key=self.api_key)
                                recipe = llm_client.extract_recipe(
                                    content, source=uploaded_file.name
                                )

                                # Save individual recipe to /recipes directory
                                formatter = RecipeFormatter()
                                recipe_dir = formatter.save_recipe(recipe)

                                # Extract recipe images
                                recipe_image_extractor = RecipeImageExtractor(
                                    llm_client
                                )
                                image_metadata = (
                                    recipe_image_extractor.extract_recipe_images(
                                        [tmp_file_path], recipe.name, recipe_dir
                                    )
                                )

                                # Update recipe with image references
                                recipe.images = [
                                    RecipeImage(
                                        filename=img["filename"],
                                        description=img.get("description", ""),
                                        is_main=img.get("is_main", False),
                                        is_step=img.get("is_step", False),
                                    )
                                    for img in image_metadata.get(
                                        "extracted_images", []
                                    )
                                ]

                                # Update the saved files with image references
                                formatter.update_recipe_files(recipe, recipe_dir)

                                extracted_recipes.append(recipe)

                            finally:
                                # Clean up temporary file
                                os.unlink(tmp_file_path)

                        except Exception as e:
                            st.warning(
                                f"⚠️ Failed to extract recipe from {uploaded_file.name}: {str(e)}"
                            )
                            continue

                    progress_bar.progress(1.0)

                status_text.text("✅ Processing complete!")
                st.session_state.extracted_recipes = extracted_recipes

                if extracted_recipes:
                    if processing_mode == "single_recipe":
                        st.success(
                            f"🎉 Successfully combined {len(uploaded_files)} images into 1 recipe!"
                        )
                    elif processing_mode == "grouped_recipes":
                        st.success(
                            f"🎉 Successfully extracted {len(extracted_recipes)} recipes from {len(valid_groups)} groups!"
                        )
                    else:
                        st.success(
                            f"🎉 Successfully extracted {len(extracted_recipes)} recipe(s)!"
                        )
                else:
                    st.error(
                        "❌ No recipes could be extracted from the uploaded images."
                    )

        # Display extracted recipes
        if st.session_state.extracted_recipes:
            self.display_batch_recipes(st.session_state.extracted_recipes)

            # Reset button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(
                    "🔄 Start New Extraction", type="primary", use_container_width=True
                ):
                    # Clear the extracted recipes
                    st.session_state.extracted_recipes = []
                    # Clear recipe groups if they exist
                    if "recipe_groups" in st.session_state:
                        st.session_state.recipe_groups = []
                    st.rerun()

    def web_url_tab(self):
        """Handle web URL extraction."""
        st.header("🌐 Web URL Extraction")
        st.markdown("Enter a URL to extract a recipe from a web page.")

        url_input = st.text_input(
            "Enter the URL:",
            placeholder="https://example.com/recipe",
            help="Enter the full URL of the webpage containing the recipe",
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            extract_button = st.button(
                "🚀 Extract Recipe",
                type="primary",
                disabled=not url_input,
                key="web_extract_button",
            )

        if extract_button and url_input:
            with st.spinner("🌐 Fetching and extracting recipe from web page..."):
                try:
                    # Extract content from web page
                    web_extractor = WebExtractor()
                    content = web_extractor.extract_from_url(url_input)

                    # Extract recipe using LLM
                    llm_client = LLMClient(api_key=self.api_key)
                    recipe = llm_client.extract_recipe(content, source=url_input)

                    st.session_state.current_recipe = recipe

                    # Save recipe to /recipes directory
                    formatter = RecipeFormatter()
                    recipe_dir = formatter.save_recipe(recipe)

                    st.success(f"✅ Recipe extracted and saved to: {recipe_dir}")
                    st.session_state.recipe_dir = recipe_dir

                except Exception as e:
                    st.error(f"❌ Error extracting recipe: {str(e)}")

        # Display extracted recipe if available
        if st.session_state.current_recipe:
            self.display_recipe(st.session_state.current_recipe)

    def pdf_upload_tab(self):
        """Handle PDF file extraction."""
        st.header("📄 PDF File Extraction")
        st.markdown("Upload a PDF file to extract recipes from it.")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF file containing recipes",
        )

        if uploaded_file is not None:
            st.success(f"📄 Uploaded: {uploaded_file.name}")

            # Optional page selection
            page_selection = st.selectbox(
                "Page selection:",
                ["All pages", "Specific pages"],
                help="Choose whether to extract from all pages or specify page numbers",
            )

            page_numbers = None
            if page_selection == "Specific pages":
                pages_input = st.text_input(
                    "Enter page numbers (comma-separated, 1-indexed):",
                    placeholder="1, 3, 5-7",
                    help="Example: '1, 3, 5-7' will extract pages 1, 3, 5, 6, and 7",
                )

                if pages_input:
                    try:
                        page_numbers = self.parse_page_numbers(pages_input)
                        st.info(f"📄 Will extract from pages: {page_numbers}")
                    except ValueError as e:
                        st.error(f"❌ Invalid page format: {str(e)}")
                        page_numbers = None

            col1, col2 = st.columns([1, 3])
            with col1:
                extract_button = st.button(
                    "🚀 Extract Recipe", type="primary", key="pdf_extract_button"
                )

            if extract_button:
                with st.spinner(
                    "📄 Extracting text from PDF and processing with AI..."
                ):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        try:
                            # Extract content from PDF
                            pdf_extractor = PDFExtractor()
                            # Convert 1-indexed to 0-indexed for the extractor
                            zero_indexed_pages = (
                                [p - 1 for p in page_numbers] if page_numbers else None
                            )
                            content = pdf_extractor.extract_from_pdf(
                                tmp_file_path, zero_indexed_pages
                            )

                            # Extract recipe using LLM
                            llm_client = LLMClient(api_key=self.api_key)
                            recipe = llm_client.extract_recipe(
                                content, source=uploaded_file.name
                            )

                            st.session_state.current_recipe = recipe

                            # Save recipe to /recipes directory
                            formatter = RecipeFormatter()
                            recipe_dir = formatter.save_recipe(recipe)

                            st.success(
                                f"✅ Recipe extracted and saved to: {recipe_dir}"
                            )
                            st.session_state.recipe_dir = recipe_dir

                        finally:
                            # Clean up temporary file
                            os.unlink(tmp_file_path)

                    except Exception as e:
                        st.error(f"❌ Error extracting recipe: {str(e)}")

        # Display extracted recipe if available
        if st.session_state.current_recipe:
            self.display_recipe(st.session_state.current_recipe)

    def parse_page_numbers(self, pages_input: str) -> List[int]:
        """Parse page numbers from user input."""
        page_numbers = []
        parts = [part.strip() for part in pages_input.split(",")]

        for part in parts:
            if "-" in part:
                # Range like "5-7"
                start, end = part.split("-", 1)
                start, end = int(start.strip()), int(end.strip())
                page_numbers.extend(range(start, end + 1))
            else:
                # Single page
                page_numbers.append(int(part))

        return sorted(list(set(page_numbers)))  # Remove duplicates and sort

    def display_recipe(self, recipe: Recipe):
        """Display a single extracted recipe."""
        st.divider()
        st.subheader("🍽️ Extracted Recipe")

        # Recipe name and basic info
        st.markdown(f"### {recipe.name}")

        if recipe.description:
            st.markdown(f"**Description:** {recipe.description}")

        # Recipe metadata in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            if recipe.servings:
                st.metric("Servings", recipe.servings)
        with col2:
            if recipe.total_time:
                st.metric("Total Time", recipe.total_time)
        with col3:
            if recipe.source:
                st.metric("Source", recipe.source)

        # Ingredients
        if recipe.ingredients:
            st.markdown("#### 🥘 Ingredients")
            for ingredient in recipe.ingredients:
                st.markdown(f"• {ingredient.to_string()}")

        # Directions
        if recipe.directions:
            st.markdown("#### 👩‍🍳 Directions")
            for i, direction in enumerate(recipe.directions, 1):
                st.markdown(f"{i}. {direction}")

        # Notes
        if recipe.notes:
            st.markdown("#### 📝 Notes")
            for note in recipe.notes:
                st.markdown(f"• {note}")

        # Download options
        st.markdown("#### 💾 Download Options")
        col1, col2 = st.columns(2)

        with col1:
            # Text format download
            text_content = recipe.to_text()
            st.download_button(
                label="📄 Download as Text",
                data=text_content,
                file_name=f"{recipe.name.replace(' ', '_').lower()}.txt",
                mime="text/plain",
            )

        with col2:
            # JSON format download
            json_content = json.dumps(recipe.model_dump(), indent=2)
            st.download_button(
                label="📊 Download as JSON",
                data=json_content,
                file_name=f"{recipe.name.replace(' ', '_').lower()}.json",
                mime="application/json",
            )

    def display_batch_recipes(self, recipes: List[Recipe]):
        """Display multiple extracted recipes."""
        st.divider()
        st.subheader(f"🍽️ Extracted Recipes ({len(recipes)})")

        # Recipe selector
        recipe_names = [f"{i+1}. {recipe.name}" for i, recipe in enumerate(recipes)]
        selected_recipe_index = st.selectbox(
            "Select a recipe to view:",
            range(len(recipes)),
            format_func=lambda x: recipe_names[x],
        )

        # Display selected recipe
        if selected_recipe_index is not None:
            selected_recipe = recipes[selected_recipe_index]
            self.display_recipe(selected_recipe)

        # Bulk download options
        st.markdown("#### 💾 Bulk Download Options")
        col1, col2 = st.columns(2)

        with col1:
            # Download all as combined text file
            combined_text = ""
            for i, recipe in enumerate(recipes, 1):
                if i > 1:
                    combined_text += "\n\n" + "=" * 80 + "\n\n"
                combined_text += f"RECIPE {i} OF {len(recipes)}\n\n"
                combined_text += recipe.to_text()

            st.download_button(
                label="📄 Download All as Text",
                data=combined_text,
                file_name="extracted_recipes.txt",
                mime="text/plain",
            )

        with col2:
            # Download all as JSON array
            json_content = json.dumps(
                [recipe.model_dump() for recipe in recipes], indent=2
            )
            st.download_button(
                label="📊 Download All as JSON",
                data=json_content,
                file_name="extracted_recipes.json",
                mime="application/json",
            )

    def show_sidebar_info(self):
        """Display information in the sidebar."""
        st.sidebar.markdown("---")
        st.sidebar.header("ℹ️ About")
        st.sidebar.markdown(
            """
        This app uses OpenAI's GPT-4 Vision to extract structured recipe data from:

        • 📸 **Images** (JPG, PNG, HEIC, etc.)
        • 🌐 **Web pages** (recipe websites)
        • 📄 **PDF files** (cookbooks, printouts)

        **Features:**
        - Structured ingredient parsing
        - Step-by-step directions
        - Recipe metadata extraction
        - Multiple output formats (TXT, JSON)
        - Batch processing support
        """
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🛠️ Tips")
        st.sidebar.markdown(
            """
        - **Images**: Ensure text is clear and well-lit
        - **Web URLs**: Use direct recipe page links
        - **PDFs**: Text-based PDFs work best
        - **Batch**: Process similar content together
        """
        )

    def run(self):
        """Run the main Streamlit application."""
        if not st.session_state.api_key_valid:
            return

        self.show_header()
        self.show_sidebar_info()

        # Create tabs for different extraction methods
        # Put Batch Images first so it's the default tab
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📚 Batch Images", "📸 Single Image", "🌐 Web URL", "📄 PDF Upload"]
        )

        with tab1:
            self.batch_images_tab()

        with tab2:
            self.single_image_tab()

        with tab3:
            self.web_url_tab()

        with tab4:
            self.pdf_upload_tab()


def main():
    """Main entry point for the Streamlit app."""
    app = StreamlitRecipeApp()
    app.run()


if __name__ == "__main__":
    main()
