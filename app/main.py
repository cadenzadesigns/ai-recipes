import base64
import io
import json
import os

# Import from the existing src modules
import sys
import tempfile
from io import BytesIO
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

from app.components.image_cropper import StreamlitImageCropper
from app.recipe_html_generator import RecipeHTMLGenerator

# Import recipe utilities
from app.recipe_utils import ensure_recipe_htmls_exist
from src.extractors.image import ImageExtractor
from src.extractors.pdf import PDFExtractor
from src.extractors.pdf_image_extractor import PDFImageExtractor
from src.extractors.recipe_image_extractor import RecipeImageExtractor
from src.extractors.web import WebExtractor
from src.extractors.web_image_extractor import WebImageExtractor
from src.formatter import RecipeFormatter
from src.llm_client import LLMClient
from src.models import Recipe, RecipeImage

# Load environment variables from .env file if it exists (local development)
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


@st.cache_data
def create_thumbnail(
    file_content: bytes, filename: str, size: tuple = (400, 400)
) -> Image.Image:
    """Create and cache a thumbnail from file content."""
    img = Image.open(io.BytesIO(file_content))
    thumbnail = img.copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    return thumbnail


@st.cache_data
def create_thumbnail_base64(
    file_content: bytes, filename: str, size: tuple = (150, 150)
) -> str:
    """Create a base64-encoded thumbnail from file content."""
    img = Image.open(io.BytesIO(file_content))
    thumbnail = img.copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)

    # Convert to base64
    buffered = io.BytesIO()
    thumbnail.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/png;base64,{img_base64}"


class StreamlitRecipeApp:
    """Main Streamlit application for AI Recipe Extraction."""

    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.setup_api_key()
        self.html_generator = RecipeHTMLGenerator()

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
        if "crop_regions" not in st.session_state:
            st.session_state.crop_regions = {}

    def save_recipe_with_html(self, recipe: Recipe, formatter: RecipeFormatter) -> str:
        """Save recipe and generate HTML page."""
        recipe_dir = formatter.save_recipe(recipe)
        self.html_generator.save_recipe_html(recipe, Path(recipe_dir))
        return recipe_dir

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

            col1, col2 = st.columns([1, 2])
            with col1:
                extract_button = st.button(
                    "🚀 Extract Recipe", type="primary", key="single_extract_button"
                )

            # Manual cropping is always enabled
            manual_crop = True

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
                            recipe_dir = self.save_recipe_with_html(recipe, formatter)

                            # Extract recipe images
                            if manual_crop:
                                # Manual cropping workflow
                                st.success(
                                    "Recipe text extracted! Now let's crop the images."
                                )

                                # Load image for cropping
                                pil_image = Image.open(tmp_file_path)

                                # Initialize cropper
                                cropper = StreamlitImageCropper()

                                # Single image cropping
                                st.write("### Manual Image Cropping")
                                st.write(
                                    "Define regions for recipe images you want to extract."
                                )

                                crop_regions = cropper.crop_single_image_canvas(
                                    pil_image, "single_image", max_crops=5
                                )

                                if st.button(
                                    "💾 Save Cropped Images", key="save_crops"
                                ):
                                    if crop_regions:
                                        # Save cropped images
                                        image_metadata = cropper.save_cropped_images(
                                            {tmp_file_path: pil_image},
                                            {tmp_file_path: crop_regions},
                                            recipe.name,
                                            Path(recipe_dir),
                                        )
                                        st.success(
                                            f"✅ Saved {len(image_metadata['extracted_images'])} cropped images!"
                                        )
                                    else:
                                        image_metadata = {"extracted_images": []}
                                        st.info(
                                            "No regions defined. Skipping image extraction."
                                        )
                            else:
                                # Automatic extraction
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

        # Clear explanation of the unified grouping system
        st.info(
            """
        **How it works:**

        📚 **Smart Recipe Grouping** - Organize your images into recipe groups
        - Each group becomes one complete recipe
        - Handles all scenarios automatically:
          • **Single-page recipes**: 1 image = 1 group = 1 recipe
          • **Multi-page recipes**: Multiple images in 1 group = 1 combined recipe
          • **Multiple recipes**: Multiple groups = Multiple separate recipes

        📸 **Manual Image Cropping** - Select the exact recipe content
        - Draw bounding boxes around recipe images
        - Mark images as main recipe photo or step-by-step photos
        - Skip pages without recipe content
        - Finish early when you've cropped what you need
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

            # Initialize deleted files tracking
            if "deleted_files" not in st.session_state:
                st.session_state.deleted_files = set()

            # Filter out deleted files
            active_files = [
                (idx, file)
                for idx, file in enumerate(uploaded_files)
                if idx not in st.session_state.deleted_files
            ]

            # Show preview of uploaded images
            with st.expander("👁️ Preview uploaded images", expanded=True):
                if active_files:
                    st.info(
                        f"📸 {len(active_files)} active file(s) | 🗑️ {len(st.session_state.deleted_files)} deleted"
                    )

                    # Sort files by name for consistent ordering
                    sorted_files = sorted(active_files, key=lambda x: x[1].name)
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

                                # Delete button for each image
                                if st.button(
                                    "🗑️ Delete",
                                    key=f"delete_{original_idx}",
                                    help=f"Remove {file.name}",
                                ):
                                    st.session_state.deleted_files.add(original_idx)
                                    # Reset saved assignments when files change
                                    if "saved_assignments" in st.session_state:
                                        del st.session_state.saved_assignments
                                    if "groups_saved_once" in st.session_state:
                                        del st.session_state.groups_saved_once
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Cannot preview {file.name}: {str(e)}")

                    if len(active_files) > 4:
                        st.write(f"... and {len(active_files) - 4} more files")

                    # Option to restore deleted files
                    if st.session_state.deleted_files:
                        if st.button("♻️ Restore All Deleted Files", key="restore_all"):
                            st.session_state.deleted_files.clear()
                            # Reset saved assignments when files change
                            if "saved_assignments" in st.session_state:
                                del st.session_state.saved_assignments
                            if "groups_saved_once" in st.session_state:
                                del st.session_state.groups_saved_once
                            st.rerun()
                else:
                    st.warning(
                        "All files have been deleted. Please restore files or upload new ones."
                    )

            # Always use grouped mode - the grouping interface handles all cases
            processing_mode = "grouped_recipes"

            # Image grouping interface
            st.subheader("📊 Group Your Images into Recipes")
            st.info(
                "✅ Organize your images by recipe. Each group becomes one recipe. "
                "Create single-page recipes (1 image per group) or multi-page recipes (multiple images per group)."
            )

            # Initialize saved assignments if not exists
            if "saved_assignments" not in st.session_state:
                # Initialize all active images to Recipe 1
                st.session_state.saved_assignments = {
                    i: 1
                    for i in range(len(uploaded_files))
                    if i not in st.session_state.deleted_files
                }
                st.session_state.num_groups = 1

            # Group controls (outside form)
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                # Use a form to prevent immediate updates
                with st.form("num_groups_form"):
                    num_groups_input = st.number_input(
                        "Number of recipe groups",
                        min_value=1,
                        max_value=min(len(uploaded_files), 10),
                        value=st.session_state.get("num_groups", 1),
                        help="How many different recipes do you want to create?",
                    )

                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        save_groups_btn = st.form_submit_button(
                            "📊 Update Groups",
                            type="secondary",
                            use_container_width=True,
                        )

                    if save_groups_btn:
                        # Only update when button is clicked
                        old_num_groups = st.session_state.get("num_groups", 1)
                        st.session_state.num_groups = num_groups_input

                        # If reducing groups, reassign higher group numbers to group 1
                        if num_groups_input < old_num_groups:
                            for idx in st.session_state.saved_assignments:
                                if (
                                    st.session_state.saved_assignments[idx]
                                    > num_groups_input
                                ):
                                    st.session_state.saved_assignments[idx] = 1

                        st.rerun()

                # Use the saved value for display
                num_groups = st.session_state.get("num_groups", 1)

            with col2:
                if st.button("🔄 Reset All", type="secondary"):
                    st.session_state.saved_assignments = {
                        i: 1 for i in range(len(uploaded_files))
                    }
                    st.session_state.num_groups = 1
                    st.rerun()

            # Define colors for each recipe group
            group_colors = [
                "#FF6B6B",  # Red
                "#4ECDC4",  # Teal
                "#45B7D1",  # Blue
                "#96CEB4",  # Green
                "#DDA0DD",  # Plum
                "#F4A460",  # Sandy brown
                "#98D8C8",  # Mint
                "#FFD93D",  # Yellow
                "#C7CEEA",  # Lavender
                "#FFAAA5",  # Pink
            ]

            # Show color legend
            st.markdown("### Recipe Groups")
            st.caption("Select images below, then click Save Groups when done")
            legend_cols = st.columns(min(num_groups, 5))
            for i in range(num_groups):
                with legend_cols[i % len(legend_cols)]:
                    color = group_colors[i % len(group_colors)]
                    st.markdown(
                        f'<div style="background-color: {color}; padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;">Recipe {i+1}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            # Create thumbnail cache if not exists
            if "thumbnail_cache" not in st.session_state:
                st.session_state.thumbnail_cache = {}

            # Sort files by filename, excluding deleted files
            sorted_indices = sorted(
                [
                    i
                    for i in range(len(uploaded_files))
                    if i not in st.session_state.deleted_files
                ],
                key=lambda i: uploaded_files[i].name.lower(),
            )

            # Use a form to prevent reruns on each radio button change
            with st.form("image_grouping_form"):
                st.markdown("### Your Images")

                # Track selections in the form
                form_selections = {}

                # Display images in rows of 4
                images_per_row = 4
                num_rows = (len(sorted_indices) + images_per_row - 1) // images_per_row

                for row in range(num_rows):
                    cols = st.columns(images_per_row)

                    for col_idx in range(images_per_row):
                        list_idx = row * images_per_row + col_idx

                        if list_idx < len(sorted_indices):
                            img_idx = sorted_indices[list_idx]

                            with cols[col_idx]:
                                file = uploaded_files[img_idx]

                                # Get or create thumbnail
                                if img_idx not in st.session_state.thumbnail_cache:
                                    file.seek(0)
                                    thumbnail = create_thumbnail(
                                        file.read(), file.name, size=(300, 300)
                                    )
                                    st.session_state.thumbnail_cache[img_idx] = (
                                        thumbnail
                                    )
                                    file.seek(0)
                                else:
                                    thumbnail = st.session_state.thumbnail_cache[
                                        img_idx
                                    ]

                                # Get current assignment from saved assignments
                                saved_group = st.session_state.saved_assignments.get(
                                    img_idx, 1
                                )

                                # During form editing, we don't show colored borders
                                # Only show gray borders to indicate the image container

                                # Convert image to base64 for inline display
                                buffered = BytesIO()
                                thumbnail.save(buffered, format="PNG")
                                img_str = base64.b64encode(buffered.getvalue()).decode()

                                # Display image with gray border
                                st.markdown(
                                    f"""
                                    <div style="
                                        border: 3px solid #E0E0E0;
                                        border-radius: 8px;
                                        padding: 4px;
                                        background-color: white;
                                        margin-bottom: 8px;
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                    ">
                                        <img src="data:image/png;base64,{img_str}" style="
                                            width: 100%;
                                            border-radius: 4px;
                                            display: block;
                                        ">
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                # Display filename
                                st.caption(
                                    f"{file.name[:20]}..."
                                    if len(file.name) > 20
                                    else file.name
                                )

                                # Recipe assignment radio buttons
                                selected_group = st.radio(
                                    "Recipe",
                                    options=list(range(1, num_groups + 1)),
                                    index=saved_group - 1,
                                    key=f"form_radio_{img_idx}",
                                    format_func=lambda x: f"R{x}",
                                    horizontal=True,
                                    label_visibility="collapsed",
                                )

                                # Store the selection
                                form_selections[img_idx] = selected_group

                # Form submit button
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    submitted = st.form_submit_button(
                        "💾 Save Groups",
                        type="primary",
                        use_container_width=True,
                    )

                if submitted:
                    # Update saved assignments with form selections
                    for img_idx, group in form_selections.items():
                        st.session_state.saved_assignments[img_idx] = group
                    # Mark that groups have been saved at least once
                    st.session_state.groups_saved_once = True
                    st.success("✅ Groups saved!")
                    st.rerun()

            # Show summary (outside form) - only after groups have been saved at least once
            if st.session_state.get("groups_saved_once", False):
                st.markdown("---")
                st.markdown("### Saved Groups")

                # Count images per group (from saved assignments)
                group_counts = {}
                for group_num in range(1, num_groups + 1):
                    count = sum(
                        1
                        for v in st.session_state.saved_assignments.values()
                        if v == group_num
                    )
                    group_counts[group_num] = count

                summary_cols = st.columns(min(num_groups, 4))
                for i, (group_num, count) in enumerate(group_counts.items()):
                    with summary_cols[i % len(summary_cols)]:
                        color = group_colors[(group_num - 1) % len(group_colors)]
                        st.markdown(
                            f'<div style="background-color: {color}; padding: 20px; border-radius: 8px; text-align: center; color: white;">'
                            f'<h3 style="margin: 0;">Recipe {group_num}</h3>'
                            f'<h1 style="margin: 0;">{count}</h1>'
                            f'<p style="margin: 0;">images</p>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                # Show saved assignments with colored borders
                st.markdown("---")
                st.markdown("### Saved Assignments")
                st.caption("Images are shown with their assigned recipe group colors")

                # Display images in rows of 6 for the summary view
                summary_images_per_row = 6
                num_summary_rows = (
                    len(sorted_indices) + summary_images_per_row - 1
                ) // summary_images_per_row

                for row in range(num_summary_rows):
                    cols = st.columns(summary_images_per_row)

                    for col_idx in range(summary_images_per_row):
                        list_idx = row * summary_images_per_row + col_idx

                        if list_idx < len(sorted_indices):
                            img_idx = sorted_indices[list_idx]

                            with cols[col_idx]:
                                file = uploaded_files[img_idx]

                                # Get thumbnail from cache
                                thumbnail = st.session_state.thumbnail_cache.get(
                                    img_idx
                                )
                                if thumbnail:
                                    # Get saved group assignment
                                    saved_group = (
                                        st.session_state.saved_assignments.get(
                                            img_idx, 1
                                        )
                                    )
                                    group_color = group_colors[
                                        (saved_group - 1) % len(group_colors)
                                    ]

                                    # Convert image to base64
                                    buffered = BytesIO()
                                    thumbnail.save(buffered, format="PNG")
                                    img_str = base64.b64encode(
                                        buffered.getvalue()
                                    ).decode()

                                    # Display image with colored border
                                    st.markdown(
                                        f"""
                                        <div style="
                                            border: 3px solid {group_color};
                                            border-radius: 8px;
                                            padding: 3px;
                                            background-color: white;
                                            margin-bottom: 4px;
                                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                        ">
                                            <img src="data:image/png;base64,{img_str}" style="
                                                width: 100%;
                                                border-radius: 4px;
                                                display: block;
                                            ">
                                        </div>
                                        <p style="text-align: center; font-size: 11px; margin: 2px 0; color: {group_color}; font-weight: bold;">R{saved_group}</p>
                                        """,
                                        unsafe_allow_html=True,
                                    )

            # Create recipe_groups list from saved assignments, excluding deleted files
            st.session_state.recipe_groups = []
            for group_num in range(1, num_groups + 1):
                group_indices = [
                    idx
                    for idx, g in st.session_state.saved_assignments.items()
                    if g == group_num and idx not in st.session_state.deleted_files
                ]
                if group_indices:
                    st.session_state.recipe_groups.append(group_indices)

            # Manual cropping is always enabled
            st.markdown("---")
            st.subheader("🖼️ Image Processing")
            st.info(
                "✨ You'll manually select recipe images using an interactive cropping interface before text extraction."
            )

            # Initialize extract_button to avoid UnboundLocalError
            extract_button = False

            # Always use manual cropping workflow
            manual_crop = (
                True  # Keep this for backward compatibility with existing code
            )

            # Determine the workflow based on cropping completion
            if not st.session_state.get("crop_step_completed", False):
                # Step 1: Manual Cropping (if enabled and not completed)
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    crop_button_text = "🖼️ Start Manual Cropping"

                    # Disable button if in grouped mode and no valid groups
                    disabled = False
                    if processing_mode == "grouped_recipes":
                        valid_groups = [
                            g for g in st.session_state.get("recipe_groups", []) if g
                        ]
                        disabled = not valid_groups

                    crop_button = st.button(
                        crop_button_text, type="primary", disabled=disabled
                    )

                with col2:
                    # Skip cropping button for pre-cropped photos
                    skip_button = st.button(
                        "⏭️ Skip Cropping",
                        type="secondary",
                        disabled=disabled,
                        help="Skip cropping if your photos are already cropped",
                    )

                if crop_button:
                    st.session_state.crop_step_active = True
                    st.rerun()

                if skip_button:
                    st.session_state.crop_step_completed = True
                    st.session_state.crop_step_active = False
                    st.info(
                        "✅ Skipping manual cropping - using full images for extraction"
                    )
                    st.rerun()

            else:
                # Step 2: Extract Recipes (either manual crop completed or no manual crop)
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

            # Manual cropping interface (appears when crop_step_active is True)
            if manual_crop and st.session_state.get("crop_step_active", False):
                st.markdown("---")
                st.subheader("🖼️ Manual Image Cropping")

                # Show which recipe groups we'll be cropping
                if processing_mode == "grouped_recipes":
                    valid_groups = [
                        g for g in st.session_state.get("recipe_groups", []) if g
                    ]
                    # Initialize crop_regions if not exists
                    if "crop_regions" not in st.session_state:
                        st.session_state.crop_regions = {}

                    # Calculate overall progress
                    total_crops = 0
                    recipes_with_crops = 0
                    for i in range(len(valid_groups)):
                        recipe_has_crops = False
                        for img_idx_in_group, img_idx in enumerate(valid_groups[i]):
                            image_key = f"group_{i}_img_{img_idx_in_group}"
                            if (
                                image_key in st.session_state.crop_regions
                                and st.session_state.crop_regions[image_key]
                            ):
                                total_crops += len(
                                    st.session_state.crop_regions[image_key]
                                )
                                recipe_has_crops = True
                        if recipe_has_crops:
                            recipes_with_crops += 1

                    # Show progress summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Recipes", len(valid_groups))
                    with col2:
                        st.metric(
                            "Recipes with Crops",
                            f"{recipes_with_crops}/{len(valid_groups)}",
                        )
                    with col3:
                        st.metric("Total Images Cropped", total_crops)

                    # Initialize cropping state
                    if "current_group_crop" not in st.session_state:
                        st.session_state.current_group_crop = 0

                    current_group = st.session_state.current_group_crop

                    # Recipe navigation header
                    st.markdown("### 📚 Recipe Navigation")
                    nav_cols = st.columns(
                        len(valid_groups) if len(valid_groups) <= 5 else 5
                    )

                    # Show navigation buttons for all recipe groups
                    for i in range(len(valid_groups)):
                        col_idx = i % len(nav_cols)
                        with nav_cols[col_idx]:
                            # Check if this recipe has any crops
                            has_crops = False
                            group_crop_count = 0
                            for img_idx_in_group, img_idx in enumerate(valid_groups[i]):
                                image_key = f"group_{i}_img_{img_idx_in_group}"
                                if (
                                    image_key in st.session_state.crop_regions
                                    and st.session_state.crop_regions[image_key]
                                ):
                                    has_crops = True
                                    group_crop_count += len(
                                        st.session_state.crop_regions[image_key]
                                    )

                            # Create button label with status
                            if has_crops:
                                button_label = f"✅ Recipe {i + 1} ({group_crop_count})"
                                button_help = f"Recipe {i + 1} - {group_crop_count} images cropped"
                            else:
                                button_label = f"📖 Recipe {i + 1}"
                                button_help = f"Recipe {i + 1} - No crops yet"

                            if i == current_group:
                                st.button(
                                    button_label,
                                    type="primary",
                                    disabled=True,
                                    use_container_width=True,
                                    help="Current recipe",
                                )
                            else:
                                if st.button(
                                    button_label,
                                    type="secondary",
                                    use_container_width=True,
                                    help=button_help,
                                ):
                                    st.session_state.current_group_crop = i
                                    st.session_state.current_image_crop = 0
                                    st.rerun()

                    st.markdown("---")

                    if current_group < len(valid_groups):
                        group_indices = valid_groups[current_group]

                        st.write(
                            f"### Recipe Group {current_group + 1} of {len(valid_groups)}"
                        )
                        st.write(f"**Images in this group:** {len(group_indices)}")

                        # Show clickable thumbnails of images in this group
                        st.write("**Click any image to jump to it:**")
                        cols = st.columns(min(len(group_indices), 4))
                        for i, img_idx in enumerate(group_indices):
                            with cols[i % len(cols)]:
                                # Create a unique button key for each image
                                button_key = f"jump_to_img_{current_group}_{i}"

                                # Make the button clickable
                                if st.button(
                                    f"📷 Image {i + 1}",
                                    key=button_key,
                                    use_container_width=True,
                                    type=(
                                        "primary"
                                        if i
                                        == st.session_state.get("current_image_crop", 0)
                                        else "secondary"
                                    ),
                                ):
                                    st.session_state.current_image_crop = i
                                    st.rerun()

                                # Show the thumbnail with border styling
                                uploaded_files[img_idx].seek(0)
                                img = Image.open(uploaded_files[img_idx])

                                # Convert image to base64 for inline display with border
                                buffered = BytesIO()
                                img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                                img.save(buffered, format="PNG")
                                img_str = base64.b64encode(buffered.getvalue()).decode()

                                # Display image with appropriate border
                                if i == st.session_state.get("current_image_crop", 0):
                                    # Highlight current image with blue border
                                    st.markdown(
                                        f"""
                                        <div style="
                                            border: 4px solid #1f77b4;
                                            border-radius: 8px;
                                            padding: 4px;
                                            background-color: white;
                                            margin-top: 8px;
                                        ">
                                            <img src="data:image/png;base64,{img_str}" style="
                                                width: 100%;
                                                border-radius: 4px;
                                                display: block;
                                            ">
                                            <p style="text-align: center; margin: 4px 0 0 0; font-size: 12px;">
                                                {i+1}. {uploaded_files[img_idx].name[:20]}{'...' if len(uploaded_files[img_idx].name) > 20 else ''}
                                            </p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    # Regular image with gray border
                                    st.markdown(
                                        f"""
                                        <div style="
                                            border: 2px solid #e0e0e0;
                                            border-radius: 8px;
                                            padding: 4px;
                                            background-color: white;
                                            margin-top: 8px;
                                        ">
                                            <img src="data:image/png;base64,{img_str}" style="
                                                width: 100%;
                                                border-radius: 4px;
                                                display: block;
                                            ">
                                            <p style="text-align: center; margin: 4px 0 0 0; font-size: 12px;">
                                                {i+1}. {uploaded_files[img_idx].name[:20]}{'...' if len(uploaded_files[img_idx].name) > 20 else ''}
                                            </p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                        # Initialize current image cropping
                        if "current_image_crop" not in st.session_state:
                            st.session_state.current_image_crop = 0

                        current_img_idx_in_group = st.session_state.current_image_crop

                        if current_img_idx_in_group < len(group_indices):
                            img_idx = group_indices[current_img_idx_in_group]
                            uploaded_file = uploaded_files[img_idx]

                            # Show current position with navigation dots
                            nav_indicators = []
                            for idx in range(len(group_indices)):
                                if idx == current_img_idx_in_group:
                                    nav_indicators.append("🔵")
                                else:
                                    nav_indicators.append("⚪")

                            st.write(
                                f"#### Cropping Image {current_img_idx_in_group + 1} of {len(group_indices)}: {uploaded_file.name}"
                            )
                            st.write("Progress: " + " ".join(nav_indicators))

                            # Load and display the image for cropping
                            image = Image.open(uploaded_file)
                            cropper = StreamlitImageCropper()

                            # Use a unique key for each image
                            image_key = (
                                f"group_{current_group}_img_{current_img_idx_in_group}"
                            )
                            crops = cropper.crop_single_image_canvas(
                                image, image_key, max_crops=5
                            )

                            # Check if the user clicked "Next Page"
                            if st.session_state.get(
                                f"{image_key}_page_complete", False
                            ):
                                # Clear the done flag
                                st.session_state[f"{image_key}_page_complete"] = False
                                # Navigate to next image or group
                                if current_img_idx_in_group < len(group_indices) - 1:
                                    st.session_state.current_image_crop += 1
                                    st.rerun()
                                else:
                                    if current_group < len(valid_groups) - 1:
                                        st.session_state.current_group_crop += 1
                                        st.session_state.current_image_crop = 0
                                        st.rerun()
                                    else:
                                        st.session_state.crop_step_completed = True
                                        st.session_state.crop_step_active = False
                                        st.success(
                                            "🎉 Manual cropping completed! Now you can extract the recipes."
                                        )
                                        st.rerun()

                            # Navigation buttons
                            col1, col2, col3 = st.columns([1, 1, 1])

                            with col1:
                                # Previous image or previous recipe
                                if current_img_idx_in_group > 0:
                                    if st.button("⬅️ Previous Image"):
                                        st.session_state.current_image_crop -= 1
                                        st.rerun()
                                elif current_group > 0:
                                    # At first image of recipe, allow going back to previous recipe
                                    if st.button("⬅️ Previous Recipe"):
                                        st.session_state.current_group_crop -= 1
                                        # Go to last image of previous recipe
                                        prev_group_indices = valid_groups[
                                            st.session_state.current_group_crop
                                        ]
                                        st.session_state.current_image_crop = (
                                            len(prev_group_indices) - 1
                                        )
                                        st.rerun()

                            with col2:
                                # Always allow navigation, don't require crops
                                if current_img_idx_in_group < len(group_indices) - 1:
                                    if st.button("Next Image ➡️"):
                                        st.session_state.current_image_crop += 1
                                        st.rerun()
                                # Always show recipe group navigation if there are more groups
                                elif current_group < len(valid_groups) - 1:
                                    if st.button("Next Recipe Group ➡️", type="primary"):
                                        st.session_state.current_group_crop += 1
                                        st.session_state.current_image_crop = 0
                                        st.rerun()

                            with col3:
                                # Always show both options - skip to next (if available) and finish
                                col3_buttons = []

                                # Show skip to next recipe group if available
                                if current_group < len(valid_groups) - 1:
                                    if st.button("⏭️ Skip to Next Recipe"):
                                        st.session_state.current_group_crop += 1
                                        st.session_state.current_image_crop = 0
                                        st.rerun()
                                    col3_buttons.append("skip")

                                # Always show finish cropping button
                                help_text = (
                                    "Complete cropping and proceed to extraction"
                                    if current_group == len(valid_groups) - 1
                                    else "Finish early and proceed to extraction"
                                )
                                if st.button(
                                    "✅ Finish Cropping",
                                    type=(
                                        "primary"
                                        if current_group == len(valid_groups) - 1
                                        else "secondary"
                                    ),
                                    help=help_text,
                                ):
                                    st.session_state.crop_step_completed = True
                                    st.session_state.crop_step_active = False
                                    st.success(
                                        "🎉 Manual cropping completed! Now you can extract the recipes."
                                    )
                                    st.rerun()

                elif processing_mode == "single_recipe":
                    st.info("📄 You'll crop images for 1 combined recipe.")
                    # Handle single recipe cropping
                    if "current_image_crop" not in st.session_state:
                        st.session_state.current_image_crop = 0

                    current_img_idx = st.session_state.current_image_crop

                    if current_img_idx < len(uploaded_files):
                        uploaded_file = uploaded_files[current_img_idx]
                        st.write(
                            f"#### Cropping Image {current_img_idx + 1} of {len(uploaded_files)}: {uploaded_file.name}"
                        )

                        image = Image.open(uploaded_file)
                        cropper = StreamlitImageCropper()
                        image_key = f"single_img_{current_img_idx}"
                        crops = cropper.crop_single_image_canvas(
                            image, image_key, max_crops=5
                        )

                        # Navigation buttons
                        col1, col2, col3 = st.columns([1, 1, 1])

                        with col1:
                            if current_img_idx > 0:
                                if st.button("⬅️ Previous Image"):
                                    st.session_state.current_image_crop -= 1
                                    st.rerun()

                        with col2:
                            if current_img_idx < len(uploaded_files) - 1:
                                if st.button("Next Image ➡️"):
                                    st.session_state.current_image_crop += 1
                                    st.rerun()

                        with col3:
                            # Finish cropping button - always visible
                            if st.button(
                                "✅ Finish Cropping",
                                type="primary",
                                help="Complete cropping and proceed to extraction",
                            ):
                                st.session_state.crop_step_completed = True
                                st.session_state.crop_step_active = False
                                st.success(
                                    "🎉 Manual cropping completed! Now you can extract the recipes."
                                )
                                st.rerun()

                else:  # multiple_recipes
                    st.info("🔄 You'll crop images for each individual recipe.")
                    # Handle multiple recipes cropping (similar to single recipe)
                    if "current_image_crop" not in st.session_state:
                        st.session_state.current_image_crop = 0

                    current_img_idx = st.session_state.current_image_crop

                    if current_img_idx < len(uploaded_files):
                        uploaded_file = uploaded_files[current_img_idx]
                        st.write(
                            f"#### Cropping Recipe {current_img_idx + 1} of {len(uploaded_files)}: {uploaded_file.name}"
                        )

                        image = Image.open(uploaded_file)
                        cropper = StreamlitImageCropper()
                        image_key = f"multi_img_{current_img_idx}"
                        crops = cropper.crop_single_image_canvas(
                            image, image_key, max_crops=5
                        )

                        # Navigation buttons
                        col1, col2, col3 = st.columns([1, 1, 1])

                        with col1:
                            if current_img_idx > 0:
                                if st.button("⬅️ Previous Recipe"):
                                    st.session_state.current_image_crop -= 1
                                    st.rerun()

                        with col2:
                            if current_img_idx < len(uploaded_files) - 1:
                                if st.button("Next Recipe ➡️"):
                                    st.session_state.current_image_crop += 1
                                    st.rerun()

                        with col3:
                            # Finish cropping button - always visible
                            if st.button(
                                "✅ Finish Cropping",
                                type="primary",
                                help="Complete cropping and proceed to extraction",
                            ):
                                st.session_state.crop_step_completed = True
                                st.session_state.crop_step_active = False
                                st.success(
                                    "🎉 Manual cropping completed! Now you can extract the recipes."
                                )
                                st.rerun()

            if extract_button:
                progress_bar = st.progress(0)
                status_text = st.empty()

                extracted_recipes = []

                if processing_mode == "single_recipe":
                    # Process all images as one combined recipe
                    status_text.text("Combining all images into one recipe...")

                    try:
                        # Save all uploaded files temporarily (excluding deleted)
                        temp_paths = []
                        active_indices = [
                            i
                            for i in range(len(uploaded_files))
                            if i not in st.session_state.deleted_files
                        ]
                        status_text.text("Preparing images...")
                        for counter, idx in enumerate(active_indices):
                            uploaded_file = uploaded_files[idx]
                            progress_bar.progress(
                                (counter / len(active_indices)) * 0.2
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
                            recipe_dir = self.save_recipe_with_html(recipe, formatter)
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

                                # Check if manual cropping was done or skipped
                                has_crops = False
                                if manual_crop and st.session_state.get(
                                    "crop_step_completed", False
                                ):
                                    for i in range(len(temp_paths)):
                                        image_key = f"single_img_{i}"
                                        if (
                                            image_key in st.session_state.crop_regions
                                            and st.session_state.crop_regions[image_key]
                                        ):
                                            has_crops = True
                                            break

                                if has_crops:
                                    # Use pre-cropped images from the manual cropping step
                                    status_text.text("Using manually cropped images...")
                                    progress_bar.progress(0.8)

                                    extracted_images = []

                                    # Save original images to /images/originals
                                    originals_dir = os.path.join(
                                        recipe_dir, "images", "originals"
                                    )
                                    os.makedirs(originals_dir, exist_ok=True)
                                    # Only process originals if we're doing manual cropping with actual crops
                                    active_indices = [
                                        i
                                        for i in range(len(uploaded_files))
                                        if i not in st.session_state.deleted_files
                                    ]
                                    for counter, idx in enumerate(active_indices):
                                        uploaded_file = uploaded_files[idx]
                                        temp_path = temp_paths[counter]
                                        # Convert HEIC/HEIF to JPEG, keep others as-is
                                        original_name = uploaded_file.name
                                        if original_name.lower().endswith(
                                            (".heic", ".heif")
                                        ):
                                            # Convert to JPEG with high quality
                                            output_name = (
                                                os.path.splitext(original_name)[0]
                                                + ".jpg"
                                            )
                                            original_path = os.path.join(
                                                originals_dir, output_name
                                            )
                                            img = Image.open(temp_path)
                                            # Convert to RGB if necessary (HEIC might have alpha channel)
                                            if img.mode in ("RGBA", "LA", "P"):
                                                rgb_img = Image.new(
                                                    "RGB", img.size, (255, 255, 255)
                                                )
                                                rgb_img.paste(
                                                    img,
                                                    mask=(
                                                        img.split()[-1]
                                                        if img.mode == "RGBA"
                                                        else None
                                                    ),
                                                )
                                                img = rgb_img
                                            img.save(
                                                original_path,
                                                "JPEG",
                                                quality=95,
                                                optimize=True,
                                            )
                                        else:
                                            # Keep original format for non-HEIC files
                                            original_path = os.path.join(
                                                originals_dir, original_name
                                            )
                                            Image.open(temp_path).save(original_path)

                                    # Retrieve cropped images from session state
                                    for i, temp_path in enumerate(temp_paths):
                                        image_key = f"single_img_{i}"

                                        if image_key in st.session_state.crop_regions:
                                            crops = st.session_state.crop_regions[
                                                image_key
                                            ]

                                            for j, crop_info in enumerate(crops):
                                                if crop_info.get("cropped_image"):
                                                    # Save the cropped image with _main or _step suffix
                                                    suffix = (
                                                        "_main"
                                                        if crop_info.get(
                                                            "is_main", False
                                                        )
                                                        else "_step"
                                                    )
                                                    cropped_filename = f"{recipe.name}_image_{len(extracted_images)+1}{suffix}.png"
                                                    cropped_path = os.path.join(
                                                        recipe_dir,
                                                        "images",
                                                        cropped_filename,
                                                    )

                                                    # Ensure images directory exists
                                                    os.makedirs(
                                                        os.path.dirname(cropped_path),
                                                        exist_ok=True,
                                                    )

                                                    # Save the pre-cropped image
                                                    crop_info["cropped_image"].save(
                                                        cropped_path
                                                    )

                                                    extracted_images.append(
                                                        {
                                                            "filename": cropped_filename,
                                                            "description": crop_info.get(
                                                                "description", ""
                                                            ),
                                                            "is_main": crop_info.get(
                                                                "is_main",
                                                                len(extracted_images)
                                                                == 0,
                                                            ),
                                                            "is_step": not crop_info.get(
                                                                "is_main",
                                                                len(extracted_images)
                                                                == 0,
                                                            ),
                                                        }
                                                    )

                                    # Create metadata
                                    image_metadata = {
                                        "extracted_images": extracted_images
                                    }
                                elif (
                                    st.session_state.get("crop_step_completed", False)
                                    and not has_crops
                                ):
                                    # Cropping was skipped - use full images
                                    status_text.text(
                                        "Using full images (cropping skipped)..."
                                    )
                                    progress_bar.progress(0.8)

                                    # Use automatic image extraction
                                    recipe_image_extractor = RecipeImageExtractor(
                                        llm_client
                                    )
                                    image_metadata = recipe_image_extractor.extract_recipe_images(
                                        temp_paths,
                                        recipe.name,
                                        recipe_dir,
                                        progress_callback=extraction_progress_callback,
                                    )
                                else:
                                    # Use automatic image extraction
                                    recipe_image_extractor = RecipeImageExtractor(
                                        llm_client
                                    )
                                    image_metadata = recipe_image_extractor.extract_recipe_images(
                                        temp_paths,
                                        recipe.name,
                                        recipe_dir,
                                        progress_callback=extraction_progress_callback,
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

                            # Store recipe with its directory path
                            recipe._recipe_dir = recipe_dir  # Add custom attribute
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
                                recipe_dir = self.save_recipe_with_html(
                                    recipe, formatter
                                )

                                # Extract images
                                if len(temp_paths) > 1:
                                    # Initialize has_crops to False
                                    has_crops = False

                                    if manual_crop and st.session_state.get(
                                        "crop_step_completed", False
                                    ):
                                        # Check if cropping was actually done or skipped
                                        for relative_idx, img_idx in enumerate(
                                            image_indices
                                        ):
                                            image_key = (
                                                f"group_{group_idx}_img_{relative_idx}"
                                            )
                                            if (
                                                image_key
                                                in st.session_state.crop_regions
                                                and st.session_state.crop_regions[
                                                    image_key
                                                ]
                                            ):
                                                has_crops = True
                                                break

                                    if has_crops:
                                        # Use pre-cropped images from the manual cropping step
                                        extracted_images = []

                                        # Save original images to /images/originals
                                        originals_dir = os.path.join(
                                            recipe_dir, "images", "originals"
                                        )
                                        os.makedirs(originals_dir, exist_ok=True)

                                        if has_crops:
                                            # Save originals for this group only if we have crops
                                            for relative_idx, img_idx in enumerate(
                                                image_indices
                                            ):
                                                original_file = uploaded_files[img_idx]
                                                temp_path = temp_paths[relative_idx]

                                                # Convert HEIC/HEIF to JPEG, keep others as-is
                                                original_name = original_file.name
                                                if original_name.lower().endswith(
                                                    (".heic", ".heif")
                                                ):
                                                    # Convert to JPEG with high quality
                                                    output_name = (
                                                        os.path.splitext(original_name)[
                                                            0
                                                        ]
                                                        + ".jpg"
                                                    )
                                                    original_path = os.path.join(
                                                        originals_dir, output_name
                                                    )
                                                    img = Image.open(temp_path)
                                                    # Convert to RGB if necessary (HEIC might have alpha channel)
                                                    if img.mode in ("RGBA", "LA", "P"):
                                                        rgb_img = Image.new(
                                                            "RGB",
                                                            img.size,
                                                            (255, 255, 255),
                                                        )
                                                        rgb_img.paste(
                                                            img,
                                                            mask=(
                                                                img.split()[-1]
                                                                if img.mode == "RGBA"
                                                                else None
                                                            ),
                                                        )
                                                        img = rgb_img
                                                    img.save(
                                                        original_path,
                                                        "JPEG",
                                                        quality=95,
                                                        optimize=True,
                                                    )
                                                else:
                                                    # Keep original format for non-HEIC files
                                                    original_path = os.path.join(
                                                        originals_dir, original_name
                                                    )
                                                    Image.open(temp_path).save(
                                                        original_path
                                                    )

                                            # Get the images for this group from the original image indices
                                            for relative_idx, img_idx in enumerate(
                                                image_indices
                                            ):
                                                image_key = f"group_{group_idx}_img_{relative_idx}"

                                                if (
                                                    image_key
                                                    in st.session_state.crop_regions
                                                ):
                                                    crops = (
                                                        st.session_state.crop_regions[
                                                            image_key
                                                        ]
                                                    )

                                                    for j, crop_info in enumerate(
                                                        crops
                                                    ):
                                                        if crop_info.get(
                                                            "cropped_image"
                                                        ):
                                                            # Save the cropped image with _main or _step suffix
                                                            suffix = (
                                                                "_main"
                                                                if crop_info.get(
                                                                    "is_main", False
                                                                )
                                                                else "_step"
                                                            )
                                                            cropped_filename = f"{recipe.name}_image_{len(extracted_images)+1}{suffix}.png"
                                                            cropped_path = os.path.join(
                                                                recipe_dir,
                                                                "images",
                                                                cropped_filename,
                                                            )

                                                            # Ensure images directory exists
                                                            os.makedirs(
                                                                os.path.dirname(
                                                                    cropped_path
                                                                ),
                                                                exist_ok=True,
                                                            )

                                                            # Save the pre-cropped image
                                                            crop_info[
                                                                "cropped_image"
                                                            ].save(cropped_path)

                                                            extracted_images.append(
                                                                {
                                                                    "filename": cropped_filename,
                                                                    "description": crop_info.get(
                                                                        "description",
                                                                        "",
                                                                    ),
                                                                    "is_main": crop_info.get(
                                                                        "is_main",
                                                                        len(
                                                                            extracted_images
                                                                        )
                                                                        == 0,
                                                                    ),
                                                                    "is_step": not crop_info.get(
                                                                        "is_main",
                                                                        len(
                                                                            extracted_images
                                                                        )
                                                                        == 0,
                                                                    ),
                                                                }
                                                            )

                                            image_metadata = {
                                                "extracted_images": extracted_images
                                            }
                                    elif (
                                        st.session_state.get(
                                            "crop_step_completed", False
                                        )
                                        and not has_crops
                                    ):
                                        # Cropping was skipped - use automatic extraction
                                        recipe_image_extractor = RecipeImageExtractor(
                                            llm_client
                                        )
                                        image_metadata = recipe_image_extractor.extract_recipe_images(
                                            temp_paths, recipe.name, recipe_dir
                                        )
                                    else:
                                        # Default: Automatic image extraction
                                        recipe_image_extractor = RecipeImageExtractor(
                                            llm_client
                                        )
                                        image_metadata = recipe_image_extractor.extract_recipe_images(
                                            temp_paths, recipe.name, recipe_dir
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
                                else:
                                    # Single image in group - save it as the main image
                                    # For single images, we'll save the original as the main image
                                    images_dir = os.path.join(recipe_dir, "images")
                                    os.makedirs(images_dir, exist_ok=True)

                                    # Also create originals directory
                                    originals_dir = os.path.join(
                                        images_dir, "originals"
                                    )
                                    os.makedirs(originals_dir, exist_ok=True)

                                    # Save the single image and its original
                                    img = Image.open(temp_paths[0])
                                    original_file = group_files[0]

                                    # Save original (with HEIC conversion if needed)
                                    original_name = original_file.name
                                    if original_name.lower().endswith(
                                        (".heic", ".heif")
                                    ):
                                        # Convert to JPEG
                                        output_name = (
                                            os.path.splitext(original_name)[0] + ".jpg"
                                        )
                                        original_path = os.path.join(
                                            originals_dir, output_name
                                        )

                                        # Convert to RGB if necessary
                                        if img.mode in ("RGBA", "LA", "P"):
                                            rgb_img = Image.new(
                                                "RGB", img.size, (255, 255, 255)
                                            )
                                            rgb_img.paste(
                                                img,
                                                mask=(
                                                    img.split()[-1]
                                                    if img.mode == "RGBA"
                                                    else None
                                                ),
                                            )
                                            img = rgb_img

                                        img.save(
                                            original_path,
                                            "JPEG",
                                            quality=95,
                                            optimize=True,
                                        )
                                    else:
                                        # Keep original format
                                        original_path = os.path.join(
                                            originals_dir, original_name
                                        )
                                        img.save(original_path)

                                    # Save as main image
                                    main_filename = f"{recipe.name}_main.jpg"
                                    main_path = os.path.join(images_dir, main_filename)

                                    # Convert to RGB if necessary
                                    if img.mode in ("RGBA", "LA", "P"):
                                        rgb_img = Image.new(
                                            "RGB", img.size, (255, 255, 255)
                                        )
                                        rgb_img.paste(
                                            img,
                                            mask=(
                                                img.split()[-1]
                                                if img.mode == "RGBA"
                                                else None
                                            ),
                                        )
                                        img = rgb_img

                                    img.save(
                                        main_path, "JPEG", quality=95, optimize=True
                                    )

                                    # Update recipe with the image
                                    recipe.images = [
                                        RecipeImage(
                                            filename=main_filename,
                                            description="Main recipe image",
                                            is_main=True,
                                            is_step=False,
                                        )
                                    ]

                                    # Update saved files
                                    formatter.update_recipe_files(recipe, recipe_dir)

                                # Store recipe with its directory path
                                recipe._recipe_dir = recipe_dir
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
                    # Process each image as a separate recipe (excluding deleted)
                    active_files = [
                        (i, uploaded_file)
                        for i, uploaded_file in enumerate(uploaded_files)
                        if i not in st.session_state.deleted_files
                    ]
                    for counter, (i, uploaded_file) in enumerate(active_files):
                        status_text.text(f"Processing {uploaded_file.name}...")
                        progress_bar.progress((counter + 1) / len(active_files))

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
                                recipe_dir = self.save_recipe_with_html(
                                    recipe, formatter
                                )

                                # Extract recipe images
                                has_crops = False
                                if manual_crop and st.session_state.get(
                                    "crop_step_completed", False
                                ):
                                    # Check if cropping was actually done or skipped
                                    image_key = f"multi_img_{i}"
                                    has_crops = (
                                        image_key in st.session_state.crop_regions
                                        and st.session_state.crop_regions[image_key]
                                    )

                                    if has_crops:
                                        # Use pre-cropped images from the manual cropping step
                                        extracted_images = []

                                        # Save original image to /images/originals
                                    else:
                                        # Cropping was skipped - use automatic extraction
                                        manual_crop = False

                                if has_crops:
                                    originals_dir = os.path.join(
                                        recipe_dir, "images", "originals"
                                    )
                                    os.makedirs(originals_dir, exist_ok=True)

                                    # Convert HEIC/HEIF to JPEG, keep others as-is
                                    original_name = uploaded_file.name
                                    if original_name.lower().endswith(
                                        (".heic", ".heif")
                                    ):
                                        # Convert to JPEG with high quality
                                        output_name = (
                                            os.path.splitext(original_name)[0] + ".jpg"
                                        )
                                        original_path = os.path.join(
                                            originals_dir, output_name
                                        )
                                        img = Image.open(tmp_file_path)
                                        # Convert to RGB if necessary (HEIC might have alpha channel)
                                        if img.mode in ("RGBA", "LA", "P"):
                                            rgb_img = Image.new(
                                                "RGB", img.size, (255, 255, 255)
                                            )
                                            rgb_img.paste(
                                                img,
                                                mask=(
                                                    img.split()[-1]
                                                    if img.mode == "RGBA"
                                                    else None
                                                ),
                                            )
                                            img = rgb_img
                                        img.save(
                                            original_path,
                                            "JPEG",
                                            quality=95,
                                            optimize=True,
                                        )
                                    else:
                                        # Keep original format for non-HEIC files
                                        original_path = os.path.join(
                                            originals_dir, original_name
                                        )
                                        Image.open(tmp_file_path).save(original_path)

                                    image_key = f"multi_img_{i}"

                                    if image_key in st.session_state.crop_regions:
                                        crops = st.session_state.crop_regions[image_key]

                                        for j, crop_info in enumerate(crops):
                                            if crop_info.get("cropped_image"):
                                                # Save the cropped image with _main or _step suffix
                                                suffix = (
                                                    "_main"
                                                    if crop_info.get("is_main", False)
                                                    else "_step"
                                                )
                                                cropped_filename = f"{recipe.name}_image_{len(extracted_images)+1}{suffix}.png"
                                                cropped_path = os.path.join(
                                                    recipe_dir,
                                                    "images",
                                                    cropped_filename,
                                                )

                                                # Ensure images directory exists
                                                os.makedirs(
                                                    os.path.dirname(cropped_path),
                                                    exist_ok=True,
                                                )

                                                # Save the pre-cropped image
                                                crop_info["cropped_image"].save(
                                                    cropped_path
                                                )

                                                extracted_images.append(
                                                    {
                                                        "filename": cropped_filename,
                                                        "description": crop_info.get(
                                                            "description", ""
                                                        ),
                                                        "is_main": crop_info.get(
                                                            "is_main",
                                                            len(extracted_images) == 0,
                                                        ),
                                                        "is_step": not crop_info.get(
                                                            "is_main",
                                                            len(extracted_images) == 0,
                                                        ),
                                                    }
                                                )

                                    image_metadata = {
                                        "extracted_images": extracted_images
                                    }
                                else:
                                    # Automatic image extraction
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

                                # Store recipe with its directory path
                                recipe._recipe_dir = recipe_dir
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
                    # Clear all session state and refresh the page
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
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

        col1, col2 = st.columns([1, 2])
        with col1:
            extract_button = st.button(
                "🚀 Extract Recipe",
                type="primary",
                disabled=not url_input,
                key="web_extract_button",
            )

        # Manual cropping is always enabled
        manual_crop = True

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
                    recipe_dir = self.save_recipe_with_html(recipe, formatter)

                    st.success(f"✅ Recipe extracted and saved to: {recipe_dir}")
                    st.session_state.recipe_dir = recipe_dir

                    # Handle image extraction from web page
                    if manual_crop:
                        st.info(
                            "Downloading images from web page for manual cropping..."
                        )

                        try:
                            # Download images from the web page
                            web_images = WebImageExtractor.download_images_from_url(
                                url_input
                            )

                            if web_images:
                                st.success(
                                    f"Downloaded {len(web_images)} images from the web page."
                                )

                                # Load images for cropping
                                images = []
                                for img_path in web_images:
                                    pil_img = Image.open(img_path)
                                    images.append((img_path, pil_img))

                                # Initialize cropper
                                cropper = StreamlitImageCropper()

                                # Perform cropping
                                st.write("### Manual Image Cropping")
                                st.write("Select recipe images from the web page.")

                                crop_regions = cropper.crop_multiple_images(
                                    images, recipe.name
                                )

                                if crop_regions is not None:
                                    # Save cropped images
                                    all_images = {path: img for path, img in images}
                                    metadata = cropper.save_cropped_images(
                                        all_images,
                                        crop_regions,
                                        recipe.name,
                                        Path(recipe_dir),
                                    )

                                    # Update recipe with images
                                    if metadata["extracted_images"]:
                                        recipe.images = [
                                            RecipeImage(
                                                filename=img["filename"],
                                                description=img["description"],
                                                is_main=img["is_main"],
                                                is_step=img["is_step"],
                                            )
                                            for img in metadata["extracted_images"]
                                        ]

                                        # Update recipe files
                                        formatter.update_recipe_files(
                                            recipe, recipe_dir
                                        )
                                        st.success(
                                            f"✅ Saved {len(metadata['extracted_images'])} cropped images!"
                                        )

                                # Clean up temporary web images
                                import shutil

                                temp_dir = os.path.dirname(web_images[0])
                                shutil.rmtree(temp_dir, ignore_errors=True)
                            else:
                                st.warning("No suitable images found on the web page.")

                        except Exception as e:
                            st.error(f"Error during image extraction: {str(e)}")

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

            col1, col2 = st.columns([1, 2])
            with col1:
                extract_button = st.button(
                    "🚀 Extract Recipe", type="primary", key="pdf_extract_button"
                )

            # Manual cropping is always enabled
            manual_crop = True

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
                            recipe_dir = self.save_recipe_with_html(recipe, formatter)

                            st.success(
                                f"✅ Recipe extracted and saved to: {recipe_dir}"
                            )
                            st.session_state.recipe_dir = recipe_dir

                            # Handle image extraction from PDF
                            if manual_crop:
                                st.info(
                                    "Converting PDF pages to images for manual cropping..."
                                )

                                try:
                                    # Convert PDF pages to images
                                    pdf_images = PDFImageExtractor.pdf_to_images(
                                        tmp_file_path, zero_indexed_pages
                                    )

                                    if pdf_images:
                                        st.success(
                                            f"Converted {len(pdf_images)} PDF pages to images."
                                        )

                                        # Load images for cropping
                                        images = []
                                        for img_path in pdf_images:
                                            pil_img = Image.open(img_path)
                                            images.append((img_path, pil_img))

                                        # Initialize cropper
                                        cropper = StreamlitImageCropper()

                                        # Perform cropping
                                        st.write("### Manual Image Cropping")
                                        st.write(
                                            "Select recipe images from the PDF pages."
                                        )

                                        crop_regions = cropper.crop_multiple_images(
                                            images, recipe.name
                                        )

                                        if crop_regions is not None:
                                            # Save cropped images
                                            all_images = {
                                                path: img for path, img in images
                                            }
                                            metadata = cropper.save_cropped_images(
                                                all_images,
                                                crop_regions,
                                                recipe.name,
                                                Path(recipe_dir),
                                            )

                                            # Update recipe with images
                                            if metadata["extracted_images"]:
                                                recipe.images = [
                                                    RecipeImage(
                                                        filename=img["filename"],
                                                        description=img["description"],
                                                        is_main=img["is_main"],
                                                        is_step=img["is_step"],
                                                    )
                                                    for img in metadata[
                                                        "extracted_images"
                                                    ]
                                                ]

                                                # Update recipe files
                                                formatter.update_recipe_files(
                                                    recipe, recipe_dir
                                                )
                                                st.success(
                                                    f"✅ Saved {len(metadata['extracted_images'])} cropped images!"
                                                )

                                        # Clean up temporary PDF images
                                        import shutil

                                        temp_dir = os.path.dirname(pdf_images[0])
                                        shutil.rmtree(temp_dir, ignore_errors=True)
                                    else:
                                        st.warning(
                                            "No images could be extracted from the PDF."
                                        )

                                except Exception as e:
                                    st.error(f"Error during image extraction: {str(e)}")

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

        # Display main recipe image if available
        recipe_dir = getattr(recipe, "_recipe_dir", None) or st.session_state.get(
            "recipe_dir"
        )
        if recipe_dir and recipe.images:
            main_images = [img for img in recipe.images if img.is_main]
            if main_images:
                image_path = Path(recipe_dir) / "images" / main_images[0].filename
                if image_path.exists():
                    st.image(
                        str(image_path),
                        caption=main_images[0].description or "Main recipe image",
                        width=600,  # Fixed width for reasonable display
                    )
            # Show all images if no main image
            elif recipe.images:
                # Just show the first image as main
                image_path = Path(recipe_dir) / "images" / recipe.images[0].filename
                if image_path.exists():
                    st.image(
                        str(image_path),
                        caption=recipe.images[0].description or "Recipe image",
                        width=600,  # Fixed width for reasonable display
                    )

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

        # Step-by-step images if available
        if recipe_dir and recipe.images:
            step_images = [img for img in recipe.images if img.is_step]
            if step_images:
                with st.expander("🖼️ Step-by-step Photos", expanded=True):
                    cols = st.columns(min(3, len(step_images)))
                    for i, img in enumerate(step_images):
                        image_path = Path(recipe_dir) / "images" / img.filename
                        if image_path.exists():
                            with cols[i % min(3, len(step_images))]:
                                st.image(
                                    str(image_path),
                                    caption=img.description or f"Step {i+1}",
                                    width=300,  # Smaller width for step images
                                )

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
        st.sidebar.markdown("### 📚 Recipe Collection")
        if st.sidebar.button(
            "Browse Recipe Collection →", use_container_width=True, type="primary"
        ):
            st.session_state.show_recipe_collection = True
            st.rerun()

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

        # Check if we should show recipe collection
        if st.session_state.get("show_recipe_collection", False):
            from app.pages.recipe_collection import show_recipe_collection

            show_recipe_collection()
            # Add back button
            if st.button("← Back to Extractor", key="back_to_extractor"):
                st.session_state.show_recipe_collection = False
                st.rerun()
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
    # Ensure all recipes have HTML files
    ensure_recipe_htmls_exist()

    app = StreamlitRecipeApp()
    app.run()


if __name__ == "__main__":
    main()
