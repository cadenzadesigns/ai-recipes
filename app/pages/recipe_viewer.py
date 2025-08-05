"""Recipe Viewer page with blog-style layout."""

import base64
import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent.parent))


def show_recipe_viewer(show_back_button=True):
    """Display recipe in a blog-style format."""

    # Get recipe name from query params or session state
    recipe_name = st.query_params.get("recipe")
    if not recipe_name and "selected_recipe_name" in st.session_state:
        recipe_name = st.session_state.selected_recipe_name

    if not recipe_name:
        st.error("No recipe specified.")
        if st.button("← Back to Recipe Collection", key="back_no_recipe"):
            # Just return - will go back to recipe collection
            return
        return

    # Load recipe data
    import json

    recipe_dir = Path("recipes") / recipe_name
    json_files = list(recipe_dir.glob("*.json"))

    if not json_files:
        st.error(f"Recipe data not found for '{recipe_name}'.")
        if st.button("← Back to Recipe Collection", key="back_no_data"):
            # Just return - will go back to recipe collection
            return
        return

    with open(json_files[0]) as f:
        recipe_data = json.load(f)

    # Set page title to recipe name for Paprika recognition
    full_title = recipe_data["name"]
    if recipe_data.get("alternate_name"):
        full_title = f"{recipe_data['name']} ({recipe_data['alternate_name']})"
    st.set_page_config(page_title=full_title, page_icon="🍳")

    # Create a professional blog-style layout
    st.markdown(
        """
    <style>
    /* Hide Streamlit branding and padding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .main > div {padding-top: 0rem;}

    /* Recipe styling - works in both light and dark mode */
    .recipe-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }

    .recipe-description {
        font-size: 1.2rem;
        margin-bottom: 1rem;
        font-style: italic;
        opacity: 0.8;
    }

    .author-credit {
        font-size: 0.9rem;
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(128, 128, 128, 0.3);
        opacity: 0.8;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Back button (only show if requested)
    if show_back_button:
        if st.button("← Back to Recipe Collection", key="back_main"):
            # Just return - will go back to recipe collection
            return

    # Recipe header
    recipe_title = recipe_data["name"]
    if recipe_data.get("alternate_name"):
        recipe_title += f' ({recipe_data["alternate_name"]})'

    st.markdown(f'<h1 class="recipe-title">{recipe_title}</h1>', unsafe_allow_html=True)

    # Author/source info
    if recipe_data.get("source"):
        st.markdown(f"By **{recipe_data['source']}**")

    # Updated date (use current date as placeholder)
    from datetime import datetime

    st.caption(f"Updated {datetime.now().strftime('%B %d, %Y')}")

    # Main recipe image
    if recipe_data.get("images"):
        main_images = [img for img in recipe_data["images"] if img.get("is_main")]
        if main_images:
            img = main_images[0]
            img_path = recipe_dir / "images" / img["filename"]
            if img_path.exists():
                # Display using Streamlit's image component
                st.image(
                    str(img_path),
                    use_container_width=True,
                    caption=img.get("description"),
                )

                # Add a small version of the image in HTML for Paprika
                with open(img_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode()

                mime_type = "image/jpeg"
                if img_path.suffix.lower() == ".png":
                    mime_type = "image/png"

                # Add a 1x1 pixel version that's hidden but detectable
                st.markdown(
                    f'<img src="data:{mime_type};base64,{encoded_string}" width="1" height="1" style="position: absolute; left: -9999px;" alt="{recipe_title}" />',
                    unsafe_allow_html=True,
                )

        elif recipe_data["images"]:
            # Use first image if no main image
            img = recipe_data["images"][0]
            img_path = recipe_dir / "images" / img["filename"]
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)

    # Recipe metadata in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Time", recipe_data.get("total_time", "—"))

    with col2:
        st.metric("Prep Time", "—")

    with col3:
        st.metric("Cook Time", "—")

    with col4:
        st.metric("Servings", recipe_data.get("servings", "—"))

    # Save/Print/Share buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        st.button("💾 Save", type="primary")
    with col2:
        st.button("🖨️ Print")

    # Description
    if recipe_data.get("description"):
        # Split description into paragraphs and display each one
        description = recipe_data["description"]
        paragraphs = description.split("\n\n")  # Split on double newlines
        for paragraph in paragraphs:
            if paragraph.strip():  # Only display non-empty paragraphs
                st.markdown(
                    f'<p class="recipe-description">{paragraph.strip()}</p>',
                    unsafe_allow_html=True,
                )

    # Two-column layout for ingredients and directions
    col_left, col_right = st.columns([1, 2])

    with col_left:
        # Ingredients section
        st.markdown("### Ingredients")

        # Use container with background styling
        with st.container():
            if recipe_data.get("components"):
                # Recipe has components
                for component in recipe_data["components"]:
                    st.markdown(f"**{component['title']}**")
                    for ingredient in component["ingredients"]:
                        # Format ingredient
                        ing_parts = []
                        if ingredient.get("amount"):
                            if ingredient["amount"].get("quantity"):
                                ing_parts.append(ingredient["amount"]["quantity"])
                            if ingredient["amount"].get("unit"):
                                ing_parts.append(ingredient["amount"]["unit"])
                            # Also check for metric measurements
                            if ingredient["amount"].get("metric_quantity"):
                                if not ingredient["amount"].get(
                                    "quantity"
                                ):  # Only add if no standard measurement
                                    ing_parts.append(
                                        ingredient["amount"]["metric_quantity"]
                                    )
                            if ingredient["amount"].get("metric_unit"):
                                if not ingredient["amount"].get(
                                    "unit"
                                ):  # Only add if no standard unit
                                    ing_parts.append(
                                        ingredient["amount"]["metric_unit"]
                                    )

                        if ingredient.get("item"):
                            item_str = ingredient["item"]["name"]
                            if ingredient["item"].get("modifiers"):
                                item_str += (
                                    f", {', '.join(ingredient['item']['modifiers'])}"
                                )
                            ing_parts.append(item_str)

                        st.markdown(f"{' '.join(ing_parts)}")
                    st.markdown("")
            elif recipe_data.get("ingredients"):
                # Simple ingredient list
                for ingredient in recipe_data["ingredients"]:
                    # Format ingredient
                    ing_parts = []
                    if ingredient.get("amount"):
                        if ingredient["amount"].get("quantity"):
                            ing_parts.append(ingredient["amount"]["quantity"])
                        if ingredient["amount"].get("unit"):
                            ing_parts.append(ingredient["amount"]["unit"])
                        # Also check for metric measurements
                        if ingredient["amount"].get("metric_quantity"):
                            if not ingredient["amount"].get(
                                "quantity"
                            ):  # Only add if no standard measurement
                                ing_parts.append(
                                    ingredient["amount"]["metric_quantity"]
                                )
                        if ingredient["amount"].get("metric_unit"):
                            if not ingredient["amount"].get(
                                "unit"
                            ):  # Only add if no standard unit
                                ing_parts.append(ingredient["amount"]["metric_unit"])

                    if ingredient.get("item"):
                        item_str = ingredient["item"]["name"]
                        if ingredient["item"].get("modifiers"):
                            item_str += (
                                f", {', '.join(ingredient['item']['modifiers'])}"
                            )
                        ing_parts.append(item_str)

                    st.markdown(f"{' '.join(ing_parts)}")

    with col_right:
        # Directions section
        st.markdown("### Directions")

        for direction in recipe_data.get("directions", []):
            st.markdown(f"{direction}")

    # Notes section
    if recipe_data.get("notes"):
        st.markdown("### Recipe Notes")
        with st.container():
            for note in recipe_data["notes"]:
                st.markdown(f"{note}")

    # Step-by-step images
    if recipe_data.get("images"):
        step_images = [img for img in recipe_data["images"] if img.get("is_step")]
        if step_images:
            st.markdown("### Step-by-Step Photos")
            cols = st.columns(3)
            for i, img in enumerate(step_images):
                img_path = recipe_dir / "images" / img["filename"]
                if img_path.exists():
                    with cols[i % 3]:
                        st.image(str(img_path), use_container_width=True)
                        if img.get("description"):
                            st.caption(img["description"])

    # Footer/credit
    if recipe_data.get("source"):
        st.markdown(
            f'<div class="author-credit">Recipe from {recipe_data["source"]}</div>',
            unsafe_allow_html=True,
        )

    # Additional original images
    originals_dir = recipe_dir / "images" / "originals"
    if originals_dir.exists():
        # Get all JPG files from originals folder
        jpg_files = (
            list(originals_dir.glob("*.jpg"))
            + list(originals_dir.glob("*.JPG"))
            + list(originals_dir.glob("*.jpeg"))
            + list(originals_dir.glob("*.JPEG"))
        )

        if jpg_files:
            st.markdown("---")
            st.markdown("### 📸 Additional Photos")
            st.caption("Original photos from the recipe source")

            # Display in a grid
            cols = st.columns(3)
            for i, jpg_path in enumerate(jpg_files):
                with cols[i % 3]:
                    st.image(str(jpg_path), use_container_width=True)
                    # Add hidden HTML img for Paprika
                    with open(jpg_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode()
                    st.markdown(
                        f'<img src="data:image/jpeg;base64,{encoded_string}" width="1" height="1" style="position: absolute; left: -9999px;" alt="Additional photo" />',
                        unsafe_allow_html=True,
                    )


# Wrapper function without back button
def show_recipe_viewer_without_back():
    """Display recipe viewer without the back button."""
    show_recipe_viewer(show_back_button=False)


# Run the page (when accessed directly)
if __name__ == "__main__":
    show_recipe_viewer()
