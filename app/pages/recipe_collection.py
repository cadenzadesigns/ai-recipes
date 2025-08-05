"""Recipe Collection page for browsing all extracted recipes."""

import os
import sys
import webbrowser
from pathlib import Path

import streamlit as st

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.recipe_html_generator import RecipeHTMLGenerator


def show_recipe_collection():
    """Display the recipe collection page."""
    st.title("🍳 Recipe Collection")
    st.markdown("Browse your extracted recipes")

    # Search bar
    search_query = st.text_input(
        "Search recipes",
        placeholder="Search by name or description...",
        label_visibility="collapsed",
    )

    # Load recipes
    generator = RecipeHTMLGenerator()
    recipes = generator.load_recipes()

    if not recipes:
        st.info("No recipes found. Extract some recipes first!")
        return

    # Filter recipes based on search
    if search_query:
        filtered_recipes = []
        for recipe_info in recipes:
            recipe = recipe_info["recipe"]
            if search_query.lower() in recipe.name.lower() or (
                recipe.description
                and search_query.lower() in recipe.description.lower()
            ):
                filtered_recipes.append(recipe_info)
    else:
        filtered_recipes = recipes

    if not filtered_recipes:
        st.warning(f"No recipes found matching '{search_query}'")
        return

    # Display recipes in a grid
    cols_per_row = 3
    for i in range(0, len(filtered_recipes), cols_per_row):
        cols = st.columns(cols_per_row)

        for j, col in enumerate(cols):
            if i + j < len(filtered_recipes):
                recipe_info = filtered_recipes[i + j]
                recipe = recipe_info["recipe"]
                recipe_dir = recipe_info["dir_name"]

                with col:
                    # Create a container for the recipe card
                    with st.container():
                        # Find main image
                        main_image = None
                        if recipe.images:
                            main_images = [img for img in recipe.images if img.is_main]
                            if main_images:
                                main_image = main_images[0]
                            elif recipe.images:
                                main_image = recipe.images[0]

                        # Display image or placeholder
                        if main_image:
                            image_path = (
                                Path("recipes")
                                / recipe_dir
                                / "images"
                                / main_image.filename
                            )
                            if image_path.exists():
                                st.image(
                                    str(image_path),
                                    use_container_width=True,
                                    caption=None,
                                )
                            else:
                                # Placeholder
                                st.markdown(
                                    """
                                    <div style="
                                        height: 200px;
                                        background-color: #f0f0f0;
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        font-size: 3em;
                                        color: #999;
                                        border-radius: 8px;
                                    ">🍽️</div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                        else:
                            # Placeholder
                            st.markdown(
                                """
                                <div style="
                                    height: 200px;
                                    background-color: #f0f0f0;
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    font-size: 3em;
                                    color: #999;
                                    border-radius: 8px;
                                ">🍽️</div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Recipe title
                        st.markdown(f"### {recipe.name}")

                        # Description
                        if recipe.description:
                            description = (
                                recipe.description[:100] + "..."
                                if len(recipe.description) > 100
                                else recipe.description
                            )
                            st.markdown(f"*{description}*")

                        # Metadata
                        meta_parts = []
                        if recipe.servings:
                            meta_parts.append(f"🍽️ {recipe.servings}")
                        if recipe.total_time:
                            meta_parts.append(f"⏱️ {recipe.total_time}")

                        if meta_parts:
                            st.caption(" • ".join(meta_parts))

                        # View button - navigate to recipe viewer
                        if st.button(
                            "View Recipe",
                            key=f"view_{recipe_dir}",
                            use_container_width=True,
                            type="primary",
                        ):
                            # Store recipe info in session state
                            st.session_state.selected_recipe_name = recipe_dir
                            # Set query params
                            st.query_params.recipe = recipe_dir
                            st.switch_page("pages/recipe_viewer.py")


# Run the page
show_recipe_collection()
