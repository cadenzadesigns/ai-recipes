"""Recipe display components for Streamlit web app."""

import json
from pathlib import Path
from typing import List, Optional

import streamlit as st

from src.models import Recipe


def display_recipe(
    recipe: Recipe, recipe_dir: Optional[str] = None, show_images: bool = True
) -> None:
    """
    Display a single recipe in Streamlit format with attractive layout.

    Args:
        recipe: Recipe object to display
        recipe_dir: Path to recipe directory containing images
        show_images: Whether to display recipe images
    """
    # Main recipe title
    st.title(recipe.name)

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

    # Description
    if recipe.description:
        st.markdown("### Description")
        st.write(recipe.description)

    # Display main image if available
    if show_images and recipe_dir and recipe.images:
        main_images = [img for img in recipe.images if img.is_main]
        if main_images:
            image_path = Path(recipe_dir) / "images" / main_images[0].filename
            if image_path.exists():
                st.image(
                    str(image_path),
                    caption=main_images[0].description,
                    use_container_width=True,
                )

    # Ingredients section
    st.markdown("### Ingredients")
    for ingredient in recipe.ingredients:
        st.markdown(f"• {ingredient.to_string()}")

    # Directions section
    st.markdown("### Directions")
    for i, direction in enumerate(recipe.directions, 1):
        st.markdown(f"**{i}.** {direction}")

    # Step-by-step images if available
    if show_images and recipe_dir and recipe.images:
        step_images = [img for img in recipe.images if img.is_step]
        if step_images:
            with st.expander("🖼️ Step-by-step photos", expanded=False):
                for img in step_images:
                    image_path = Path(recipe_dir) / "images" / img.filename
                    if image_path.exists():
                        st.image(str(image_path), caption=img.description, width=400)

    # Additional images
    if show_images and recipe_dir and recipe.images:
        other_images = [
            img for img in recipe.images if not img.is_main and not img.is_step
        ]
        if other_images:
            with st.expander("📸 Additional photos", expanded=False):
                cols = st.columns(min(3, len(other_images)))
                for i, img in enumerate(other_images):
                    image_path = Path(recipe_dir) / "images" / img.filename
                    if image_path.exists():
                        with cols[i % 3]:
                            st.image(
                                str(image_path), caption=img.description, width=200
                            )

    # Notes section (collapsible)
    if recipe.notes:
        with st.expander("📝 Notes & Tips", expanded=False):
            for note in recipe.notes:
                st.markdown(f"• {note}")


def display_recipe_gallery(
    recipes: List[Recipe], recipes_dir: str = "recipes"
) -> Optional[Recipe]:
    """
    Display recipes in a gallery view with selection capability.

    Args:
        recipes: List of Recipe objects to display
        recipes_dir: Base directory containing recipe folders

    Returns:
        Selected Recipe object or None
    """
    if not recipes:
        st.info("No recipes found.")
        return None

    st.markdown(f"### Recipe Gallery ({len(recipes)} recipes)")

    # Search/filter functionality
    search_term = st.text_input(
        "🔍 Search recipes", placeholder="Enter recipe name or ingredient..."
    )

    # Filter recipes based on search
    filtered_recipes = recipes
    if search_term:
        filtered_recipes = [
            recipe
            for recipe in recipes
            if search_term.lower() in recipe.name.lower()
            or any(
                search_term.lower() in ingredient.to_string().lower()
                for ingredient in recipe.ingredients
            )
        ]

    if not filtered_recipes:
        st.warning(f"No recipes found matching '{search_term}'")
        return None

    # Display recipes in a grid
    cols_per_row = 3
    rows = [
        filtered_recipes[i : i + cols_per_row]
        for i in range(0, len(filtered_recipes), cols_per_row)
    ]

    selected_recipe = None

    for row in rows:
        cols = st.columns(cols_per_row)

        for i, recipe in enumerate(row):
            with cols[i]:
                # Create recipe card
                with st.container():
                    st.markdown(f"**{recipe.name}**")

                    # Show main image thumbnail if available
                    recipe_dir = Path(recipes_dir) / _get_recipe_dirname(recipe.name)
                    if recipe.images:
                        main_images = [img for img in recipe.images if img.is_main]
                        if main_images:
                            image_path = recipe_dir / "images" / main_images[0].filename
                            if image_path.exists():
                                st.image(str(image_path), width=200)

                    # Show recipe description (truncated)
                    if recipe.description:
                        desc = (
                            recipe.description[:100] + "..."
                            if len(recipe.description) > 100
                            else recipe.description
                        )
                        st.write(desc)

                    # Show metadata
                    metadata = []
                    if recipe.servings:
                        metadata.append(f"🍽️ {recipe.servings}")
                    if recipe.total_time:
                        metadata.append(f"⏱️ {recipe.total_time}")

                    if metadata:
                        st.caption(" • ".join(metadata))

                    # Select button
                    if st.button("View Recipe", key=f"select_{recipe.name}"):
                        selected_recipe = recipe

    return selected_recipe


def create_recipe_download(
    recipe: Recipe, recipe_dir: Optional[str] = None, format_type: str = "text"
) -> bytes:
    """
    Create downloadable content for a recipe.

    Args:
        recipe: Recipe object to download
        recipe_dir: Path to recipe directory
        format_type: "text", "json", or "both"

    Returns:
        Bytes content for download
    """
    if format_type == "text":
        return recipe.to_text().encode("utf-8")
    elif format_type == "json":
        return json.dumps(recipe.model_dump(), indent=2).encode("utf-8")
    elif format_type == "both":
        # Create a combined text format
        content = []
        content.append("=" * 60)
        content.append("TEXT FORMAT")
        content.append("=" * 60)
        content.append(recipe.to_text())
        content.append("\n" + "=" * 60)
        content.append("JSON FORMAT")
        content.append("=" * 60)
        content.append(json.dumps(recipe.model_dump(), indent=2))
        return "\n".join(content).encode("utf-8")
    else:
        raise ValueError(f"Invalid format_type: {format_type}")


def display_recipe_with_download(
    recipe: Recipe, recipe_dir: Optional[str] = None
) -> None:
    """
    Display a recipe with download options.

    Args:
        recipe: Recipe object to display
        recipe_dir: Path to recipe directory containing images
    """
    # Display the recipe
    display_recipe(recipe, recipe_dir)

    # Download options
    st.markdown("---")
    st.markdown("### Download Recipe")

    col1, col2, col3 = st.columns(3)

    with col1:
        text_content = create_recipe_download(recipe, recipe_dir, "text")
        st.download_button(
            label="📄 Download as Text",
            data=text_content,
            file_name=f"{_get_recipe_dirname(recipe.name)}.txt",
            mime="text/plain",
        )

    with col2:
        json_content = create_recipe_download(recipe, recipe_dir, "json")
        st.download_button(
            label="📋 Download as JSON",
            data=json_content,
            file_name=f"{_get_recipe_dirname(recipe.name)}.json",
            mime="application/json",
        )

    with col3:
        both_content = create_recipe_download(recipe, recipe_dir, "both")
        st.download_button(
            label="📦 Download Both",
            data=both_content,
            file_name=f"{_get_recipe_dirname(recipe.name)}_complete.txt",
            mime="text/plain",
        )


def display_recipe_comparison(
    recipes: List[Recipe], recipe_dirs: Optional[List[str]] = None
) -> None:
    """
    Display multiple recipes side by side for comparison.

    Args:
        recipes: List of Recipe objects to compare
        recipe_dirs: List of recipe directory paths
    """
    if not recipes:
        st.info("No recipes to compare.")
        return

    if len(recipes) > 3:
        st.warning("Comparison limited to first 3 recipes for better display.")
        recipes = recipes[:3]
        if recipe_dirs:
            recipe_dirs = recipe_dirs[:3]

    st.markdown("### Recipe Comparison")

    cols = st.columns(len(recipes))

    for i, recipe in enumerate(recipes):
        recipe_dir = recipe_dirs[i] if recipe_dirs and i < len(recipe_dirs) else None

        with cols[i]:
            st.markdown(f"#### {recipe.name}")

            # Basic info
            if recipe.servings:
                st.write(f"**Servings:** {recipe.servings}")
            if recipe.total_time:
                st.write(f"**Time:** {recipe.total_time}")

            # Description (truncated)
            if recipe.description:
                desc = (
                    recipe.description[:150] + "..."
                    if len(recipe.description) > 150
                    else recipe.description
                )
                st.write(f"**Description:** {desc}")

            # Ingredients count
            st.write(f"**Ingredients:** {len(recipe.ingredients)} items")

            # Steps count
            st.write(f"**Steps:** {len(recipe.directions)} steps")

            # Show main image if available
            if recipe_dir and recipe.images:
                main_images = [img for img in recipe.images if img.is_main]
                if main_images:
                    image_path = Path(recipe_dir) / "images" / main_images[0].filename
                    if image_path.exists():
                        st.image(str(image_path), width=200)


def _get_recipe_dirname(recipe_name: str) -> str:
    """Get a safe directory name for a recipe (matches formatter.py logic)."""
    safe_name = "".join(
        c for c in recipe_name if c.isalnum() or c in (" ", "-", "_")
    ).rstrip()
    return safe_name.replace(" ", "_").lower()


def display_recipe_stats(recipes: List[Recipe]) -> None:
    """
    Display statistics about a collection of recipes.

    Args:
        recipes: List of Recipe objects to analyze
    """
    if not recipes:
        st.info("No recipes to analyze.")
        return

    st.markdown("### Recipe Collection Statistics")

    # Basic stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Recipes", len(recipes))

    with col2:
        total_ingredients = sum(len(recipe.ingredients) for recipe in recipes)
        avg_ingredients = total_ingredients / len(recipes) if recipes else 0
        st.metric("Avg Ingredients", f"{avg_ingredients:.1f}")

    with col3:
        total_steps = sum(len(recipe.directions) for recipe in recipes)
        avg_steps = total_steps / len(recipes) if recipes else 0
        st.metric("Avg Steps", f"{avg_steps:.1f}")

    with col4:
        recipes_with_images = sum(1 for recipe in recipes if recipe.images)
        st.metric("With Images", recipes_with_images)

    # Most common ingredients
    ingredient_counts = {}
    for recipe in recipes:
        for ingredient in recipe.ingredients:
            name = ingredient.item.name.lower()
            ingredient_counts[name] = ingredient_counts.get(name, 0) + 1

    if ingredient_counts:
        st.markdown("#### Most Common Ingredients")
        sorted_ingredients = sorted(
            ingredient_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

        for ingredient, count in sorted_ingredients:
            st.write(f"• **{ingredient.title()}**: {count} recipes")

    # Recipe sources
    sources = {}
    for recipe in recipes:
        if recipe.source:
            sources[recipe.source] = sources.get(recipe.source, 0) + 1

    if sources:
        st.markdown("#### Recipe Sources")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            st.write(f"• **{source}**: {count} recipes")
