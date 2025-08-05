"""Utility functions for recipe management."""

from app.recipe_html_generator import RecipeHTMLGenerator


def ensure_recipe_htmls_exist():
    """Ensure all recipes have HTML files generated."""
    try:
        generator = RecipeHTMLGenerator()
        recipes = generator.load_recipes()

        for recipe_info in recipes:
            recipe = recipe_info["recipe"]
            recipe_dir = recipe_info["dir_path"]
            html_path = recipe_dir / "index.html"

            # Generate HTML if it doesn't exist
            if not html_path.exists():
                generator.save_recipe_html(recipe, recipe_dir)

    except Exception as e:
        print(f"Error generating recipe HTMLs: {e}")
