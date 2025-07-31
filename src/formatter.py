import json
from datetime import datetime
from pathlib import Path
from typing import List

from .models import Recipe


class RecipeFormatter:
    """Format and save recipes to text files."""

    def __init__(self, output_dir: str = "recipes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_recipe(self, recipe: Recipe, filename: str = None) -> str:
        """Save a recipe to both text and JSON files in a recipe-specific directory."""
        if not filename:
            # Generate filename from recipe name
            safe_name = "".join(
                c for c in recipe.name if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()
            safe_name = safe_name.replace(" ", "_").lower()
            filename = safe_name

        # Create recipe-specific directory
        recipe_dir = self.output_dir / filename

        # Handle duplicate directory names
        counter = 1
        while recipe_dir.exists():
            recipe_dir = self.output_dir / f"{filename}_{counter}"
            counter += 1

        recipe_dir.mkdir(parents=True, exist_ok=True)

        # Create paths for both formats
        txt_path = recipe_dir / f"{filename}.txt"
        json_path = recipe_dir / f"{filename}.json"

        # Save text version
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(recipe.to_text())

        # Save JSON version
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(recipe.model_dump(), indent=2))

        return str(recipe_dir)

    def update_recipe_files(self, recipe: Recipe, recipe_dir: str) -> None:
        """Update existing recipe files with new data (e.g., after adding images)."""
        recipe_path = Path(recipe_dir)

        # Update text version
        txt_path = recipe_path / f"{recipe_path.name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(recipe.to_text())

        # Update JSON version
        json_path = recipe_path / f"{recipe_path.name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(recipe.model_dump(), indent=2))

    def get_recipe_dirname(self, recipe_name: str) -> str:
        """Get a safe directory name for a recipe."""
        safe_name = "".join(
            c for c in recipe_name if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        return safe_name.replace(" ", "_").lower()

    def save_recipes_batch(self, recipes: List[Recipe], batch_name: str = None) -> str:
        """Save multiple recipes to a single file in the root output directory."""
        if not batch_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_name = f"recipes_batch_{timestamp}.txt"

        filepath = self.output_dir / batch_name

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("Recipe Collection\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total recipes: {len(recipes)}\n")
            f.write("=" * 70 + "\n\n")

            for i, recipe in enumerate(recipes, 1):
                if i > 1:
                    f.write("\n\n")
                f.write(f"[Recipe {i} of {len(recipes)}]\n")
                f.write(recipe.to_text())

        return str(filepath)

    def create_index(self, recipes: List[Recipe], index_name: str = "index.txt") -> str:
        """Create an index file listing all recipes in the root output directory."""
        filepath = self.output_dir / index_name

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("Recipe Index\n")
            f.write("=" * 50 + "\n\n")

            for i, recipe in enumerate(recipes, 1):
                f.write(f"{i}. {recipe.name}\n")
                if recipe.description:
                    f.write(
                        f"   {recipe.description[:100]}{'...' if len(recipe.description) > 100 else ''}\n"
                    )
                if recipe.source:
                    f.write(f"   Source: {recipe.source}\n")
                f.write("\n")

        return str(filepath)
