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

        # Create subdirectories for txt and json
        self.txt_dir = self.output_dir / "txt"
        self.json_dir = self.output_dir / "json"
        self.txt_dir.mkdir(exist_ok=True)
        self.json_dir.mkdir(exist_ok=True)

    def save_recipe(self, recipe: Recipe, filename: str = None) -> str:
        """Save a recipe to both text and JSON files."""
        if not filename:
            # Generate filename from recipe name
            safe_name = "".join(c for c in recipe.name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_').lower()
            filename = safe_name

        # Create paths for both formats
        txt_path = self.txt_dir / f"{filename}.txt"
        json_path = self.json_dir / f"{filename}.json"

        # Handle duplicate filenames
        counter = 1
        while txt_path.exists() or json_path.exists():
            txt_path = self.txt_dir / f"{filename}_{counter}.txt"
            json_path = self.json_dir / f"{filename}_{counter}.json"
            counter += 1

        # Save text version
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(recipe.to_text())

        # Save JSON version
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(recipe.model_dump(), indent=2))

        return str(txt_path)

    def save_recipes_batch(self, recipes: List[Recipe], batch_name: str = None) -> str:
        """Save multiple recipes to a single file."""
        if not batch_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_name = f"recipes_batch_{timestamp}.txt"

        filepath = self.txt_dir / batch_name

        with open(filepath, 'w', encoding='utf-8') as f:
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
        """Create an index file listing all recipes."""
        filepath = self.txt_dir / index_name

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("Recipe Index\n")
            f.write("=" * 50 + "\n\n")

            for i, recipe in enumerate(recipes, 1):
                f.write(f"{i}. {recipe.name}\n")
                if recipe.description:
                    f.write(f"   {recipe.description[:100]}{'...' if len(recipe.description) > 100 else ''}\n")
                if recipe.source:
                    f.write(f"   Source: {recipe.source}\n")
                f.write("\n")

        return str(filepath)
