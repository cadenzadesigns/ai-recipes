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
        """Save a single recipe to a text file."""
        if not filename:
            # Generate filename from recipe name
            safe_name = "".join(c for c in recipe.name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_').lower()
            filename = f"{safe_name}.txt"

        filepath = self.output_dir / filename

        # Handle duplicate filenames
        if filepath.exists():
            base = filepath.stem
            suffix = filepath.suffix
            counter = 1
            while filepath.exists():
                filepath = self.output_dir / f"{base}_{counter}{suffix}"
                counter += 1

        # Write recipe to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(recipe.to_text())

        return str(filepath)

    def save_recipes_batch(self, recipes: List[Recipe], batch_name: str = None) -> str:
        """Save multiple recipes to a single file."""
        if not batch_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_name = f"recipes_batch_{timestamp}.txt"

        filepath = self.output_dir / batch_name

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
        filepath = self.output_dir / index_name

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
