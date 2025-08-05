"""Generate HTML pages for recipes with navigation."""

import json
from pathlib import Path
from typing import Dict, List

from src.models import Recipe


class RecipeHTMLGenerator:
    """Generate HTML pages for recipes with a table of contents."""

    def __init__(self, recipes_dir: str = "recipes"):
        self.recipes_dir = Path(recipes_dir)
        self.html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .metadata-item {{
            display: inline-block;
            margin-right: 20px;
            font-weight: bold;
        }}
        .description {{
            font-style: italic;
            color: #555;
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
        }}
        .ingredients {{
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .ingredient-item {{
            margin-bottom: 8px;
            padding-left: 20px;
        }}
        .directions {{
            background-color: #fffaf0;
            padding: 20px;
            border-radius: 5px;
        }}
        .direction-step {{
            margin-bottom: 15px;
            padding-left: 30px;
            position: relative;
        }}
        .direction-step::before {{
            content: attr(data-step);
            position: absolute;
            left: 0;
            font-weight: bold;
            color: #3498db;
        }}
        .notes {{
            background-color: #fff9e6;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }}
        .image-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .recipe-image {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .image-caption {{
            font-size: 0.9em;
            color: #666;
            margin-top: 8px;
            font-style: italic;
        }}
        .step-images {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .step-image-container {{
            text-align: center;
        }}
        .step-image {{
            width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .back-link {{
            display: inline-block;
            margin-bottom: 20px;
            color: #3498db;
            text-decoration: none;
            font-weight: bold;
        }}
        .back-link:hover {{
            text-decoration: underline;
        }}
        .component {{
            margin-bottom: 25px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
        .component-title {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        {content}
    </div>
</body>
</html>
"""

        self.index_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recipe Collection</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .subtitle {{
            color: #7f8c8d;
            font-size: 1.2em;
        }}
        .recipe-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }}
        .recipe-card {{
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
        }}
        .recipe-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }}
        .recipe-image {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            background-color: #ecf0f1;
        }}
        .no-image {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 200px;
            background-color: #ecf0f1;
            color: #95a5a6;
            font-size: 3em;
        }}
        .recipe-content {{
            padding: 20px;
        }}
        .recipe-title {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .recipe-description {{
            color: #666;
            font-size: 0.95em;
            margin-bottom: 15px;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}
        .recipe-meta {{
            display: flex;
            gap: 20px;
            font-size: 0.9em;
            color: #95a5a6;
        }}
        .meta-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        a {{
            text-decoration: none;
            color: inherit;
        }}
        .search-container {{
            max-width: 600px;
            margin: 0 auto 30px;
        }}
        .search-input {{
            width: 100%;
            padding: 12px 20px;
            font-size: 1.1em;
            border: 2px solid #ecf0f1;
            border-radius: 25px;
            outline: none;
            transition: border-color 0.3s;
        }}
        .search-input:focus {{
            border-color: #3498db;
        }}
        .no-results {{
            text-align: center;
            color: #7f8c8d;
            font-size: 1.2em;
            margin-top: 50px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🍳 Recipe Collection</h1>
        <p class="subtitle">Browse your extracted recipes</p>
    </div>

    <div class="search-container">
        <input type="text" class="search-input" id="searchInput" placeholder="Search recipes...">
    </div>

    <div class="recipe-grid" id="recipeGrid">
        {recipe_cards}
    </div>

    <div class="no-results" id="noResults" style="display: none;">
        No recipes found matching your search.
    </div>

    <script>
        const searchInput = document.getElementById('searchInput');
        const recipeGrid = document.getElementById('recipeGrid');
        const noResults = document.getElementById('noResults');
        const allCards = recipeGrid.querySelectorAll('.recipe-card');

        searchInput.addEventListener('input', function() {{
            const searchTerm = this.value.toLowerCase();
            let hasResults = false;

            allCards.forEach(card => {{
                const title = card.querySelector('.recipe-title').textContent.toLowerCase();
                const description = card.querySelector('.recipe-description').textContent.toLowerCase();

                if (title.includes(searchTerm) || description.includes(searchTerm)) {{
                    card.style.display = 'block';
                    hasResults = true;
                }} else {{
                    card.style.display = 'none';
                }}
            }});

            noResults.style.display = hasResults ? 'none' : 'block';
        }});
    </script>
</body>
</html>
"""

    def generate_recipe_html(self, recipe: Recipe, recipe_dir: Path) -> str:
        """Generate HTML content for a single recipe."""
        content_parts = []

        # Back link - go back to the main app
        content_parts.append(
            '<a href="/" class="back-link">← Back to Recipe Collection</a>'
        )

        # Title
        title = recipe.name
        if recipe.alternate_name:
            title += f" ({recipe.alternate_name})"
        content_parts.append(f"<h1>{title}</h1>")

        # Metadata
        if recipe.servings or recipe.total_time:
            content_parts.append('<div class="metadata">')
            if recipe.servings:
                content_parts.append(
                    f'<span class="metadata-item">🍽️ {recipe.servings}</span>'
                )
            if recipe.total_time:
                content_parts.append(
                    f'<span class="metadata-item">⏱️ {recipe.total_time}</span>'
                )
            content_parts.append("</div>")

        # Description
        if recipe.description:
            content_parts.append(f'<div class="description">{recipe.description}</div>')

        # Main image
        if recipe.images:
            main_images = [img for img in recipe.images if img.is_main]
            if main_images:
                img = main_images[0]
                img_path = f"images/{img.filename}"
                if (recipe_dir / "images" / img.filename).exists():
                    content_parts.append('<div class="image-container">')
                    content_parts.append(
                        f'<img src="{img_path}" alt="{img.description or "Recipe image"}" class="recipe-image">'
                    )
                    if img.description:
                        content_parts.append(
                            f'<p class="image-caption">{img.description}</p>'
                        )
                    content_parts.append("</div>")

        # Ingredients
        content_parts.append('<div class="ingredients">')
        content_parts.append("<h2>Ingredients</h2>")

        if recipe.components:
            # Recipe has components
            for component in recipe.components:
                content_parts.append('<div class="component">')
                content_parts.append(
                    f'<div class="component-title">{component.title}:</div>'
                )
                for ingredient in component.ingredients:
                    content_parts.append(
                        f'<div class="ingredient-item">• {ingredient.to_string()}</div>'
                    )
                content_parts.append("</div>")
        elif recipe.ingredients:
            # Simple ingredient list
            for ingredient in recipe.ingredients:
                content_parts.append(
                    f'<div class="ingredient-item">• {ingredient.to_string()}</div>'
                )

        content_parts.append("</div>")

        # Directions
        content_parts.append('<div class="directions">')
        content_parts.append("<h2>Directions</h2>")
        for i, direction in enumerate(recipe.directions, 1):
            content_parts.append(
                f'<div class="direction-step" data-step="{i}.">{direction}</div>'
            )
        content_parts.append("</div>")

        # Notes
        if recipe.notes:
            content_parts.append('<div class="notes">')
            content_parts.append("<h3>Notes</h3>")
            for note in recipe.notes:
                content_parts.append(f"<p>{note}</p>")
            content_parts.append("</div>")

        # Step-by-step images
        if recipe.images:
            step_images = [img for img in recipe.images if img.is_step]
            if step_images:
                content_parts.append("<h3>Step-by-Step Photos</h3>")
                content_parts.append('<div class="step-images">')
                for img in step_images:
                    img_path = f"images/{img.filename}"
                    if (recipe_dir / "images" / img.filename).exists():
                        content_parts.append('<div class="step-image-container">')
                        content_parts.append(
                            f'<img src="{img_path}" alt="{img.description or "Step photo"}" class="step-image">'
                        )
                        if img.description:
                            content_parts.append(
                                f'<p class="image-caption">{img.description}</p>'
                            )
                        content_parts.append("</div>")
                content_parts.append("</div>")

        # Source
        if recipe.source:
            content_parts.append(
                f'<p style="margin-top: 30px; color: #7f8c8d;"><em>Source: {recipe.source}</em></p>'
            )

        return self.html_template.format(title=title, content="\n".join(content_parts))

    def generate_index_html(self, recipes: List[Dict]) -> str:
        """Generate HTML content for the recipe index."""
        recipe_cards = []

        for recipe_info in recipes:
            recipe = recipe_info["recipe"]
            recipe_dir = recipe_info["dir_name"]

            # Find main image
            main_image = None
            if recipe.images:
                main_images = [img for img in recipe.images if img.is_main]
                if main_images:
                    main_image = main_images[0]
                elif recipe.images:  # Use first image if no main image
                    main_image = recipe.images[0]

            # Create card HTML
            card_parts = [f'<a href="/recipes/{recipe_dir}">']
            card_parts.append('<div class="recipe-card">')

            # Image or placeholder
            if (
                main_image
                and (
                    Path(self.recipes_dir) / recipe_dir / "images" / main_image.filename
                ).exists()
            ):
                img_url = f"/recipes/{recipe_dir}/images/{main_image.filename}"
                card_parts.append(
                    f'<img src="{img_url}" alt="{recipe.name}" class="recipe-image">'
                )
            else:
                card_parts.append('<div class="no-image">🍽️</div>')

            # Content
            card_parts.append('<div class="recipe-content">')
            card_parts.append(f'<div class="recipe-title">{recipe.name}</div>')

            if recipe.description:
                description = (
                    recipe.description[:150] + "..."
                    if len(recipe.description) > 150
                    else recipe.description
                )
                card_parts.append(
                    f'<div class="recipe-description">{description}</div>'
                )

            # Metadata
            card_parts.append('<div class="recipe-meta">')
            if recipe.servings:
                card_parts.append(f'<div class="meta-item">🍽️ {recipe.servings}</div>')
            if recipe.total_time:
                card_parts.append(f'<div class="meta-item">⏱️ {recipe.total_time}</div>')
            card_parts.append("</div>")

            card_parts.append("</div>")  # recipe-content
            card_parts.append("</div>")  # recipe-card
            card_parts.append("</a>")

            recipe_cards.append("\n".join(card_parts))

        return self.index_template.format(recipe_cards="\n".join(recipe_cards))

    def load_recipes(self) -> List[Dict]:
        """Load all recipes from the recipes directory."""
        recipes = []

        if not self.recipes_dir.exists():
            return recipes

        # Iterate through recipe directories
        for recipe_dir in self.recipes_dir.iterdir():
            if recipe_dir.is_dir():
                # Look for JSON file
                json_files = list(recipe_dir.glob("*.json"))
                if json_files:
                    try:
                        with open(json_files[0], encoding="utf-8") as f:
                            recipe_data = json.load(f)
                            recipe = Recipe(**recipe_data)
                            recipes.append(
                                {
                                    "recipe": recipe,
                                    "dir_name": recipe_dir.name,
                                    "dir_path": recipe_dir,
                                }
                            )
                    except Exception as e:
                        print(f"Error loading recipe from {recipe_dir}: {e}")

        # Sort by recipe name
        recipes.sort(key=lambda x: x["recipe"].name.lower())

        return recipes

    def save_recipe_html(self, recipe: Recipe, recipe_dir: Path) -> None:
        """Save HTML file for a recipe."""
        html_content = self.generate_recipe_html(recipe, recipe_dir)
        html_path = recipe_dir / "index.html"

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def save_all_recipe_htmls(self) -> None:
        """Generate and save HTML files for all recipes."""
        recipes = self.load_recipes()

        for recipe_info in recipes:
            recipe = recipe_info["recipe"]
            recipe_dir = recipe_info["dir_path"]
            self.save_recipe_html(recipe, recipe_dir)
