import json
import re
from pathlib import Path
from typing import Dict

from .models import Recipe


class RecipeConverter:
    """Convert between different recipe formats."""

    @staticmethod
    def txt_to_recipe(txt_content: str) -> Recipe:
        """Parse a text file and convert it to a Recipe object."""
        lines = txt_content.strip().split('\n')

        # Initialize recipe data
        recipe_data = {
            "name": "",
            "description": "",
            "servings": None,
            "total_time": None,
            "ingredients": [],
            "directions": [],
            "notes": [],
            "source": None
        }

        current_section = None
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and separators
            if not line or line == "=" * len(line) or line == "-" * len(line):
                i += 1
                continue

            # Extract recipe name (first non-empty line)
            if not recipe_data["name"] and not line.startswith('['):
                recipe_data["name"] = line
                i += 1
                continue

            # Check for section headers
            if line == "DESCRIPTION:":
                current_section = "description"
                i += 1
                continue
            elif line.startswith("SERVINGS:"):
                recipe_data["servings"] = line.replace("SERVINGS:", "").strip()
                i += 1
                continue
            elif line.startswith("TOTAL TIME:"):
                recipe_data["total_time"] = line.replace("TOTAL TIME:", "").strip()
                i += 1
                continue
            elif line == "INGREDIENTS:":
                current_section = "ingredients"
                i += 1
                continue
            elif line == "DIRECTIONS:":
                current_section = "directions"
                i += 1
                continue
            elif line == "NOTES:":
                current_section = "notes"
                i += 1
                continue
            elif line.startswith("SOURCE:"):
                recipe_data["source"] = line.replace("SOURCE:", "").strip()
                current_section = None
                i += 1
                continue

            # Process content based on current section
            if current_section == "description":
                # Description is usually a single paragraph
                if line and not line.endswith(':'):
                    recipe_data["description"] = line
                    current_section = None

            elif current_section == "ingredients":
                # Ingredients start with bullet or dash
                if line.startswith(('•', '-', '*', '·')):
                    ingredient = re.sub(r'^[•\-*·]\s*', '', line).strip()
                    recipe_data["ingredients"].append(ingredient)
                elif line and line[0].isdigit():
                    # Some formats might use numbers
                    recipe_data["ingredients"].append(line)

            elif current_section == "directions":
                # Directions are numbered
                match = re.match(r'^\d+\.\s*(.+)$', line)
                if match:
                    recipe_data["directions"].append(match.group(1))
                elif line and not line.endswith(':'):
                    # Sometimes directions might not be numbered
                    recipe_data["directions"].append(line)

            elif current_section == "notes":
                # Notes can be bullet points or paragraphs
                if line.startswith(('•', '-', '*', '·')):
                    note = re.sub(r'^[•\-*·]\s*', '', line).strip()
                    recipe_data["notes"].append(note)
                elif line and not line.endswith(':'):
                    recipe_data["notes"].append(line)

            i += 1

        # Clean up empty values
        if not recipe_data["description"]:
            recipe_data["description"] = ""
        if not recipe_data["notes"]:
            recipe_data["notes"] = None

        return Recipe(**recipe_data)

    @staticmethod
    def convert_txt_file_to_json(txt_path: Path, json_path: Path) -> bool:
        """Convert a single txt file to JSON format."""
        try:
            # Read the text file
            with open(txt_path, encoding='utf-8') as f:
                txt_content = f.read()

            # Convert to Recipe object
            recipe = RecipeConverter.txt_to_recipe(txt_content)

            # Save as JSON
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(recipe.model_dump(), f, indent=2)

            return True

        except Exception as e:
            print(f"Error converting {txt_path}: {str(e)}")
            return False

    @staticmethod
    def batch_convert(txt_dir: Path, json_dir: Path) -> Dict[str, int]:
        """Convert all txt files in a directory to JSON format."""
        stats = {"success": 0, "failed": 0, "skipped": 0}

        # Find all txt files
        txt_files = list(txt_dir.glob("*.txt"))

        for txt_file in txt_files:
            # Skip index files
            if txt_file.name in ['index.txt', 'recipes_batch_*.txt']:
                stats["skipped"] += 1
                continue

            # Determine output path
            json_file = json_dir / txt_file.with_suffix('.json').name

            # Skip if JSON already exists
            if json_file.exists():
                stats["skipped"] += 1
                continue

            # Convert the file
            if RecipeConverter.convert_txt_file_to_json(txt_file, json_file):
                stats["success"] += 1
            else:
                stats["failed"] += 1

        return stats
