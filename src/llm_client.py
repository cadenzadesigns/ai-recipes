import os
from typing import Any, Dict, List, Union

from openai import OpenAI

from .config import OPENAI_MODEL
from .models import Recipe


class LLMClient:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable"
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model or OPENAI_MODEL

    def _clean_ingredient_name(self, name: str) -> str:
        """Remove bullet points and clean up ingredient names."""
        if not name:
            return name

        # Common bullet point patterns to remove
        bullet_patterns = [
            "• ",
            "· ",
            "- ",
            "* ",
            "• ",
            "· ",
            "− ",
            "∙ ",
            "◦ ",
            "▪ ",
            "▫ ",
        ]

        cleaned = name
        for pattern in bullet_patterns:
            if cleaned.startswith(pattern):
                cleaned = cleaned[len(pattern) :]

        # Also strip any leading/trailing whitespace
        return cleaned.strip()

    def _clean_recipe_ingredients(self, recipe: Recipe) -> Recipe:
        """Clean all ingredient names in a recipe to remove bullet points."""
        # Clean ingredients in the main ingredients list
        if recipe.ingredients:
            for ingredient in recipe.ingredients:
                if ingredient.item and ingredient.item.name:
                    ingredient.item.name = self._clean_ingredient_name(
                        ingredient.item.name
                    )

        # Clean ingredients in components
        if recipe.components:
            for component in recipe.components:
                if component.ingredients:
                    for ingredient in component.ingredients:
                        if ingredient.item and ingredient.item.name:
                            ingredient.item.name = self._clean_ingredient_name(
                                ingredient.item.name
                            )

        return recipe

    def extract_recipe(
        self, content: Union[str, List[Dict[str, Any]]], source: str = None
    ) -> Recipe:
        """Extract recipe from content using LLM."""

        system_prompt = """You are a recipe extraction assistant. Your task is to extract recipe information from the provided content EXACTLY as it appears in the source.

CRITICAL: DO NOT ADD ANY FORMATTING that wasn't in the original:
- DO NOT add bullet points (•, -, *) to ingredients if they weren't there originally
- DO NOT add "Step 1:", "Step 2:", etc. to directions if they weren't numbered in the original
- PRESERVE the exact formatting and text as it appears in the source

Extract the following information:
- Name: The recipe title
- Alternate_name: If there's a subtitle or alternate name provided in parentheses or as a secondary title, capture it here
- Description: The EXACT description text as it appears in the source, typically found after the title and before ingredients. Copy this verbatim - do not paraphrase or summarize.
- Servings: Number of servings or yield (e.g., "4 servings", "12 cookies", "1 9-inch pie")
- Total_time: Total time ONLY if explicitly stated in the recipe (e.g., "45 minutes", "1 hour 30 minutes"). Do NOT calculate or infer this - only include if the source explicitly states total time. Leave as null if not provided.

For Ingredients - Check if the recipe has multiple components:
- If the recipe has distinct sections with titles like "For the Dough", "For the Filling", "Frosting", etc.:
  Use 'components' field (leave 'ingredients' as null):
  - Each component should have:
    - title: The component title (e.g., "For the Dough", "Filling", "Glaze")
    - ingredients: List of ingredients for that component
- If the recipe has a single unified ingredient list:
  Use 'ingredients' field (leave 'components' as null)

For each ingredient (whether in components or main ingredients), extract as a structured object with:
  - amount: Object containing:
    - quantity: The numeric amount (like "1", "2.5", "1/2", "1-2")
    - unit: The measurement unit (like "cup", "tablespoon", "pound", "ounce")
    - metric_quantity: If a metric measurement is also provided (like "200" for 200g)
    - metric_unit: The metric unit if provided (like "g", "ml", "kg", "L")
  - item: Object containing:
    - name: The core ingredient name ONLY - DO NOT include any bullet points (•, -, *) even if they appear in the source
    - modifiers: List of descriptors/specifications that come AFTER the ingredient name
    - alternative: If there's a substitution mentioned (like "or serrano peppers"), create another item object for the alternative

CRITICAL - Follow this EXACT parsing order: AMOUNT → ITEM NAME → DESCRIPTORS/MODIFIERS
Examples of correct parsing:
  * "1 pound brined, cut into nuggets chicken thighs" → quantity="1", unit="pound", name="chicken thighs", modifiers=["brined", "cut into nuggets"]
  * "1/2 teaspoon ground, white pepper" → quantity="1/2", unit="teaspoon", name="pepper", modifiers=["white", "ground"]
  * "474 g heavy cream" → metric_quantity="474", metric_unit="g", name="heavy cream"
  * "150 g butter, unsalted, cold, cubed" → metric_quantity="150", metric_unit="g", name="butter", modifiers=["unsalted", "cold", "cubed"]
  * "2 green onions, cut on an angle into 2-inch slices" → quantity="2", name="green onions", modifiers=["cut on an angle into 2-inch slices"]
  * "604 g ricotta, sheep's milk" → metric_quantity="604", metric_unit="g", name="ricotta", modifiers=["sheep's milk"]
  * "2 cloves garlic, thinly sliced" → quantity="2", unit="cloves", name="garlic", modifiers=["thinly sliced"]

IMPORTANT:
  - The ingredient NAME comes IMMEDIATELY after the amount, BEFORE any descriptors
  - ALL descriptive text (preparation, quality, type, etc.) goes in modifiers
  - Use standard fractions like 1/2, 1/3, 1/4 instead of Unicode fraction characters (½, ⅓, ¼)
  - For dual measurements, capture BOTH regardless of order. Common formats include:
    * "220g / 1 cup white wine" → metric_quantity="220", metric_unit="g", quantity="1", unit="cup", name="white wine"
    * "1 cup / 220g white wine" → quantity="1", unit="cup", metric_quantity="220", metric_unit="g", name="white wine"
    * "1 cup (240ml) milk" → quantity="1", unit="cup", metric_quantity="240", metric_unit="ml", name="milk"
    * "200g (1 cup) flour" → metric_quantity="200", metric_unit="g", quantity="1", unit="cup", name="flour"
    * "8 ounces (225g) butter" → quantity="8", unit="ounces", metric_quantity="225", metric_unit="g", name="butter"
  - Identify metric vs standard units correctly:
    * Metric units: g, kg, ml, L, cl, dl (grams, kilograms, milliliters, liters, centiliters, deciliters)
    * Standard/Imperial units: cup, tablespoon, teaspoon, ounce, pound, quart, pint, gallon, stick, etc.
  - Convert any diacritical marks to their ASCII equivalents (e.g., "jalapeños" → "jalapenos", "crème" → "creme")
  - Use EITHER 'components' OR 'ingredients', never both

- Directions: Extract the cooking instructions EXACTLY as they appear in the source:
  * If the original has numbered steps (1., 2., etc.), keep those numbers
  * If the original has no numbers, DO NOT add them
  * Each instruction should be on a new line
  * DO NOT add "Step 1:", "Step 2:" formatting unless it's in the original
  * DO NOT reword or paraphrase - copy the exact text
- Notes: Any additional tips, variations, storage instructions, or important information (as an array, one note per item)
- Source: The source of the recipe (if not provided, leave empty)

Be thorough and accurate. Copy text EXACTLY as it appears. If information is missing, use null for optional fields."""

        messages = [{"role": "system", "content": system_prompt}]

        if isinstance(content, str):
            # Text content
            user_message = {"role": "user", "content": content}
        else:
            # Image content (list of image dictionaries)
            user_message = {"role": "user", "content": content}

        messages.append(user_message)

        try:
            # Use temperature 1.0 for o4-mini model (required), 0.1 for others
            temperature = 1.0 if self.model == "o4-mini" else 0.1

            # Use structured output with Pydantic model
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                temperature=temperature,
                response_format=Recipe,
            )

            # Get the parsed recipe directly
            recipe = response.choices[0].message.parsed

            if recipe is None:
                # Check if it was a refusal
                if response.choices[0].message.refusal:
                    raise ValueError(
                        f"No recipe found: {response.choices[0].message.refusal}"
                    )
                else:
                    raise ValueError("LLM returned empty response")

            # Clean any bullet points from ingredient names
            recipe = self._clean_recipe_ingredients(recipe)

            # Add source if provided and not already set
            if source and not recipe.source:
                recipe.source = source

            return recipe

        except Exception as e:
            # Handle parsing errors or other exceptions
            if "No recipe found" in str(e):
                raise
            raise ValueError(f"Failed to extract recipe: {str(e)}")

    def extract_recipes_batch(
        self,
        contents: List[Union[str, List[Dict[str, Any]]]],
        sources: List[str] = None,
    ) -> List[Recipe]:
        """Extract multiple recipes from a batch of contents."""
        recipes = []
        sources = sources or [None] * len(contents)

        for content, source in zip(contents, sources):
            try:
                recipe = self.extract_recipe(content, source)
                # Note: recipe is already cleaned in extract_recipe
                recipes.append(recipe)
            except Exception as e:
                print(f"Failed to extract recipe: {str(e)}")
                continue

        return recipes
