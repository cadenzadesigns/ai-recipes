import os
from typing import Any, Dict, List, Union

from openai import OpenAI

from .models import Recipe


class LLMClient:
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable"
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def extract_recipe(
        self, content: Union[str, List[Dict[str, Any]]], source: str = None
    ) -> Recipe:
        """Extract recipe from content using LLM."""

        system_prompt = """You are a recipe extraction assistant. Your task is to extract recipe information from the provided content.

Extract the following information:
- Name: The recipe title
- Description: The EXACT description text as it appears in the source, typically found after the title and before ingredients. Copy this verbatim - do not paraphrase or summarize.
- Servings: Number of servings or yield (e.g., "4 servings", "12 cookies", "1 9-inch pie")
- Total_time: Total time ONLY if explicitly stated in the recipe (e.g., "45 minutes", "1 hour 30 minutes"). Do NOT calculate or infer this - only include if the source explicitly states total time. Leave as null if not provided.
- Ingredients: Extract each ingredient as a structured object with:
  - amount: Object containing:
    - quantity: The numeric amount (like "1", "2.5", "1/2", "1-2")
    - unit: The measurement unit (like "cup", "tablespoon", "pound", "ounce")
  - item: Object containing:
    - name: The core ingredient name ONLY (like "salt", "cumin", "flour", "butter", "jalapeños")
    - modifiers: List of descriptors/specifications (like ["kosher"], ["ground"], ["all-purpose"], ["unsalted", "softened"])
    - alternative: If there's a substitution mentioned (like "or serrano peppers"), create another item object for the alternative
  IMPORTANT:
  - Separate the base ingredient name from its modifiers (e.g., "kosher salt" → name: "salt", modifiers: ["kosher"])
  - Use standard fractions like 1/2, 1/3, 1/4 instead of Unicode fraction characters (½, ⅓, ¼)
- Directions: Step-by-step cooking instructions, each step on a new line
- Notes: Any additional tips, variations, storage instructions, or important information (as an array, one note per item)
- Source: The source of the recipe (if not provided, leave empty)

Be thorough and accurate. If information is missing, use null for optional fields."""

        messages = [{"role": "system", "content": system_prompt}]

        if isinstance(content, str):
            # Text content
            user_message = {"role": "user", "content": content}
        else:
            # Image content (list of image dictionaries)
            user_message = {"role": "user", "content": content}

        messages.append(user_message)

        try:
            # Use structured output with Pydantic model
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                temperature=0.1,
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
                recipes.append(recipe)
            except Exception as e:
                print(f"Failed to extract recipe: {str(e)}")
                continue

        return recipes
