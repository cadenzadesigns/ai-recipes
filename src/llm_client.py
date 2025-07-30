import json
import os
from typing import Any, Dict, List, Union

from openai import OpenAI

from .models import Recipe


class LLMClient:
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def extract_recipe(self, content: Union[str, List[Dict[str, Any]]], source: str = None) -> Recipe:
        """Extract recipe from content using LLM."""

        system_prompt = """You are a recipe extraction assistant. Your task is to extract recipe information from the provided content and format it according to the given schema.

Extract the following information:
- Name: The recipe title
- Description: The EXACT description text as it appears in the source, typically found after the title and before ingredients. Copy this verbatim - do not paraphrase or summarize.
- Servings: Number of servings or yield (e.g., "4 servings", "12 cookies", "1 9-inch pie")
- Total_time: Total time including prep and cooking (e.g., "45 minutes", "1 hour 30 minutes")
- Ingredients: List each ingredient with its quantity on a separate line
- Directions: Step-by-step cooking instructions, each step on a new line
- Notes: Any additional tips, variations, storage instructions, or important information
- Source: The source of the recipe (if not provided, leave empty)

Format your response as a valid JSON object with these exact keys: name, description, servings, total_time, ingredients (array), directions (array), notes, source.

Be thorough and accurate. If information is missing, use null for optional fields."""

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        if isinstance(content, str):
            # Text content
            user_message = {"role": "user", "content": content}
        else:
            # Image content (list of image dictionaries)
            user_message = {
                "role": "user",
                "content": content
            }

        messages.append(user_message)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        try:
            recipe_data = json.loads(response.choices[0].message.content)

            # Add source if provided
            if source and not recipe_data.get("source"):
                recipe_data["source"] = source

            return Recipe(**recipe_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse LLM response: {str(e)}")

    def extract_recipes_batch(self, contents: List[Union[str, List[Dict[str, Any]]]], sources: List[str] = None) -> List[Recipe]:
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
