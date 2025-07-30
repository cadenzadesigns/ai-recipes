from typing import Optional

import requests
from bs4 import BeautifulSoup


class WebExtractor:
    """Extract recipe content from web pages."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def extract_from_url(self, url: str) -> str:
        """Extract text content from a web page."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()

            # Try to find recipe-specific content first
            recipe_content = self._extract_recipe_content(soup)
            if recipe_content:
                return recipe_content

            # Fallback to general content extraction
            content_areas = [
                soup.find('main'),
                soup.find('article'),
                soup.find('div', class_=['recipe', 'recipe-content', 'recipe-card']),
                soup.find('div', id=['recipe', 'content', 'main-content'])
            ]

            for area in content_areas:
                if area:
                    text = area.get_text(separator='\n', strip=True)
                    if len(text) > 100:  # Ensure we have substantial content
                        return f"Content from {url}:\n\n{text}"

            # Final fallback - get all text
            text = soup.get_text(separator='\n', strip=True)
            return f"Content from {url}:\n\n{text}"

        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch URL {url}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to parse content from {url}: {str(e)}")

    def _extract_recipe_content(self, soup: BeautifulSoup) -> Optional[str]:
        """Try to extract structured recipe data if available."""

        # Look for JSON-LD structured data
        json_ld = soup.find('script', type='application/ld+json')
        if json_ld:
            try:
                import json
                data = json.loads(json_ld.string)
                if isinstance(data, dict) and data.get('@type') == 'Recipe':
                    return self._format_structured_recipe(data)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get('@type') == 'Recipe':
                            return self._format_structured_recipe(item)
            except Exception:
                pass

        # Look for common recipe microdata
        recipe_card = soup.find(['div', 'article'], itemtype=lambda x: x and 'Recipe' in x)
        if recipe_card:
            content_parts = []

            # Extract title
            title = recipe_card.find(['h1', 'h2'], itemprop='name')
            if title:
                content_parts.append(f"Recipe: {title.get_text(strip=True)}")

            # Extract description
            desc = recipe_card.find(itemprop='description')
            if desc:
                content_parts.append(f"Description: {desc.get_text(strip=True)}")

            # Extract ingredients
            ingredients = recipe_card.find_all(itemprop='recipeIngredient')
            if ingredients:
                content_parts.append("\nIngredients:")
                for ing in ingredients:
                    content_parts.append(f"- {ing.get_text(strip=True)}")

            # Extract instructions
            instructions = recipe_card.find_all(itemprop='recipeInstructions')
            if instructions:
                content_parts.append("\nInstructions:")
                for i, inst in enumerate(instructions, 1):
                    content_parts.append(f"{i}. {inst.get_text(strip=True)}")

            if content_parts:
                return '\n'.join(content_parts)

        return None

    def _format_structured_recipe(self, data: dict) -> str:
        """Format structured recipe data into text."""
        parts = []

        if data.get('name'):
            parts.append(f"Recipe: {data['name']}")

        if data.get('description'):
            parts.append(f"Description: {data['description']}")

        if data.get('recipeIngredient'):
            parts.append("\nIngredients:")
            for ing in data['recipeIngredient']:
                parts.append(f"- {ing}")

        if data.get('recipeInstructions'):
            parts.append("\nInstructions:")
            instructions = data['recipeInstructions']
            if isinstance(instructions, list):
                for i, inst in enumerate(instructions, 1):
                    if isinstance(inst, dict):
                        text = inst.get('text', inst.get('name', ''))
                    else:
                        text = str(inst)
                    parts.append(f"{i}. {text}")
            else:
                parts.append(str(instructions))

        return '\n'.join(parts)
