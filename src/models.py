from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator


class Amount(BaseModel):
    """Represents the amount/quantity for an ingredient."""

    quantity: str | None = Field(
        default=None,
        description="The numeric quantity (e.g., '1', '2.5', '1/2', '1-2'). This should ONLY be numbers or numeric expressions, never descriptive text like 'for serving'.",
    )
    unit: str | None = Field(
        default=None,
        description="The unit of measurement (e.g., 'cup', 'tablespoon', 'pound'). This should ONLY be actual units of measurement, not usage instructions.",
    )

    @field_validator("quantity", mode="before")
    @classmethod
    def convert_amount_fractions(cls, v: str | None) -> str | None:
        """Convert Unicode fractions in quantities to readable format."""
        return convert_unicode_fractions(v) if v else v


class Item(BaseModel):
    """Represents the ingredient item with name and modifiers."""

    name: str = Field(
        description="The main ingredient name (e.g., 'flour', 'butter', 'eggs', 'salt', 'cumin', 'neutral oil'). This should be the core ingredient, not descriptive phrases."
    )
    modifiers: list[str] | None = Field(
        default=None,
        description="List of modifiers/specifications including preparation notes and usage instructions (e.g., ['all-purpose'], ['unsalted', 'softened'], ['kosher'], ['ground'], ['such as grapeseed or canola', 'for the grill']). Include ALL descriptive text that isn't the core ingredient name.",
    )
    alternative: Item | None = Field(
        default=None,
        description="Alternative ingredient that can be substituted (e.g., 'serrano peppers' instead of 'jalapeños')",
    )

    @field_validator("name", mode="before")
    @classmethod
    def convert_name_fractions(cls, v: str) -> str:
        """Convert Unicode fractions in names to readable format."""
        return convert_unicode_fractions(v) if v else v

    @field_validator("modifiers", mode="before")
    @classmethod
    def convert_modifiers_fractions(cls, v: list[str] | None) -> list[str] | None:
        """Convert Unicode fractions in modifiers to readable format."""
        if isinstance(v, list):
            return [convert_unicode_fractions(mod) for mod in v]
        return v

    def to_string(self) -> str:
        """Convert item to human-readable string."""
        # Name always comes first
        parts = [self.name]
        
        # Then modifiers come after
        if self.modifiers:
            parts.append(", ".join(self.modifiers))
        
        result = ", ".join(parts)

        if self.alternative:
            result += f" (or {self.alternative.to_string()})"

        return result


class Ingredient(BaseModel):
    """Represents a single ingredient with amount and item details."""

    amount: Amount | None = Field(
        default=None, description="The amount/quantity of the ingredient. For ingredients like 'Ritz crackers, for serving' or 'salt to taste', the amount should be None."
    )
    item: Item = Field(description="The ingredient item with name and modifiers. For 'Ritz crackers, for serving', name='Ritz crackers' and modifiers=['for serving'].")

    def to_string(self) -> str:
        """Convert ingredient to human-readable string."""
        if self.amount and self.amount.quantity:
            parts = [self.amount.quantity]
            if self.amount.unit:
                parts.append(self.amount.unit)
            parts.append(self.item.to_string())
            return " ".join(parts)
        return self.item.to_string()


class RecipeImage(BaseModel):
    filename: str = Field(description="Image filename")
    description: str = Field(description="Description of what the image shows")
    is_main: bool = Field(
        default=False, description="Whether this is the main recipe photo"
    )
    is_step: bool = Field(
        default=False, description="Whether this is a step-by-step photo"
    )


def convert_unicode_fractions(text: str) -> str:
    """Convert Unicode fraction characters to their readable equivalents."""
    if not text:
        return text

    # Mapping of Unicode fractions to readable format
    fraction_map = {
        "½": "1/2",
        "⅓": "1/3",
        "⅔": "2/3",
        "¼": "1/4",
        "¾": "3/4",
        "⅕": "1/5",
        "⅖": "2/5",
        "⅗": "3/5",
        "⅘": "4/5",
        "⅙": "1/6",
        "⅚": "5/6",
        "⅐": "1/7",
        "⅛": "1/8",
        "⅜": "3/8",
        "⅝": "5/8",
        "⅞": "7/8",
        "⅑": "1/9",
        "⅒": "1/10",
        "↉": "0/3",
    }

    result = text
    for unicode_frac, readable_frac in fraction_map.items():
        result = result.replace(unicode_frac, readable_frac)

    return result


class Recipe(BaseModel):
    name: str = Field(description="The name of the recipe")
    description: str = Field(
        description="The exact description text from the source document, typically found after the recipe title and before the ingredients list. Copy this verbatim from the source."
    )
    servings: str | None = Field(
        default=None,
        description="Number of servings or yield (e.g., '4 servings', '12 cookies', '1 9-inch pie')",
    )
    total_time: str | None = Field(
        default=None,
        description="Total cook time including prep and cooking (e.g., '45 minutes', '1 hour 30 minutes')",
    )
    ingredients: list[Ingredient] = Field(
        description="List of ingredients with structured amounts and items. Parse each ingredient line as: [amount] [ingredient name], [modifiers]. Examples: '1 pound cooked crabmeat' → amount='1 pound', name='crabmeat', modifiers=['cooked']. 'Ritz crackers, for serving' → no amount, name='Ritz crackers', modifiers=['for serving']. '2 tablespoons fresh, minced chives' → amount='2 tablespoons', name='chives', modifiers=['fresh', 'minced']. The ingredient name is the core item (crabmeat, chives, Ritz crackers), everything else is modifiers."
    )
    directions: list[str] = Field(description="Step-by-step cooking directions")
    notes: list[str] | None = Field(
        default=None,
        description="Additional notes, tips, variations, or storage instructions (e.g., 'Store in airtight container for up to 5 days', 'Can be frozen for 3 months', 'Refrigerate leftovers')",
    )
    source: str | None = Field(
        default=None, description="Source of the recipe (cookbook, website, etc.)"
    )
    images: list[RecipeImage] | None = Field(
        default=None, description="Associated recipe images"
    )

    @field_validator("ingredients", mode="before")
    @classmethod
    def parse_ingredients(
        cls, v: list[str] | list[dict] | list[Ingredient]
    ) -> list[Ingredient]:
        """Parse ingredients from various input formats."""
        if not isinstance(v, list):
            return v

        result = []
        for item in v:
            # If it's already an Ingredient object, keep it
            if isinstance(item, Ingredient):
                result.append(item)
            # If it's a dict (from JSON), parse it
            elif isinstance(item, dict):
                result.append(Ingredient(**item))
            # If it's a string (legacy format), parse it into structured format
            elif isinstance(item, str):
                # Enhanced parsing to handle common patterns
                item = item.strip()

                # Try to extract quantity and unit
                amount = None
                remaining = item

                # Pattern: "1 1/2 cups" or "2 tablespoons" or "1/2 teaspoon"
                quantity_pattern = r"^([0-9]+(?:\s+[0-9]+/[0-9]+)?|[0-9]+/[0-9]+|[0-9]+(?:\.[0-9]+)?|[0-9]+-[0-9]+)\s+"
                match = re.match(quantity_pattern, item)

                if match:
                    quantity = match.group(1)
                    remaining = item[match.end() :]

                    # Try to extract unit
                    unit_words = [
                        "cup",
                        "cups",
                        "tablespoon",
                        "tablespoons",
                        "teaspoon",
                        "teaspoons",
                        "pound",
                        "pounds",
                        "ounce",
                        "ounces",
                        "stick",
                        "sticks",
                        "can",
                        "cans",
                        "package",
                        "packages",
                        "clove",
                        "cloves",
                        "inch",
                        "inches",
                        "piece",
                        "pieces",
                    ]
                    unit_found = False
                    for unit in unit_words:
                        if remaining.lower().startswith(unit):
                            amount = Amount(quantity=quantity, unit=unit)
                            remaining = remaining[len(unit) :].strip()
                            unit_found = True
                            break

                    if not unit_found:
                        # No unit found, just quantity
                        amount = Amount(quantity=quantity)

                # Parse the item (ingredient name and modifiers)
                # For now, treat the whole remaining string as the name
                # In future, we could parse modifiers more intelligently
                item_obj = Item(name=remaining, modifiers=None)

                result.append(Ingredient(amount=amount, item=item_obj))
        return result

    @field_validator("directions", mode="before")
    @classmethod
    def convert_direction_fractions(cls, v: list[str]) -> list[str]:
        """Convert Unicode fractions in directions to readable format."""
        if isinstance(v, list):
            return [convert_unicode_fractions(direction) for direction in v]
        return v

    @field_validator("description", "servings", "total_time", mode="before")
    @classmethod
    def convert_text_fractions(cls, v: str | None) -> str | None:
        """Convert Unicode fractions in text fields to readable format."""
        return convert_unicode_fractions(v) if v else v

    @field_validator("notes", mode="before")
    @classmethod
    def convert_notes_fractions(cls, v: list[str] | None) -> list[str] | None:
        """Convert Unicode fractions in notes to readable format."""
        if isinstance(v, list):
            return [convert_unicode_fractions(note) for note in v]
        return v

    def to_text(self) -> str:
        """Convert recipe to formatted text for easy copying."""
        lines = []

        # Name
        lines.append(f"{self.name}")
        lines.append("=" * (len(self.name) + 8))
        lines.append("")

        # Description
        if self.description:
            lines.append("DESCRIPTION:")
            lines.append(self.description)
            lines.append("")

        # Servings and Time
        if self.servings or self.total_time:
            if self.servings:
                lines.append(f"SERVINGS: {self.servings}")
            if self.total_time:
                lines.append(f"TOTAL TIME: {self.total_time}")
            lines.append("")

        # Ingredients
        lines.append("INGREDIENTS:")
        for ingredient in self.ingredients:
            lines.append(f"  {ingredient.to_string()}")
        lines.append("")

        # Directions
        lines.append("DIRECTIONS:")
        for i, direction in enumerate(self.directions, 1):
            lines.append(f"  {i}. {direction}")
        lines.append("")

        # Notes
        if self.notes:
            lines.append("NOTES:")
            for note in self.notes:
                lines.append(f"  {note}")
            lines.append("")

        # Source
        if self.source:
            lines.append(f"SOURCE: {self.source}")
            lines.append("")

        # Images
        if self.images:
            lines.append("IMAGES:")
            for image in self.images:
                prefix = ""
                if image.is_main:
                    prefix = "[Main] "
                elif image.is_step:
                    prefix = "[Step] "
                lines.append(f"  {prefix}{image.filename}")
                if image.description:
                    lines.append(f"    {image.description}")
            lines.append("")

        lines.append("-" * 50)
        lines.append("")

        return "\n".join(lines)
