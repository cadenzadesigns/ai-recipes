from __future__ import annotations

import re
import unicodedata

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
    metric_quantity: str | None = Field(
        default=None,
        description="The metric quantity if provided (e.g., '200' for 200g, '500' for 500ml)",
    )
    metric_unit: str | None = Field(
        default=None,
        description="The metric unit if provided (e.g., 'g', 'ml', 'kg', 'L')",
    )

    @field_validator("quantity", "metric_quantity", mode="before")
    @classmethod
    def convert_amount_fractions(cls, v: str | None) -> str | None:
        """Convert Unicode fractions in quantities to readable format."""
        if v:
            v = convert_unicode_fractions(v)
            v = remove_diacritics(v)
        return v


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
        """Convert Unicode fractions and diacritics in names to readable format."""
        if v:
            v = convert_unicode_fractions(v)
            v = remove_diacritics(v)
        return v

    @field_validator("modifiers", mode="before")
    @classmethod
    def convert_modifiers_fractions(cls, v: list[str] | None) -> list[str] | None:
        """Convert Unicode fractions and diacritics in modifiers to readable format."""
        if isinstance(v, list):
            return [remove_diacritics(convert_unicode_fractions(mod)) for mod in v]
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
        default=None,
        description="The amount/quantity of the ingredient. For ingredients like 'Ritz crackers, for serving' or 'salt to taste', the amount should be None.",
    )
    item: Item = Field(
        description="The ingredient item with name and modifiers. For 'Ritz crackers, for serving', name='Ritz crackers' and modifiers=['for serving']."
    )

    def to_string(self, prefer_metric_first: bool = False) -> str:
        """Convert ingredient to human-readable string.

        Args:
            prefer_metric_first: If True and both measurements exist, show metric first.
                               If False (default), show standard first.
        """
        if self.amount:
            parts = []

            # Check if we have both metric and standard measurements
            has_metric = self.amount.metric_quantity and self.amount.metric_unit
            has_standard = self.amount.quantity and self.amount.unit

            if has_metric and has_standard:
                # Display both measurements in preferred order
                metric_str = f"{self.amount.metric_quantity}{self.amount.metric_unit}"
                standard_str = f"{self.amount.quantity} {self.amount.unit}"

                if prefer_metric_first:
                    parts.append(f"{metric_str} / {standard_str}")
                else:
                    parts.append(f"{standard_str} / {metric_str}")
            elif has_standard:
                # Only standard measurement
                parts.append(self.amount.quantity)
                if self.amount.unit:
                    parts.append(self.amount.unit)
            elif has_metric:
                # Only metric measurement
                parts.append(f"{self.amount.metric_quantity}{self.amount.metric_unit}")
            elif self.amount.quantity:
                # Just a quantity with no unit
                parts.append(self.amount.quantity)

            if parts:  # Only add item if we had any measurement
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


def remove_diacritics(text: str) -> str:
    """Remove diacritical marks from text and convert to ASCII-compatible form."""
    if not text:
        return text

    # First handle smart quotes and apostrophes
    # Using individual replacements to avoid false positive from linter
    text = text.replace("'", "'")  # Left single quotation mark (U+2018)
    text = text.replace("'", "'")  # Right single quotation mark (U+2019)
    text = text.replace(
        """, '"')  # Left double quotation mark
    text = text.replace(""",
        '"',
    )  # Right double quotation mark
    text = text.replace("„", '"')  # Double low-9 quotation mark
    text = text.replace("‚", "'")  # Single low-9 quotation mark

    # Normalize to NFD (decomposed form) then encode to ASCII, ignoring errors
    # This removes accents and diacritical marks
    normalized = unicodedata.normalize("NFD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")

    # Handle some special cases that might not convert well
    replacements = {
        "Æ": "AE",
        "æ": "ae",
        "Œ": "OE",
        "œ": "oe",
        "Ø": "O",
        "ø": "o",
        "ß": "ss",
        "Þ": "Th",
        "þ": "th",
        "Ð": "D",
        "ð": "d",
    }

    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    return ascii_text if ascii_text else text


class RecipeComponent(BaseModel):
    """Represents a component/section of a recipe (e.g., 'For the Dough', 'For the Filling')."""

    title: str = Field(
        description="The title of this component (e.g., 'For the Dough', 'Filling', 'Frosting')"
    )
    ingredients: list[Ingredient] = Field(
        description="List of ingredients for this component with structured amounts and items."
    )

    @field_validator("title", mode="before")
    @classmethod
    def convert_title(cls, v: str) -> str:
        """Convert Unicode fractions and diacritics in title."""
        if v:
            v = convert_unicode_fractions(v)
            v = remove_diacritics(v)
        return v

    def to_text(self, indent="  ") -> str:
        """Convert component to formatted text."""
        lines = []
        lines.append(f"{indent}{self.title}:")
        for ingredient in self.ingredients:
            lines.append(f"{indent}  • {ingredient.to_string()}")
        return "\n".join(lines)


class Recipe(BaseModel):
    name: str = Field(description="The name of the recipe")
    alternate_name: str | None = Field(
        default=None, description="An alternate or subtitle for the recipe if provided"
    )
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
    ingredients: list[Ingredient] | None = Field(
        default=None,
        description="List of ingredients if the recipe doesn't have separate components. Parse each ingredient line as: [amount] [ingredient name], [modifiers]. Examples: '1 pound cooked crabmeat' → amount='1 pound', name='crabmeat', modifiers=['cooked']. 'Ritz crackers, for serving' → no amount, name='Ritz crackers', modifiers=['for serving']. '2 tablespoons fresh, minced chives' → amount='2 tablespoons', name='chives', modifiers=['fresh', 'minced']. The ingredient name is the core item (crabmeat, chives, Ritz crackers), everything else is modifiers.",
    )
    components: list[RecipeComponent] | None = Field(
        default=None,
        description="List of recipe components if the recipe has multiple sections with separate ingredient lists (e.g., 'For the Dough', 'For the Filling'). Use this INSTEAD of ingredients field when recipe has distinct components.",
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
        cls, v: list[str] | list[dict] | list[Ingredient] | None
    ) -> list[Ingredient] | None:
        """Parse ingredients from various input formats."""
        if v is None:
            return None
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
        """Convert Unicode fractions and diacritics in directions to readable format."""
        if isinstance(v, list):
            return [
                remove_diacritics(convert_unicode_fractions(direction))
                for direction in v
            ]
        return v

    @field_validator(
        "name", "alternate_name", "description", "servings", "total_time", mode="before"
    )
    @classmethod
    def convert_text_fractions(cls, v: str | None) -> str | None:
        """Convert Unicode fractions and diacritics in text fields to readable format."""
        if v:
            v = convert_unicode_fractions(v)
            v = remove_diacritics(v)
        return v

    @field_validator("notes", mode="before")
    @classmethod
    def convert_notes_fractions(cls, v: list[str] | None) -> list[str] | None:
        """Convert Unicode fractions and diacritics in notes to readable format."""
        if isinstance(v, list):
            return [remove_diacritics(convert_unicode_fractions(note)) for note in v]
        return v

    def to_text(self) -> str:
        """Convert recipe to formatted text for easy copying."""
        lines = []

        # Name with alternate if available
        name_display = self.name
        if self.alternate_name:
            name_display = f"{self.name} ({self.alternate_name})"

        lines.append(name_display)
        lines.append("=" * (len(name_display) + 8))
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

        # Ingredients or Components
        if self.components:
            lines.append("INGREDIENTS:")
            for i, component in enumerate(self.components):
                if i > 0:  # Add spacing between components
                    lines.append("")
                lines.append(component.to_text())
            lines.append("")
        elif self.ingredients:
            lines.append("INGREDIENTS:")
            for ingredient in self.ingredients:
                lines.append(f"  • {ingredient.to_string()}")
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
