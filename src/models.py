from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


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
    servings: Optional[str] = Field(
        default=None,
        description="Number of servings or yield (e.g., '4 servings', '12 cookies', '1 9-inch pie')",
    )
    total_time: Optional[str] = Field(
        default=None,
        description="Total cook time including prep and cooking (e.g., '45 minutes', '1 hour 30 minutes')",
    )
    ingredients: List[str] = Field(
        description="List of ingredients with quantities. Use standard fractions like 1/2, 1/3, 1/4 instead of Unicode fraction characters."
    )
    directions: List[str] = Field(description="Step-by-step cooking directions")
    notes: Optional[List[str]] = Field(
        default=None,
        description="Additional notes, tips, variations, or storage instructions (e.g., 'Store in airtight container for up to 5 days', 'Can be frozen for 3 months', 'Refrigerate leftovers')",
    )
    source: Optional[str] = Field(
        default=None, description="Source of the recipe (cookbook, website, etc.)"
    )
    images: Optional[List[RecipeImage]] = Field(
        default=None, description="Associated recipe images"
    )

    @field_validator("ingredients", mode="before")
    @classmethod
    def convert_ingredient_fractions(cls, v: List[str]) -> List[str]:
        """Convert Unicode fractions in ingredients to readable format."""
        if isinstance(v, list):
            return [convert_unicode_fractions(ingredient) for ingredient in v]
        return v

    @field_validator("directions", mode="before")
    @classmethod
    def convert_direction_fractions(cls, v: List[str]) -> List[str]:
        """Convert Unicode fractions in directions to readable format."""
        if isinstance(v, list):
            return [convert_unicode_fractions(direction) for direction in v]
        return v

    @field_validator("description", "servings", "total_time", mode="before")
    @classmethod
    def convert_text_fractions(cls, v: Optional[str]) -> Optional[str]:
        """Convert Unicode fractions in text fields to readable format."""
        return convert_unicode_fractions(v) if v else v

    @field_validator("notes", mode="before")
    @classmethod
    def convert_notes_fractions(cls, v: Optional[List[str]]) -> Optional[List[str]]:
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
            lines.append(f"  • {ingredient}")
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
                lines.append(f"  • {note}")
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
                lines.append(f"  • {prefix}{image.filename}")
                if image.description:
                    lines.append(f"    {image.description}")
            lines.append("")

        lines.append("-" * 50)
        lines.append("")

        return "\n".join(lines)
