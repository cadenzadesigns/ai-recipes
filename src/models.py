from typing import List, Optional

from pydantic import BaseModel, Field


class Recipe(BaseModel):
    name: str = Field(description="The name of the recipe")
    description: str = Field(description="The exact description text from the source document, typically found after the recipe title and before the ingredients list. Copy this verbatim from the source.")
    servings: Optional[str] = Field(default=None, description="Number of servings or yield (e.g., '4 servings', '12 cookies', '1 9-inch pie')")
    total_time: Optional[str] = Field(default=None, description="Total cook time including prep and cooking (e.g., '45 minutes', '1 hour 30 minutes')")
    ingredients: List[str] = Field(description="List of ingredients with quantities")
    directions: List[str] = Field(description="Step-by-step cooking directions")
    notes: Optional[List[str]] = Field(default=None, description="Additional notes, tips, variations, or storage instructions (e.g., 'Store in airtight container for up to 5 days', 'Can be frozen for 3 months', 'Refrigerate leftovers')")
    source: Optional[str] = Field(default=None, description="Source of the recipe (cookbook, website, etc.)")

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

        lines.append("-" * 50)
        lines.append("")

        return "\n".join(lines)
