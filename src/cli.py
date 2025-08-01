import json
from pathlib import Path
from typing import Dict, List

import click
from dotenv import load_dotenv

from .config import OPENAI_MODEL
from .converter import RecipeConverter
from .extractors.clipboard import ClipboardExtractor
from .extractors.image import ImageExtractor
from .extractors.pdf import PDFExtractor
from .extractors.recipe_image_extractor import RecipeImageExtractor
from .extractors.web import WebExtractor
from .formatter import RecipeFormatter
from .llm_client import LLMClient
from .paprika_client import PaprikaClient

# Load environment variables
load_dotenv()


def parse_manifest(manifest_path: str) -> List[List[str]]:
    """Parse a manifest file that contains an array of image groups.

    Expected format:
    [
        ["image1.jpg", "image2.jpg"],
        ["image3.jpg", "image4.jpg", "image5.jpg"],
        ["image6.jpg"]
    ]
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Validate manifest structure
    if not isinstance(manifest, list):
        raise ValueError("Manifest must be a JSON array of image groups")

    for i, group in enumerate(manifest):
        if not isinstance(group, list):
            raise ValueError(f"Group {i} must be a list of image paths")
        if not group:
            raise ValueError(f"Group {i} has no images")

    return manifest


def interactive_group_images(image_paths: List[str]) -> Dict[str, List[str]]:
    """Interactively group images into recipes."""
    click.echo("\n=== Interactive Recipe Grouping ===")
    click.echo(f"You have {len(image_paths)} images to group into recipes.\n")

    # Show all images
    for i, path in enumerate(image_paths, 1):
        click.echo(f"{i}. {Path(path).name}")

    recipes = {}
    grouped_images = set()
    recipe_count = 1

    while len(grouped_images) < len(image_paths):
        click.echo(f"\n--- Recipe {recipe_count} ---")
        recipe_name = click.prompt("Recipe name", type=str)

        # Show remaining images
        remaining = [
            i for i, path in enumerate(image_paths) if path not in grouped_images
        ]
        if not remaining:
            break

        click.echo("\nAvailable images:")
        for idx in remaining:
            click.echo(f"{idx + 1}. {Path(image_paths[idx]).name}")

        # Get image selection
        selection = click.prompt(
            "Select images for this recipe (comma-separated numbers, e.g., 1,2,3)",
            type=str,
        )

        try:
            selected_indices = [int(x.strip()) - 1 for x in selection.split(",")]
            selected_paths = []

            for idx in selected_indices:
                if idx < 0 or idx >= len(image_paths):
                    click.echo(f"Invalid image number: {idx + 1}", err=True)
                    continue
                if image_paths[idx] in grouped_images:
                    click.echo(
                        f"Image {idx + 1} already assigned to another recipe", err=True
                    )
                    continue
                selected_paths.append(image_paths[idx])
                grouped_images.add(image_paths[idx])

            if selected_paths:
                recipes[recipe_name] = selected_paths
                recipe_count += 1
                click.echo(f"✓ Added {len(selected_paths)} images to '{recipe_name}'")

        except (ValueError, IndexError) as e:
            click.echo(f"Invalid selection: {e}", err=True)
            continue

        # Ask if more recipes
        if len(grouped_images) < len(image_paths):
            if not click.confirm("\nAdd another recipe?"):
                # Group remaining images
                remaining_paths = [p for p in image_paths if p not in grouped_images]
                if remaining_paths:
                    if click.confirm(
                        f"\n{len(remaining_paths)} images remaining. Group them as one recipe?"
                    ):
                        recipe_name = click.prompt(
                            "Recipe name for remaining images", type=str
                        )
                        recipes[recipe_name] = remaining_paths
                break

    return recipes


@click.group()
@click.option("--api-key", envvar="OPENAI_API_KEY", help="OpenAI API key")
@click.option("--model", default=OPENAI_MODEL, help="LLM model to use")
@click.pass_context
def cli(ctx, api_key, model):
    """AI Recipe Extractor - Extract recipes from images, web pages, and PDFs."""
    ctx.ensure_object(dict)
    ctx.obj["api_key"] = api_key
    ctx.obj["model"] = model


@cli.command()
@click.argument("image_paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--output-dir", "-o", default="recipes", help="Output directory for recipe files"
)
@click.option(
    "--batch",
    "-b",
    is_flag=True,
    help="Treat multiple images as one recipe (e.g., multi-page recipes)",
)
@click.option("--source", "-s", help="Source information for the recipes")
@click.option(
    "--manifest",
    "-m",
    type=click.Path(exists=True),
    help="JSON file mapping images to recipes for multi-recipe batch processing",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Interactively group images into recipes",
)
@click.pass_context
def image(ctx, image_paths, output_dir, batch, source, manifest, interactive):
    """Extract recipes from one or more images.

    By default, each image is treated as a separate recipe.
    Use --batch to combine multiple images into a single recipe (e.g., for multi-page recipes).
    Use --manifest with a JSON file to group multiple multi-page recipes.
    Use --interactive to interactively group images into recipes.
    """

    llm_client = LLMClient(api_key=ctx.obj["api_key"], model=ctx.obj["model"])
    image_extractor = ImageExtractor()
    formatter = RecipeFormatter(output_dir)
    recipe_image_extractor = RecipeImageExtractor(llm_client)

    recipes = []
    recipe_groups = {}

    # Determine grouping mode
    if manifest:
        # Load groupings from manifest file
        try:
            manifest_groups = parse_manifest(manifest)
            # Convert relative paths in manifest to absolute paths matching input
            image_path_map = {Path(p).name: p for p in image_paths}

            for group_idx, image_names in enumerate(manifest_groups):
                matched_paths = []
                for img_name in image_names:
                    if img_name in image_path_map:
                        matched_paths.append(image_path_map[img_name])
                    else:
                        # Try absolute path
                        if Path(img_name).exists():
                            matched_paths.append(img_name)
                        else:
                            click.echo(
                                f"⚠️  Image '{img_name}' not found for recipe group {group_idx + 1}",
                                err=True,
                            )

                if matched_paths:
                    # Use placeholder name that will be replaced by LLM
                    recipe_groups[f"Recipe_{group_idx + 1}"] = matched_paths

        except Exception as e:
            click.echo(f"✗ Failed to parse manifest: {str(e)}", err=True)
            return

    elif interactive and len(image_paths) > 1:
        # Interactive grouping
        recipe_groups = interactive_group_images(list(image_paths))

    elif len(image_paths) > 1 and batch:
        # Single multi-page recipe
        recipe_groups = {"Combined Recipe": list(image_paths)}

    else:
        # Each image is a separate recipe
        recipe_groups = {f"Recipe_{i}": [path] for i, path in enumerate(image_paths, 1)}

    # Process each recipe group
    total_groups = len(recipe_groups)
    for group_idx, (group_name, group_images) in enumerate(recipe_groups.items(), 1):
        click.echo(f"\n[{group_idx}/{total_groups}] Processing recipe: {group_name}")
        click.echo(f"  Images: {len(group_images)} files")

        try:
            if len(group_images) > 1:
                # Multi-page recipe
                content = image_extractor.process_multiple_images(group_images)
            else:
                # Single image
                content = image_extractor.process_image(group_images[0])

            # Extract recipe
            recipe = llm_client.extract_recipe(content, source or group_name)

            recipes.append((recipe, group_images))
            click.echo(f"✓ Extracted: {recipe.name}")

        except Exception as e:
            click.echo(f"✗ Failed to extract recipe: {str(e)}", err=True)

    # Save recipes
    if recipes:
        click.echo(f"\nSaving {len(recipes)} recipe(s)...")

        for recipe, recipe_images in recipes:
            try:
                recipe_dir = formatter.save_recipe(recipe)
                click.echo(f"Saved recipe to: {recipe_dir}")

                # Extract images for this specific recipe
                if recipe_images:
                    click.echo("  Analyzing for recipe images...")
                    try:
                        image_metadata = recipe_image_extractor.extract_recipe_images(
                            recipe_images, recipe.name, recipe_dir
                        )

                        # Update recipe with image references
                        from .models import RecipeImage

                        recipe.images = [
                            RecipeImage(
                                filename=img["filename"],
                                description=img["description"],
                                is_main=img["is_main"],
                                is_step=img["is_step"],
                            )
                            for img in image_metadata["extracted_images"]
                        ]

                        # Update existing files with image info
                        formatter.update_recipe_files(recipe, recipe_dir)

                        click.echo(
                            f"  ✓ Extracted {len(image_metadata['extracted_images'])} images"
                        )
                    except Exception as e:
                        click.echo(f"  ✗ Failed to extract images: {str(e)}", err=True)

            except Exception as e:
                click.echo(
                    f"✗ Failed to save recipe '{recipe.name}': {str(e)}", err=True
                )

        # Create summary
        if len(recipes) > 1:
            click.echo("\n=== Summary ===")
            click.echo(f"Successfully processed {len(recipes)} recipes:")
            for recipe, images in recipes:
                click.echo(f"  • {recipe.name} ({len(images)} images)")

    else:
        click.echo("No recipes were successfully extracted.", err=True)


@cli.command()
@click.argument("url")
@click.option(
    "--output-dir", "-o", default="recipes", help="Output directory for recipe files"
)
@click.pass_context
def web(ctx, url, output_dir):
    """Extract a recipe from a web page."""

    llm_client = LLMClient(api_key=ctx.obj["api_key"], model=ctx.obj["model"])
    web_extractor = WebExtractor()
    formatter = RecipeFormatter(output_dir)

    click.echo(f"Fetching content from {url}...")

    try:
        content = web_extractor.extract_from_url(url)
        click.echo("Extracting recipe...")
        recipe = llm_client.extract_recipe(content, url)

        output_path = formatter.save_recipe(recipe)
        click.echo(f"✓ Extracted: {recipe.name}")
        click.echo(f"Saved to: {output_path}")

    except Exception as e:
        click.echo(f"✗ Failed to extract recipe: {str(e)}", err=True)


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--pages", "-p", help='Page numbers to extract (e.g., "1,3,5" or "1-5")')
@click.option(
    "--output-dir", "-o", default="recipes", help="Output directory for recipe files"
)
@click.pass_context
def pdf(ctx, pdf_path, pages, output_dir):
    """Extract recipes from a PDF file."""

    llm_client = LLMClient(api_key=ctx.obj["api_key"], model=ctx.obj["model"])
    pdf_extractor = PDFExtractor()
    formatter = RecipeFormatter(output_dir)

    # Parse page numbers if provided
    page_numbers = None
    if pages:
        page_numbers = []
        for part in pages.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                page_numbers.extend(range(start - 1, end))  # Convert to 0-based
            else:
                page_numbers.append(int(part) - 1)  # Convert to 0-based

    click.echo(f"Extracting text from {pdf_path}...")

    try:
        content = pdf_extractor.extract_from_pdf(pdf_path, page_numbers)
        click.echo("Extracting recipe...")
        recipe = llm_client.extract_recipe(content, str(pdf_path))

        output_path = formatter.save_recipe(recipe)
        click.echo(f"✓ Extracted: {recipe.name}")
        click.echo(f"Saved to: {output_path}")

    except Exception as e:
        click.echo(f"✗ Failed to extract recipe: {str(e)}", err=True)


@cli.command()
@click.option(
    "--output-dir", "-o", default="recipes", help="Output directory for recipe files"
)
@click.option("--source", "-s", help="Source information for the recipe")
@click.pass_context
def clipboard(ctx, output_dir, source):
    """Extract recipe from image in clipboard."""

    llm_client = LLMClient(api_key=ctx.obj["api_key"], model=ctx.obj["model"])
    clipboard_extractor = ClipboardExtractor()
    formatter = RecipeFormatter(output_dir)

    click.echo("Getting image from clipboard...")

    try:
        content = clipboard_extractor.process_clipboard()
        click.echo("Extracting recipe...")
        recipe = llm_client.extract_recipe(content, source or "Clipboard")

        output_path = formatter.save_recipe(recipe)
        click.echo(f"✓ Extracted: {recipe.name}")
        click.echo(f"Saved to: {output_path}")

    except ValueError as e:
        click.echo(f"✗ {str(e)}", err=True)
        if (
            "pngpaste" in str(e).lower()
            and click.get_current_context().info_name == "darwin"
        ):
            click.echo(
                "\nTip: Install pngpaste for better clipboard support:", err=True
            )
            click.echo("  brew install pngpaste", err=True)
    except Exception as e:
        click.echo(f"✗ Failed to extract recipe from clipboard: {str(e)}", err=True)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output-dir", "-o", default="recipes", help="Output directory for recipe files"
)
@click.option("--batch", "-b", is_flag=True, help="Save all recipes in a single file")
@click.pass_context
def batch(ctx, input_file, output_dir, batch):
    """Process multiple inputs from a file (one path/URL per line)."""

    llm_client = LLMClient(api_key=ctx.obj["api_key"], model=ctx.obj["model"])
    image_extractor = ImageExtractor()
    web_extractor = WebExtractor()
    pdf_extractor = PDFExtractor()
    formatter = RecipeFormatter(output_dir)
    recipe_image_extractor = RecipeImageExtractor(llm_client)

    recipes = []
    image_paths_by_recipe = {}  # Track image paths for each recipe

    with open(input_file) as f:
        inputs = [line.strip() for line in f if line.strip()]

    for input_path in inputs:
        click.echo(f"\nProcessing: {input_path}")

        try:
            if input_path.startswith(("http://", "https://")):
                # Web URL
                content = web_extractor.extract_from_url(input_path)
                source = input_path
            elif input_path.lower().endswith(".pdf"):
                # PDF file
                content = pdf_extractor.extract_from_pdf(input_path)
                source = input_path
            elif Path(input_path).suffix.lower() in ImageExtractor.SUPPORTED_FORMATS:
                # Image file
                content = image_extractor.process_image(input_path)
                source = input_path
            else:
                click.echo(f"✗ Unsupported input type: {input_path}", err=True)
                continue

            recipe = llm_client.extract_recipe(content, source)
            recipes.append(recipe)
            click.echo(f"✓ Extracted: {recipe.name}")

            # Track image paths for image extraction
            if Path(input_path).suffix.lower() in ImageExtractor.SUPPORTED_FORMATS:
                if recipe.name not in image_paths_by_recipe:
                    image_paths_by_recipe[recipe.name] = []
                image_paths_by_recipe[recipe.name].append(input_path)

        except Exception as e:
            click.echo(f"✗ Failed: {str(e)}", err=True)

    # Save recipes
    if recipes:
        if batch:
            output_path = formatter.save_recipes_batch(recipes)
            click.echo(f"\nSaved {len(recipes)} recipes to: {output_path}")
        else:
            click.echo(f"\nSaving {len(recipes)} recipes...")
            for recipe in recipes:
                recipe_dir = formatter.save_recipe(recipe)
                click.echo(f"Saved: {recipe_dir}")

                # Extract images if we have image paths for this recipe
                if recipe.name in image_paths_by_recipe:
                    click.echo("Analyzing for recipe images...")
                    try:
                        image_metadata = recipe_image_extractor.extract_recipe_images(
                            image_paths_by_recipe[recipe.name], recipe.name, recipe_dir
                        )

                        # Update recipe with image references
                        from .models import RecipeImage

                        recipe.images = [
                            RecipeImage(
                                filename=img["filename"],
                                description=img["description"],
                                is_main=img["is_main"],
                                is_step=img["is_step"],
                            )
                            for img in image_metadata["extracted_images"]
                        ]

                        # Update existing files with image info
                        formatter.update_recipe_files(recipe, recipe_dir)

                        click.echo(
                            f"✓ Extracted {len(image_metadata['extracted_images'])} images"
                        )
                    except Exception as e:
                        click.echo(f"✗ Failed to extract images: {str(e)}", err=True)

        # Create index
        index_path = formatter.create_index(recipes)
        click.echo(f"Created index: {index_path}")
    else:
        click.echo("\nNo recipes were successfully extracted.", err=True)


@cli.command()
@click.argument("recipe_file", type=click.Path(exists=True))
@click.option("--email", "-e", envvar="PAPRIKA_EMAIL", help="Paprika email")
@click.option("--password", "-p", envvar="PAPRIKA_PASSWORD", help="Paprika password")
@click.pass_context
def paprika(ctx, recipe_file, email, password):
    """Upload a recipe to Paprika Recipe Manager."""

    try:
        paprika_client = PaprikaClient(email=email, password=password)
    except ValueError as e:
        click.echo(f"✗ {str(e)}", err=True)
        click.echo("\nSet credentials in .env file or use -e and -p options", err=True)
        return

    # Read the recipe file
    import json

    from .models import Recipe

    try:
        with open(recipe_file) as f:
            content = f.read()

        # Try to parse as JSON (in case we saved recipes as JSON)
        try:
            recipe_data = json.loads(content)
            recipe = Recipe(**recipe_data)
        except json.JSONDecodeError:
            # Otherwise, we need to parse the text format
            # For now, we'll need to extract from the text file
            click.echo(
                "✗ Currently only JSON recipe files are supported for upload", err=True
            )
            return

        click.echo(f"Uploading '{recipe.name}' to Paprika...")

        # Skip the search for now since it requires fetching each recipe
        # # Check if recipe already exists
        # existing = paprika_client.search_recipe(recipe.name)
        # if existing:
        #     if not click.confirm(f"Recipe '{recipe.name}' already exists. Overwrite?"):
        #         return

        # Upload the recipe
        result = paprika_client.upload_recipe(recipe)
        click.echo("✓ Successfully uploaded recipe to Paprika!")

        if isinstance(result, dict) and "uid" in result:
            click.echo(f"Recipe ID: {result['uid']}")

    except Exception as e:
        click.echo(f"✗ Failed to upload recipe: {str(e)}", err=True)


@cli.command()
@click.option(
    "--input-dir", "-i", default="recipes", help="Input directory containing txt files"
)
@click.option(
    "--output-dir",
    "-o",
    help="Output directory for JSON files (default: input_dir/json)",
)
@click.option(
    "--single", "-s", type=click.Path(exists=True), help="Convert a single txt file"
)
@click.pass_context
def convert(ctx, input_dir, output_dir, single):
    """Convert recipe txt files to JSON format."""

    if single:
        # Convert a single file
        txt_path = Path(single)
        if not txt_path.suffix == ".txt":
            click.echo("✗ File must be a .txt file", err=True)
            return

        # Determine output path
        if output_dir:
            json_dir = Path(output_dir)
        else:
            # Default to json subdirectory
            json_dir = txt_path.parent / "json"

        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / txt_path.with_suffix(".json").name

        click.echo(f"Converting {txt_path.name}...")

        if RecipeConverter.convert_txt_file_to_json(txt_path, json_path):
            click.echo(f"✓ Converted to: {json_path}")
        else:
            click.echo("✗ Conversion failed", err=True)

    else:
        # Batch convert
        txt_dir = Path(input_dir)

        # For legacy support, check if txt files are in root recipes dir
        if not (txt_dir / "txt").exists() and list(txt_dir.glob("*.txt")):
            click.echo("Found txt files in root directory. Converting...")
            json_dir = txt_dir / "json" if not output_dir else Path(output_dir)
        else:
            txt_dir = txt_dir / "txt"
            json_dir = txt_dir.parent / "json" if not output_dir else Path(output_dir)

        if not txt_dir.exists():
            click.echo(f"✗ Directory not found: {txt_dir}", err=True)
            return

        click.echo(f"Converting txt files from {txt_dir} to {json_dir}...")

        stats = RecipeConverter.batch_convert(txt_dir, json_dir)

        click.echo("\n✓ Conversion complete:")
        click.echo(f"  - Converted: {stats['success']} files")
        click.echo(f"  - Failed: {stats['failed']} files")
        click.echo(
            f"  - Skipped: {stats['skipped']} files (already exist or index files)"
        )


if __name__ == "__main__":
    cli()
