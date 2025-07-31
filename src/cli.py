from pathlib import Path

import click
from dotenv import load_dotenv

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


@click.group()
@click.option("--api-key", envvar="OPENAI_API_KEY", help="OpenAI API key")
@click.option("--model", default="gpt-4o", help="LLM model to use")
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
@click.pass_context
def image(ctx, image_paths, output_dir, batch, source):
    """Extract recipes from one or more images.

    By default, each image is treated as a separate recipe.
    Use --batch to combine multiple images into a single recipe (e.g., for multi-page recipes).
    """

    llm_client = LLMClient(api_key=ctx.obj["api_key"], model=ctx.obj["model"])
    image_extractor = ImageExtractor()
    formatter = RecipeFormatter(output_dir)
    recipe_image_extractor = RecipeImageExtractor(llm_client)

    recipes = []

    # Group images by recipe (assume consecutive images are same recipe)
    if len(image_paths) > 1 and batch:
        click.echo(f"Processing {len(image_paths)} images as a single recipe...")
        try:
            content = image_extractor.process_multiple_images(list(image_paths))
            recipe = llm_client.extract_recipe(content, source)
            recipes.append(recipe)
            click.echo(f"✓ Extracted: {recipe.name}")
        except Exception as e:
            click.echo(f"✗ Failed to extract recipe: {str(e)}", err=True)
    else:
        # Process each image individually
        for image_path in image_paths:
            click.echo(f"Processing {image_path}...")
            try:
                content = image_extractor.process_image(image_path)
                recipe = llm_client.extract_recipe(content, source or str(image_path))
                recipes.append(recipe)
                click.echo(f"✓ Extracted: {recipe.name}")
            except Exception as e:
                click.echo(f"✗ Failed to process {image_path}: {str(e)}", err=True)

    # Save recipes
    if recipes:
        if batch and len(recipes) > 1:
            output_path = formatter.save_recipes_batch(recipes)
            click.echo(f"\nSaved {len(recipes)} recipes to: {output_path}")
        else:
            for recipe in recipes:
                recipe_dir = formatter.save_recipe(recipe)
                click.echo(f"Saved recipe to: {recipe_dir}")

                # Always extract images
                click.echo("Analyzing for recipe images...")
                try:
                    image_metadata = recipe_image_extractor.extract_recipe_images(
                        list(image_paths), recipe.name, recipe_dir
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
