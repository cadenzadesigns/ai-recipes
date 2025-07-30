import click
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from .llm_client import LLMClient
from .extractors.image import ImageExtractor
from .extractors.web import WebExtractor
from .extractors.pdf import PDFExtractor
from .extractors.clipboard import ClipboardExtractor
from .formatter import RecipeFormatter
from .models import Recipe


# Load environment variables
load_dotenv()


@click.group()
@click.option('--api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
@click.option('--model', default='gpt-4o', help='LLM model to use')
@click.pass_context
def cli(ctx, api_key, model):
    """AI Recipe Extractor - Extract recipes from images, web pages, and PDFs."""
    ctx.ensure_object(dict)
    ctx.obj['api_key'] = api_key
    ctx.obj['model'] = model


@cli.command()
@click.argument('image_paths', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='recipes', help='Output directory for recipe files')
@click.option('--batch', '-b', is_flag=True, help='Save all recipes in a single file')
@click.option('--source', '-s', help='Source information for the recipes')
@click.pass_context
def image(ctx, image_paths, output_dir, batch, source):
    """Extract recipes from one or more images."""
    
    llm_client = LLMClient(api_key=ctx.obj['api_key'], model=ctx.obj['model'])
    image_extractor = ImageExtractor()
    formatter = RecipeFormatter(output_dir)
    
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
                output_path = formatter.save_recipe(recipe)
                click.echo(f"Saved recipe to: {output_path}")
    else:
        click.echo("No recipes were successfully extracted.", err=True)


@cli.command()
@click.argument('url')
@click.option('--output-dir', '-o', default='recipes', help='Output directory for recipe files')
@click.pass_context
def web(ctx, url, output_dir):
    """Extract a recipe from a web page."""
    
    llm_client = LLMClient(api_key=ctx.obj['api_key'], model=ctx.obj['model'])
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
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--pages', '-p', help='Page numbers to extract (e.g., "1,3,5" or "1-5")')
@click.option('--output-dir', '-o', default='recipes', help='Output directory for recipe files')
@click.pass_context
def pdf(ctx, pdf_path, pages, output_dir):
    """Extract recipes from a PDF file."""
    
    llm_client = LLMClient(api_key=ctx.obj['api_key'], model=ctx.obj['model'])
    pdf_extractor = PDFExtractor()
    formatter = RecipeFormatter(output_dir)
    
    # Parse page numbers if provided
    page_numbers = None
    if pages:
        page_numbers = []
        for part in pages.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                page_numbers.extend(range(start-1, end))  # Convert to 0-based
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
@click.option('--output-dir', '-o', default='recipes', help='Output directory for recipe files')
@click.option('--source', '-s', help='Source information for the recipe')
@click.pass_context
def clipboard(ctx, output_dir, source):
    """Extract recipe from image in clipboard."""
    
    llm_client = LLMClient(api_key=ctx.obj['api_key'], model=ctx.obj['model'])
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
        if "pngpaste" in str(e).lower() and click.get_current_context().info_name == "darwin":
            click.echo("\nTip: Install pngpaste for better clipboard support:", err=True)
            click.echo("  brew install pngpaste", err=True)
    except Exception as e:
        click.echo(f"✗ Failed to extract recipe from clipboard: {str(e)}", err=True)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='recipes', help='Output directory for recipe files')
@click.option('--batch', '-b', is_flag=True, help='Save all recipes in a single file')
@click.pass_context
def batch(ctx, input_file, output_dir, batch):
    """Process multiple inputs from a file (one path/URL per line)."""
    
    llm_client = LLMClient(api_key=ctx.obj['api_key'], model=ctx.obj['model'])
    image_extractor = ImageExtractor()
    web_extractor = WebExtractor()
    pdf_extractor = PDFExtractor()
    formatter = RecipeFormatter(output_dir)
    
    recipes = []
    
    with open(input_file, 'r') as f:
        inputs = [line.strip() for line in f if line.strip()]
    
    for input_path in inputs:
        click.echo(f"\nProcessing: {input_path}")
        
        try:
            if input_path.startswith(('http://', 'https://')):
                # Web URL
                content = web_extractor.extract_from_url(input_path)
                source = input_path
            elif input_path.lower().endswith('.pdf'):
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
                output_path = formatter.save_recipe(recipe)
                click.echo(f"Saved: {output_path}")
        
        # Create index
        index_path = formatter.create_index(recipes)
        click.echo(f"Created index: {index_path}")
    else:
        click.echo("\nNo recipes were successfully extracted.", err=True)


if __name__ == '__main__':
    cli()