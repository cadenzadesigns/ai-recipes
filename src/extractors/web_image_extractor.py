"""Extract images from web pages for manual cropping."""

import tempfile
from pathlib import Path
from typing import List
from urllib.parse import urljoin, urlparse

import click
import requests
from bs4 import BeautifulSoup


class WebImageExtractor:
    """Extract images from web pages for manual cropping."""

    @staticmethod
    def download_images_from_url(url: str, max_images: int = 10) -> List[str]:
        """Download images from a web page.

        Args:
            url: Web page URL
            max_images: Maximum number of images to download

        Returns:
            List of paths to downloaded image files
        """
        try:
            # Fetch the web page
            response = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Find all images
            img_tags = soup.find_all("img")

            # Create temp directory for images
            temp_dir = tempfile.mkdtemp(prefix="recipe_web_")
            downloaded_paths = []

            for i, img in enumerate(img_tags[:max_images]):
                # Get image URL
                img_url = (
                    img.get("src") or img.get("data-src") or img.get("data-lazy-src")
                )
                if not img_url:
                    continue

                # Make URL absolute
                img_url = urljoin(url, img_url)

                # Skip data URLs
                if img_url.startswith("data:"):
                    continue

                # Skip small images (likely icons)
                width = img.get("width")
                height = img.get("height")
                if width and height:
                    try:
                        if int(width) < 100 or int(height) < 100:
                            continue
                    except (ValueError, TypeError):
                        pass

                try:
                    # Download image
                    img_response = requests.get(
                        img_url,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        },
                    )
                    img_response.raise_for_status()

                    # Determine file extension
                    content_type = img_response.headers.get("content-type", "")
                    if "jpeg" in content_type or "jpg" in content_type:
                        ext = ".jpg"
                    elif "png" in content_type:
                        ext = ".png"
                    elif "gif" in content_type:
                        ext = ".gif"
                    elif "webp" in content_type:
                        ext = ".webp"
                    else:
                        # Try to get from URL
                        parsed = urlparse(img_url)
                        ext = Path(parsed.path).suffix or ".jpg"

                    # Save image
                    img_path = Path(temp_dir) / f"web_image_{i + 1:03d}{ext}"
                    with open(img_path, "wb") as f:
                        f.write(img_response.content)

                    downloaded_paths.append(str(img_path))

                except Exception as e:
                    click.echo(
                        f"Failed to download image from {img_url}: {e}", err=True
                    )
                    continue

            return downloaded_paths

        except Exception as e:
            click.echo(f"Failed to extract images from {url}: {e}", err=True)
            return []

    @staticmethod
    def find_recipe_images(soup: BeautifulSoup) -> List[str]:
        """Find likely recipe images in parsed HTML.

        Looks for images with recipe-related classes or attributes.
        """
        recipe_images = []

        # Common recipe image selectors
        selectors = [
            "img.recipe-photo",
            "img.recipe-image",
            'img[itemprop="image"]',
            "div.recipe-image img",
            "div.recipe-photo img",
            "figure.recipe-card__image img",
            "div.tasty-recipes-image img",
            "div.wprm-recipe-image img",
        ]

        for selector in selectors:
            images = soup.select(selector)
            for img in images:
                src = img.get("src") or img.get("data-src")
                if src and src not in recipe_images:
                    recipe_images.append(src)

        # Also look for images with recipe-related alt text
        all_images = soup.find_all("img")
        for img in all_images:
            alt = img.get("alt", "").lower()
            if any(keyword in alt for keyword in ["recipe", "dish", "food", "cooking"]):
                src = img.get("src") or img.get("data-src")
                if src and src not in recipe_images:
                    recipe_images.append(src)

        return recipe_images
