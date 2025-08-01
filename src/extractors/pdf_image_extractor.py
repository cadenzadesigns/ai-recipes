"""Extract images from PDF files for manual cropping."""

import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from pdf2image import convert_from_path

    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class PDFImageExtractor:
    """Extract pages from PDF as images for manual cropping."""

    @staticmethod
    def pdf_to_images(pdf_path: str, pages: Optional[List[int]] = None) -> List[str]:
        """Convert PDF pages to images.

        Args:
            pdf_path: Path to PDF file
            pages: Optional list of page numbers (0-based) to extract

        Returns:
            List of paths to temporary image files
        """
        temp_dir = tempfile.mkdtemp(prefix="recipe_pdf_")
        image_paths = []

        if PYMUPDF_AVAILABLE:
            # Use PyMuPDF (faster and more reliable)
            doc = fitz.open(pdf_path)

            # Determine which pages to process
            if pages:
                page_nums = pages
            else:
                page_nums = range(len(doc))

            for page_num in page_nums:
                if page_num >= len(doc):
                    continue

                page = doc.load_page(page_num)

                # Render page at high resolution
                mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)

                # Save as image
                image_path = Path(temp_dir) / f"page_{page_num + 1:03d}.png"
                pix.save(str(image_path))
                image_paths.append(str(image_path))

            doc.close()

        elif PDF2IMAGE_AVAILABLE:
            # Fallback to pdf2image
            # Convert pages (1-based for pdf2image)
            if pages:
                first_page = min(pages) + 1
                last_page = max(pages) + 1
                images = convert_from_path(
                    pdf_path,
                    first_page=first_page,
                    last_page=last_page,
                    dpi=200,
                    output_folder=temp_dir,
                    fmt="png",
                )
                # Filter to only requested pages
                page_set = set(p + 1 for p in pages)
                images = [
                    img for i, img in enumerate(images, first_page) if i in page_set
                ]
            else:
                images = convert_from_path(
                    pdf_path, dpi=200, output_folder=temp_dir, fmt="png"
                )

            # Save images
            for i, image in enumerate(images):
                image_path = Path(temp_dir) / f"page_{i + 1:03d}.png"
                image.save(str(image_path), "PNG")
                image_paths.append(str(image_path))

        else:
            raise ImportError(
                "Neither PyMuPDF nor pdf2image is available. "
                "Install with: pip install PyMuPDF or pip install pdf2image"
            )

        return image_paths

    @staticmethod
    def extract_embedded_images(pdf_path: str) -> List[Tuple[int, str]]:
        """Extract embedded images from PDF.

        Returns:
            List of (page_number, image_path) tuples
        """
        if not PYMUPDF_AVAILABLE:
            return []

        temp_dir = tempfile.mkdtemp(prefix="recipe_pdf_images_")
        extracted_images = []

        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                # Extract image
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    image_path = (
                        Path(temp_dir) / f"page_{page_num + 1}_img_{img_index + 1}.png"
                    )
                    pix.save(str(image_path))
                    extracted_images.append((page_num, str(image_path)))
                else:  # CMYK, convert to RGB
                    pix1 = fitz.Pixmap(fitz.csRGB, pix)
                    image_path = (
                        Path(temp_dir) / f"page_{page_num + 1}_img_{img_index + 1}.png"
                    )
                    pix1.save(str(image_path))
                    extracted_images.append((page_num, str(image_path)))
                    pix1 = None

                pix = None

        doc.close()

        return extracted_images
