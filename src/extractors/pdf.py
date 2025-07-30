from pathlib import Path
from typing import Optional

import PyPDF2


class PDFExtractor:
    """Extract recipe content from PDF files."""

    def __init__(self):
        pass

    def extract_from_pdf(self, pdf_path: str, page_numbers: Optional[list] = None) -> str:
        """Extract text content from a PDF file."""
        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")

        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)

                # Determine which pages to extract
                if page_numbers:
                    pages_to_extract = page_numbers
                else:
                    pages_to_extract = range(len(reader.pages))

                extracted_text = []

                for page_num in pages_to_extract:
                    if page_num < len(reader.pages):
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        if text.strip():
                            extracted_text.append(f"--- Page {page_num + 1} ---\n{text}")
                    else:
                        print(f"Warning: Page {page_num + 1} does not exist in the PDF")

                if not extracted_text:
                    raise ValueError("No text could be extracted from the PDF")

                return f"Content from {pdf_path}:\n\n" + "\n\n".join(extracted_text)

        except PyPDF2.errors.PdfReadError as e:
            raise ValueError(f"Failed to read PDF file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    def extract_from_pdf_with_ocr(self, pdf_path: str, page_numbers: Optional[list] = None) -> str:
        """Extract text from PDF using OCR for scanned documents.

        This is a placeholder for OCR functionality. To implement:
        1. Install pdf2image and pytesseract
        2. Convert PDF pages to images
        3. Run OCR on each image
        """
        raise NotImplementedError(
            "OCR extraction is not yet implemented. "
            "For scanned PDFs, please convert to images first and use the image extractor."
        )
