"""
Streamlit components for image upload functionality.

This module provides reusable components for uploading and processing images
in the Streamlit app, with support for all formats handled by ImageExtractor.
"""

import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import streamlit as st
from PIL import Image

from src.extractors.image import ImageExtractor


class ImageUploadComponent:
    """Reusable image upload component for Streamlit apps."""

    def __init__(self, max_file_size_mb: int = 10):
        """
        Initialize the image upload component.

        Args:
            max_file_size_mb: Maximum file size in MB for uploaded images
        """
        self.max_file_size_mb = max_file_size_mb
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.extractor = ImageExtractor()

    @property
    def supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(ImageExtractor.SUPPORTED_FORMATS)

    @property
    def supported_types(self) -> List[str]:
        """Get list of MIME types for file uploader."""
        type_mapping = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
            ".heic": "image/heic",
            ".heif": "image/heif",
        }
        return list(type_mapping.values())

    def validate_image_files(
        self, uploaded_files: Union[List, object]
    ) -> Tuple[List, List[str]]:
        """
        Validate uploaded image files.

        Args:
            uploaded_files: Files from st.file_uploader

        Returns:
            Tuple of (valid_files, error_messages)
        """
        if not uploaded_files:
            return [], []

        # Handle single file case
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]

        valid_files = []
        errors = []

        for file in uploaded_files:
            if file is None:
                continue

            # Check file size
            if file.size > self.max_file_size_bytes:
                errors.append(
                    f"'{file.name}' is too large ({file.size / 1024 / 1024:.1f}MB). "
                    f"Maximum size is {self.max_file_size_mb}MB."
                )
                continue

            # Check file extension
            file_ext = Path(file.name).suffix.lower()
            if file_ext not in self.supported_formats:
                errors.append(
                    f"'{file.name}' has unsupported format '{file_ext}'. "
                    f"Supported formats: {', '.join(self.supported_formats)}"
                )
                continue

            # Check if it's actually an image by trying to open it
            try:
                file.seek(0)  # Reset file pointer
                img = Image.open(file)
                img.verify()  # Verify it's a valid image
                file.seek(0)  # Reset file pointer again
                valid_files.append(file)
            except Exception as e:
                errors.append(f"'{file.name}' is not a valid image file: {str(e)}")

        return valid_files, errors

    def display_image_preview(self, uploaded_file, max_width: int = 300) -> None:
        """
        Display a preview/thumbnail of an uploaded image.

        Args:
            uploaded_file: File from st.file_uploader
            max_width: Maximum width for the preview image
        """
        try:
            uploaded_file.seek(0)
            img = Image.open(uploaded_file)

            # Handle HEIC/HEIF files
            file_ext = Path(uploaded_file.name).suffix.lower()
            if file_ext in {".heic", ".heif"}:
                try:
                    from pillow_heif import register_heif_opener

                    register_heif_opener()
                except ImportError:
                    st.warning(
                        f"⚠️ HEIC/HEIF support not available for preview of '{uploaded_file.name}'. "
                        "Install pillow-heif for full support: `uv add pillow-heif`"
                    )
                    return

            # Display image info
            st.write(f"**{uploaded_file.name}**")
            st.write(f"Size: {img.size[0]} × {img.size[1]} pixels")
            st.write(f"File size: {uploaded_file.size / 1024:.1f} KB")

            # Display the image
            st.image(img, width=max_width, caption=uploaded_file.name)

        except Exception as e:
            st.error(f"Cannot preview '{uploaded_file.name}': {str(e)}")
        finally:
            uploaded_file.seek(0)  # Reset file pointer

    def single_image_uploader(
        self,
        key: str = "single_image",
        label: str = "Upload a recipe image",
        help_text: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a single image uploader component.

        Args:
            key: Unique key for the uploader widget
            label: Label text for the uploader
            help_text: Optional help text

        Returns:
            Path to temporary file if image uploaded successfully, None otherwise
        """
        if help_text is None:
            help_text = (
                f"Supported formats: {', '.join(self.supported_formats)}. "
                f"Maximum file size: {self.max_file_size_mb}MB."
            )

        uploaded_file = st.file_uploader(
            label,
            type=self.supported_types,
            accept_multiple_files=False,
            key=key,
            help=help_text,
        )

        if uploaded_file is not None:
            # Validate the file
            valid_files, errors = self.validate_image_files(uploaded_file)

            if errors:
                for error in errors:
                    st.error(error)
                return None

            if valid_files:
                # Display preview
                with st.expander("Image Preview", expanded=True):
                    self.display_image_preview(valid_files[0])

                # Save to temporary file and return path
                return self._save_temp_file(valid_files[0])

        return None

    def multiple_image_uploader(
        self,
        key: str = "multiple_images",
        label: str = "Upload recipe images",
        help_text: Optional[str] = None,
        max_files: int = 10,
    ) -> List[str]:
        """
        Create a multiple image uploader component.

        Args:
            key: Unique key for the uploader widget
            label: Label text for the uploader
            help_text: Optional help text
            max_files: Maximum number of files to accept

        Returns:
            List of paths to temporary files for successfully uploaded images
        """
        if help_text is None:
            help_text = (
                f"Supported formats: {', '.join(self.supported_formats)}. "
                f"Maximum file size: {self.max_file_size_mb}MB per file. "
                f"Maximum {max_files} files."
            )

        uploaded_files = st.file_uploader(
            label,
            type=self.supported_types,
            accept_multiple_files=True,
            key=key,
            help=help_text,
        )

        if uploaded_files:
            # Check file count limit
            if len(uploaded_files) > max_files:
                st.error(f"Too many files uploaded. Maximum allowed: {max_files}")
                return []

            # Validate files
            valid_files, errors = self.validate_image_files(uploaded_files)

            # Display errors
            if errors:
                st.error("Some files could not be processed:")
                for error in errors:
                    st.write(f"• {error}")

            # Display valid files and previews
            if valid_files:
                st.success(f"✅ {len(valid_files)} valid image(s) uploaded")

                # Show previews in columns for better layout
                if len(valid_files) <= 3:
                    cols = st.columns(len(valid_files))
                    for i, file in enumerate(valid_files):
                        with cols[i]:
                            self.display_image_preview(file, max_width=200)
                else:
                    # Use expander for many images
                    with st.expander(
                        f"Image Previews ({len(valid_files)} images)", expanded=False
                    ):
                        cols = st.columns(3)
                        for i, file in enumerate(valid_files):
                            with cols[i % 3]:
                                self.display_image_preview(file, max_width=150)

                # Save all valid files to temporary files
                temp_paths = []
                for file in valid_files:
                    temp_path = self._save_temp_file(file)
                    if temp_path:
                        temp_paths.append(temp_path)

                return temp_paths

        return []

    def _save_temp_file(self, uploaded_file) -> Optional[str]:
        """
        Save uploaded file to temporary location.

        Args:
            uploaded_file: File from st.file_uploader

        Returns:
            Path to temporary file, or None if failed
        """
        try:
            uploaded_file.seek(0)

            # Get file extension
            file_ext = Path(uploaded_file.name).suffix.lower()

            # Create temporary file with proper extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            # Reset file pointer
            uploaded_file.seek(0)

            return temp_path

        except Exception as e:
            st.error(f"Failed to save '{uploaded_file.name}': {str(e)}")
            return None

    def process_uploaded_images(self, temp_paths: List[str]) -> Optional[List]:
        """
        Process uploaded images using ImageExtractor.

        Args:
            temp_paths: List of temporary file paths

        Returns:
            Processed content for LLM, or None if failed
        """
        if not temp_paths:
            return None

        try:
            if len(temp_paths) == 1:
                return self.extractor.process_image(temp_paths[0])
            else:
                return self.extractor.process_multiple_images(temp_paths)
        except Exception as e:
            st.error(f"Failed to process images: {str(e)}")
            return None

    def cleanup_temp_files(self, temp_paths: List[str]) -> None:
        """
        Clean up temporary files.

        Args:
            temp_paths: List of temporary file paths to delete
        """
        for temp_path in temp_paths:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass  # Ignore cleanup errors


# Convenience functions for easy use in Streamlit apps


def single_image_uploader(
    key: str = "single_image",
    label: str = "Upload a recipe image",
    help_text: Optional[str] = None,
    max_file_size_mb: int = 10,
) -> Optional[str]:
    """
    Convenience function for single image upload.

    Returns:
        Path to temporary file if image uploaded successfully, None otherwise
    """
    component = ImageUploadComponent(max_file_size_mb=max_file_size_mb)
    return component.single_image_uploader(key=key, label=label, help_text=help_text)


def multiple_image_uploader(
    key: str = "multiple_images",
    label: str = "Upload recipe images",
    help_text: Optional[str] = None,
    max_files: int = 10,
    max_file_size_mb: int = 10,
) -> List[str]:
    """
    Convenience function for multiple image upload.

    Returns:
        List of paths to temporary files for successfully uploaded images
    """
    component = ImageUploadComponent(max_file_size_mb=max_file_size_mb)
    return component.multiple_image_uploader(
        key=key, label=label, help_text=help_text, max_files=max_files
    )


def validate_image_files(
    uploaded_files, max_file_size_mb: int = 10
) -> Tuple[List, List[str]]:
    """
    Convenience function for validating uploaded image files.

    Returns:
        Tuple of (valid_files, error_messages)
    """
    component = ImageUploadComponent(max_file_size_mb=max_file_size_mb)
    return component.validate_image_files(uploaded_files)


def display_image_preview(uploaded_file, max_width: int = 300) -> None:
    """
    Convenience function for displaying image preview.
    """
    component = ImageUploadComponent()
    component.display_image_preview(uploaded_file, max_width=max_width)
