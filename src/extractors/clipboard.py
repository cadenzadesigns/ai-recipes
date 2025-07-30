import io
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

from PIL import Image

from .image import ImageExtractor

# Platform-specific imports
if sys.platform == "win32":
    # Windows
    from PIL import ImageGrab


class ClipboardExtractor:
    """Extract recipe content from clipboard images."""

    def __init__(self):
        self.image_extractor = ImageExtractor()

    def get_clipboard_image(self) -> Image.Image:
        """Get image from clipboard based on the platform."""

        if sys.platform == "darwin":
            # macOS: Use pbpaste and osascript
            return self._get_macos_clipboard_image()
        elif sys.platform == "win32":
            # Windows: Use PIL's ImageGrab
            return self._get_windows_clipboard_image()
        else:
            # Linux: Try xclip or wl-paste
            return self._get_linux_clipboard_image()

    def _get_macos_clipboard_image(self) -> Image.Image:
        """Get image from clipboard on macOS."""
        import os

        # Try pngpaste first (most reliable)
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                result = subprocess.run(
                    ["pngpaste", tmp.name],
                    capture_output=True
                )

                if result.returncode == 0:
                    # Verify the file has content
                    if os.path.getsize(tmp.name) > 0:
                        return Image.open(tmp.name)
                elif b"No image data found" in result.stderr or b"No image on clipboard" in result.stderr:
                    # pngpaste found no image
                    pass
                else:
                    # Some other pngpaste error
                    print(f"pngpaste error: {result.stderr.decode('utf-8', errors='ignore')}")
        except FileNotFoundError:
            # pngpaste not installed, try osascript
            print("pngpaste not found, trying AppleScript method...")

        # Fallback to osascript method
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                # Try to save clipboard as PNG using osascript
                script = f'''
                on run
                    try
                        -- Check if clipboard contains image
                        set cbInfo to clipboard info
                        -- Check if any image format is present
                        if (cbInfo as string) does not contain "«class PNGf»" and (cbInfo as string) does not contain "TIFF picture" and (cbInfo as string) does not contain "JPEG picture" and (cbInfo as string) does not contain "GIF picture" then
                            error "No image in clipboard"
                        end if

                        -- Get the clipboard content as TIFF (most compatible)
                        set imgData to the clipboard as «class TIFF»

                        -- Write to temporary file
                        set filePath to POSIX file "{tmp.name}"
                        set fileRef to open for access filePath with write permission
                        set eof fileRef to 0
                        write imgData to fileRef
                        close access fileRef

                        return "success"
                    on error errMsg
                        try
                            close access fileRef
                        end try
                        error errMsg
                    end try
                end run
                '''

                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    print(f"AppleScript error: {result.stderr}")
                    # Try a simpler approach - just check clipboard info
                    info_result = subprocess.run(
                        ["osascript", "-e", "clipboard info"],
                        capture_output=True,
                        text=True
                    )
                    print(f"Clipboard info: {info_result.stdout}")

                if result.returncode == 0 and os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
                    # Convert TIFF to PNG if needed
                    img = Image.open(tmp.name)
                    if img.format == 'TIFF':
                        png_path = tmp.name.replace('.png', '_converted.png')
                        img.save(png_path, 'PNG')
                        return Image.open(png_path)
                    return img
                else:
                    error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                    if "No image in clipboard" in error_msg or result.returncode != 0:
                        raise ValueError("No image found in clipboard")

        except Exception as e:
            if "No image found" in str(e):
                raise

        raise ValueError(
            "Could not access clipboard image. Try:\n"
            "1. Make sure you've copied an image to the clipboard\n"
            "2. Install pngpaste for better support: brew install pngpaste"
        )

    def _get_windows_clipboard_image(self) -> Image.Image:
        """Get image from clipboard on Windows."""
        img = ImageGrab.grabclipboard()
        if img is None:
            raise ValueError("No image found in clipboard")
        if not isinstance(img, Image.Image):
            raise ValueError("Clipboard content is not an image")
        return img

    def _get_linux_clipboard_image(self) -> Image.Image:
        """Get image from clipboard on Linux."""
        # Try xclip first
        try:
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
                capture_output=True
            )
            if result.returncode == 0 and result.stdout:
                return Image.open(io.BytesIO(result.stdout))
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Try wl-paste for Wayland
        try:
            result = subprocess.run(
                ["wl-paste", "-t", "image/png"],
                capture_output=True
            )
            if result.returncode == 0 and result.stdout:
                return Image.open(io.BytesIO(result.stdout))
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        raise ValueError(
            "No image found in clipboard. Make sure xclip or wl-paste is installed."
        )

    def process_clipboard(self) -> List[Dict[str, Any]]:
        """Process clipboard image and return content for LLM."""
        img = self.get_clipboard_image()

        # Convert image to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)

        # Save to temporary file and use image extractor
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name, format='PNG')
            return self.image_extractor.process_image(tmp.name)
