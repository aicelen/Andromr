import os
import re
from pathlib import Path
import cv2
import numpy as np
from kivy.utils import platform


def get_sys_theme():
    """
    Gets the theme of the device.
    Returns:
        "Dark": Dark mode
        "Light": Light mode
    """
    if platform == "android":
        from jnius import autoclass  # type: ignore

        Configuration = autoclass("android.content.res.Configuration")
        activity = autoclass("org.kivy.android.PythonActivity").mActivity

        night_mode_flags = (
            activity.getResources().getConfiguration().uiMode & Configuration.UI_MODE_NIGHT_MASK
        )

        if night_mode_flags == Configuration.UI_MODE_NIGHT_YES:
            return "Dark"
        else:
            return "Light"

    elif platform == "win":
        import winreg

        try:
            registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
            key = winreg.OpenKey(
                registry,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            )
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return "Light" if value == 1 else "Dark"

        except Exception as e:
            print("Error getting theme:", e)
            return "Light"  # Default fallback

    else:
        print("Getting the theme is only working on Android and Windows")
        return "Dark"


def safe_filename(user_input: str) -> str:
    """
    Ensures a safe filename to stop the file from ending up in unwanted locations.
    Returns the safe filename.
    Args:
        user_input(str): The string the user inputted
    """
    # Strip any path parts
    name = Path(user_input).name

    # Keep only the part before the first dot
    name = name.split(".", 1)[0]
    # Only allow Letters, Numbers, _ and -
    name = re.sub(r"[^A-Za-z0-9_-]", "", name)

    return name

def pdf_to_img(
    pdf_path: str, output_path: str, page_index: int | None = None, render_scale: float = 3.0
) -> list[str]:
    """
    Render a PDF to PNG image files on Android.

    Args:
        pdf_path: Local path to a PDF file.
        output_path: Output filename stem or absolute/relative path without extension.
        page_index: Optional zero-based page index. If omitted, all pages are rendered.
        render_scale: Multiplier applied to the PDF page size before rendering.

    Returns:
        A list of PNG image paths, one per rendered page.
    """
    if platform != "android":
        raise NotImplementedError("Using PDFs is only supported on Android")

    from jnius import autoclass  # type: ignore  # noqa: PLC0415

    from globals import IMAGE_PATH  # noqa: PLC0415

    File = autoclass("java.io.File")
    FileOutputStream = autoclass("java.io.FileOutputStream")
    ParcelFileDescriptor = autoclass("android.os.ParcelFileDescriptor")
    PdfRenderer = autoclass("android.graphics.pdf.PdfRenderer")
    PdfRendererPage = autoclass("android.graphics.pdf.PdfRenderer$Page")
    Bitmap = autoclass("android.graphics.Bitmap")
    BitmapConfig = autoclass("android.graphics.Bitmap$Config")
    Canvas = autoclass("android.graphics.Canvas")
    CompressFormat = autoclass("android.graphics.Bitmap$CompressFormat")
    Matrix = autoclass("android.graphics.Matrix")
    Paint = autoclass("android.graphics.Paint")

    pdf_file = File(pdf_path)
    output_base = Path(output_path)
    if not output_base.parent or str(output_base.parent) == ".":
        output_base = Path(IMAGE_PATH) / output_base

    safe_stem = safe_filename(output_base.name) or "converted-pdf"
    output_dir = output_base.parent
    os.makedirs(output_dir, exist_ok=True)

    file_descriptor = None
    renderer = None
    rendered_paths: list[str] = []

    try:
        file_descriptor = ParcelFileDescriptor.open(
            pdf_file, ParcelFileDescriptor.MODE_READ_ONLY
        )
        renderer = PdfRenderer(file_descriptor)
        page_count = renderer.getPageCount()

        if page_count <= 0:
            raise ValueError("PDF does not contain any pages")

        if page_index is None:
            page_indexes = range(page_count)
        else:
            if page_index < 0 or page_index >= page_count:
                raise IndexError(
                    f"PDF page index {page_index} is outside the available range 0-{page_count - 1}"
                )
            page_indexes = [page_index]

        for current_page_index in page_indexes:
            page = None
            bitmap = None
            stream = None
            try:
                page = renderer.openPage(current_page_index)
                matrix = Matrix()
                matrix.postScale(render_scale, render_scale)
                bitmap = Bitmap.createBitmap(
                    int(page.getWidth() * render_scale),
                    int(page.getHeight() * render_scale),
                    BitmapConfig.ARGB_8888,
                )
                canvas = Canvas(bitmap)
                paint = Paint()
                paint.setARGB(255, 255, 255, 255)
                canvas.drawPaint(paint)
                page.render(bitmap, None, matrix, PdfRendererPage.RENDER_MODE_FOR_DISPLAY)

                rendered_path = str(
                    output_dir / f"{safe_stem}-page-{current_page_index + 1}.png"
                )
                stream = FileOutputStream(rendered_path)
                if not bitmap.compress(CompressFormat.PNG, 100, stream):
                    raise RuntimeError(f"Failed to save rendered PDF page to {rendered_path}")

                rendered_paths.append(rendered_path)
            finally:
                if stream is not None:
                    stream.close()
                if bitmap is not None:
                    bitmap.recycle()
                if page is not None:
                    page.close()
    finally:
        if renderer is not None:
            renderer.close()
        if file_descriptor is not None:
            file_descriptor.close()

    return rendered_paths

def downscale_cv2(input_path: str, scale: float):
    """
    Fast downscaling using OpenCV
    """
    img = cv2.imread(input_path)
    size_old = (img.shape[1], img.shape[0])
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    flipped = np.flip(img_rgb, 0)
    buf = flipped.tobytes()
    return buf, size_old, new_size
