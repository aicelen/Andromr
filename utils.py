from PIL import Image, ImageOps
import numpy as np
import cv2
from kivy.utils import platform

def rotate_image(input_path, output_path):
    """
    Rotates an image 90 degrees clockwise and saves it.
    Swaps dimensions (e.g., 2000x1000 becomes 1000x2000).

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path where the rotated image will be saved.
    """
    try:
        # load
        img = cv2.imread(input_path)

        # rotate
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # save
        cv2.imwrite(output_path, rotated_img)

    except Exception as e:
        print(f"Error: {e}")


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
            activity.getResources().getConfiguration().uiMode
            & Configuration.UI_MODE_NIGHT_MASK
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


def crop_image_by_corners(path, points, output_path):
    """
    Crops an image based on up to 4 points
    Args:
        path(str): Path to input image
        points(List[int or float]): Points to cut from
        output_path(str): Path to save the image

    Returns:
        None
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    arr = np.array(img)

    # Find bounding box of the 4 points
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    min_x = int(round(min(x_coords)))
    min_y = int(round(min(y_coords)))
    max_x = int(round(max(x_coords)))
    max_y = int(round(max(y_coords)))

    # Crop the image
    arr = arr[min_y:max_y, min_x:max_x]
    img = Image.fromarray(arr)

    # Save the cropped image
    img.save(output_path)

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
