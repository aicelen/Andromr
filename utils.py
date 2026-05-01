import numpy as np
import cv2
from kivy.utils import platform
from pathlib import Path
import re
from datetime import datetime



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


def safe_filename(user_input: str) -> str:
    """
    Ensures a safe filename to stop the file from ending up in unwanted locations.
    Returns the safe filename.
    Args:
        user_input(str): The string the user inputted
    """
    # Strip any path parts
    name = Path(user_input).name

    print(name)

    # Keep only the part before the first dot
    name = name.split('.', 1)[0]
    print(name)
    # Only allow Letters, Numbers, _ and -
    name = re.sub(r'[^A-Za-z0-9_-]', '', name)
    print(name)

    if not name:
        # in case we're left with nothing we use the default name
        return f"transcribed-music-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    return name
