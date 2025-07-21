from kivy.utils import platform

def get_sys_theme():
    """
    Gets the theme of the device.
    Returns:
        "Dark": Dark mode 
        "Light": Light mode
    """
    if platform == 'android':
        from jnius import autoclass # pylint: disable=import-error # type: ignore

        Configuration = autoclass("android.content.res.Configuration")
        activity = autoclass("org.kivy.android.PythonActivity").mActivity


        night_mode_flags = activity.getResources().getConfiguration().uiMode & Configuration.UI_MODE_NIGHT_MASK
        
        if night_mode_flags == Configuration.UI_MODE_NIGHT_YES:
            return "Dark"
        else:
            return "Light"


    elif platform =='win':
        import winreg

        try:
            registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
            key = winreg.OpenKey(registry, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return "Light" if value == 1 else "Dark"
        
        except Exception as e:
            print("Error getting theme:", e)
            return "Light"  # Default fallback

    else:
        raise RuntimeError('Getting the theme is only working on Android and Windows')