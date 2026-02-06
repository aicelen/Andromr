# main file of the app
# Andromr class is the main class of the app

# Kivy imports
from kivy.lang import Builder
from kivy.clock import Clock
from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDIconButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
from kivymd.uix.menu import MDDropdownMenu
from kivy.core.window import Window
from kivy.utils import platform
from kivymd.toast import toast
from kivy.metrics import dp, sp
from kivy.uix.recycleview import RecycleView
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.camera import Camera

# Built-in imports
from threading import Thread
import os
from datetime import datetime
from time import sleep
from collections import deque

# Package imports
import numpy as np
import cv2

# Own imports
from homr.main import download_weights, homr, check_for_missing_models
from homr.segmentation.inference_segnet import preload_segnet
from homr.relieur import merge_xmls
from globals import APP_PATH, XML_PATH, appdata
from utils import get_sys_theme, downscale_cv2
from plyer import filechooser

if platform == "android":
    from android_camera_api import take_picture
    from androidstorage4kivy import SharedStorage, ShareSheet  # type: ignore
    from jnius import autoclass  # type: ignore
    from android.permissions import request_permissions, Permission, check_permission  # type: ignore

    required_permissions = [Permission.CAMERA]

    def has_all_permissions():
        return all(check_permission(perm) for perm in required_permissions)

    # Request permissions if not already granted
    if not has_all_permissions():
        request_permissions(required_permissions)
        while not has_all_permissions():
            sleep(0.1)
            print("waiting for permissions...")

    # Custom camera widget for android
    class KvCam(Camera):
        CameraInfo = autoclass("android.hardware.Camera$CameraInfo")
        resolution = (640, 480)  # 960, 720
        index = CameraInfo.CAMERA_FACING_BACK

        def on_tex(self, *l):
            if self._camera._buffer is None:
                return None

            super(KvCam, self).on_tex(*l)
            self.texture = Texture.create(size=np.flip(self.resolution), colorfmt="rgb")
            frame = self.frame_from_buf()
            self.frame_to_screen(frame)

        def frame_from_buf(self):
            w, h = self.resolution
            frame = np.frombuffer(self._camera._buffer.tostring(), "uint8").reshape((h + h // 2, w))
            frame_bgr = cv2.cvtColor(frame, 93)
            if self.index:
                return np.flip(np.rot90(frame_bgr, 1), 1)
            else:
                return np.rot90(frame_bgr, 3)

        def frame_to_screen(self, frame):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            flipped = np.flip(frame_rgb, 0)
            buf = flipped.tobytes()
            self.texture.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")

    # Preloading during inferene cuased a black screen; this works well
    preload_segnet(num_threads=appdata.threads, use_gpu=appdata.gpu)

else:

    class KvCam(MDBoxLayout):
        """A placeholder for the camera on desktop platforms."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Add a label to indicate that the camera is not available
            self.add_widget(
                MDLabel(
                    text="Camera not available on Desktop.\nPress capture to select a file.",
                    halign="center",
                )
            )

    def take_picture(widget, function, filename):
        path = filechooser.open_file(title="Select your document")
        function(path[0])


# Classes of Screens used by kivy
class LandingPage(Screen):
    def on_enter(self):
        self.app = MDApp.get_running_app()
        self.app.img_paths = []  # clean up the image path list
        self.app.xml_paths = []  # clean up the xml list

        Clock.schedule_once(self.update_scrollview, 0)

    def update_scrollview(self, *args):
        # Unload camera
        if platform == "android":
            self.app.root.get_screen("camera").ids.camera_pre._camera._release_camera()

        self.app.root.get_screen("image_page").ids.image_box.clear_widgets() # clean up widgets

        self.files = os.listdir(XML_PATH)
        text_lables = [os.path.splitext(file)[0] for file in self.files]
        scroll_box = self.ids.scroll_box
        scroll_box.clear_widgets()

        for file, text in zip(self.files, text_lables):
            # Create a horizontal box layout
            row = MDBoxLayout(
                orientation="horizontal",
                size_hint_y=None,
                height=dp(50),
                spacing=10,  # Adjust spacing between label and button
            )

            l_name = MDLabel(
                text=text,
                size_hint_x=0.9,  # Make label take most space
                size_hint_y=None,
                height=dp(50),
                halign="center",
            )

            b_delete = MDIconButton(
                icon="delete-outline",
                on_release=lambda func, path=os.path.join(XML_PATH, file): self.confirm_delete(path),
                size_hint_x=None,
                pos_hint={"center_y": 0.5},
                theme_icon_color="Custom",
                icon_color=(1, 0, 0, 1),
            )

            b_export = MDIconButton(
                icon="export-variant",
                on_release=lambda func, path=os.path.join(XML_PATH, file): self.export_file(path),
                size_hint_x=None,
                pos_hint={"center_y": 0.5},
                theme_icon_color="Custom",
                icon_color=(0, 1, 1, 1),
            )

            row.add_widget(l_name)
            row.add_widget(b_delete)
            row.add_widget(b_export)
            scroll_box.add_widget(row)  # Add row instead of individual widgets

    def export_file(self, path: str, btn=None):
        """
        Save a file to Android External Storage
        Args:
            musicxml(bool): export as musicxml
            idx(int): index of the element in self.files

            Returns:
                None
        """
        # export (.musicxml)
        print(f"Trying to export {path}")
        if platform == 'android':
            self.share_file(path)
        else:
            print("Exporting is not implemented on Desktop")

    def share_file(self, path: str):
        """
        Share a file from a file path using android share sheets.
        Based on https://github.com/Android-for-Python/share_send_example/
        Args:
            path(str): path to file located in app storage.
        """
        uri = SharedStorage().copy_to_shared(path)
        ShareSheet().share_file(uri)

    def confirm_delete(self, path: str):
        """
        creates a special pop-up with two buttons; one is bound to delete an element in the scrollview, the other to cancel
        Args:
            idx(int): index of the element that we want to be deleted
        """
        print(f"Trying to delete {path}")
        self.dialog_delete = MDDialog(
            text="Are you sure you want to delete this scan? This cannot be undone.",
            buttons=[
                MDFlatButton(text="CANCEL", on_release=lambda dt: self.dialog_delete.dismiss()),
                MDFlatButton(text="CONFIRM", on_release=lambda func: self.delete_element(path)),
            ],
        )
        self.dialog_delete.open()

    def delete_element(self, path: str):
        """
        Deletes a certain element of the scrollview
        Args:
            index(int): index of the element in the scrollview that should be deleted
        """
        os.remove(path)
        self.dialog_delete.dismiss()
        self.update_scrollview()


class CameraPage(Screen):
    def on_enter(self):
        """
        Restores the camerawidget
        """
        self.app = MDApp.get_running_app()

        # Reload camera
        parent = self.ids.camera_pre.parent
        parent.remove_widget(self.ids.camera_pre)
        new_cam = KvCam(fit_mode="contain", play=True)
        self.ids.camera_pre = new_cam
        parent.add_widget(new_cam, index=0)

    def display_img(self, path):
        """
        displays the taken image in the image_box
        """
        # Schedule UI updates on the main thread
        Clock.schedule_once(lambda dt: self.app.root.get_screen("image_page")._display_img(path), 0)

    def take_picture(self, filename=f"image-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"):
        """Take an image"""
        take_picture(self.ids.camera_pre, self.display_img, filename)


class ProgressPage(Screen):
    def on_enter(self):
        app = MDApp.get_running_app()
        self.ids.title.text = ""
        # Unload camera
        if platform == "android":
            screen = app.root.get_screen("camera")
            screen.ids.camera_pre._camera._release_camera()
            screen.remove_widget(screen.ids.camera_pre)

    def update_progress_bar(self):
        """
        Update the progress bar used while running homr
        """
        self.update_progress_event = Clock.schedule_interval(
            lambda dt: self._update_progress_bar_status(), 0.1
        )

    def _update_progress_bar_status(self):
        app = MDApp.get_running_app()
        # Update the UI
        self.ids.progress_bar.value = int(appdata.homr_progress)
        self.ids.progress_label.text = str(appdata.homr_state)
        # Check if the thread is finished
        if not app.ml_thread.is_alive():
            # Stop the scheduled updates
            Clock.unschedule(self.update_progress_event)


class SettingsPage(Screen):
    def on_leave(self):
        app = MDApp.get_running_app()
        gpu = self.ids.checkbox_gpu.active
        use_xnnpack = self.ids.checkbox_xnnpack.active
        num_threads = self.ids.slider_threads.value
        if appdata.threads != num_threads or appdata.xnnpack != use_xnnpack or appdata.gpu != gpu:
            appdata.settings_changed = True
        else:
            appdata.settings_changed = False

        if gpu and not appdata.gpu:
            app.show_info("Please restart the app to use GPU acceleration")

        appdata.threads = int(num_threads)
        appdata.xnnpack = use_xnnpack
        appdata.gpu = gpu
        appdata.save_settings()


class OSSLicensePage(Screen):
    pass


class LicensePage(Screen):
    pass


class LicensePageButton(Screen):
    def agree_license(self):
        """Function that is triggered when the user agreed to the license"""
        app = MDApp.get_running_app()
        appdata.agreed = True
        appdata.save_settings()
        app.change_screen("landing")


class EditImagePage(Screen):
    def delete_image(self, img_idx):
        app = MDApp.get_running_app()
        del app.img_paths[img_idx]
        if not app.img_paths:
            app.change_screen("camera")
        else:
            app.show_toast(f"Deleted Image {img_idx+1}")
        self.ids.image_box.remove_widget(self.ids.image_box.slides[img_idx])
    
    def _display_img(self, path):
        """
        Internal method that actually performs the UI updates
        """
        # display screen to image_page
        app = MDApp.get_running_app()
        app.change_screen("image_page")
        downscale_cv2(path, 0.25)
        app.img_paths.append(path)

        # downscale image to save time during rendering
        buf, self.size, text_res = downscale_cv2(path, 0.25)

        # create texture from buffer
        texture = Texture.create(size=text_res, colorfmt="rgb")
        texture.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")

        # create Image widget
        img_widget = Image(fit_mode="contain")
        self.ids.image_box.add_widget(img_widget)

        # and set texture
        img_widget.texture = texture


class DownloadPage(Screen):
    def update_download_bar(self, camera_page):
        """
        Start the periodic update of the progress bar.
        """
        # Schedule the update function to run 10 times per second
        self.update_download_event = Clock.schedule_interval(
            lambda dt: self._update_download_bar_status(camera_page), 0.1
        )

    def _update_download_bar_status(self, camera_page):
        app = MDApp.get_running_app()

        # Update the UI
        self.ids.download_bar.value = int(appdata.download_progress)
        self.ids.download_label.text = str(appdata.downloaded_assets)

        # Check if the thread is finished
        if not app.download_thread.is_alive():
            # Stop the scheduled updates
            Clock.unschedule(self.update_download_event)

            if appdata.downloaded_assets.startswith("A"):  # Error occurd
                app.change_screen("landing")
                app.show_info(appdata.downloaded_assets)
            elif camera_page:
                app.change_screen("camera")
            else:
                app.change_screen("settings")


class License(RecycleView):
    # shows the license using a recycling view widget for better performance
    def __init__(self, **kwargs):
        super(License, self).__init__(**kwargs)
        self.dark_mode = get_sys_theme() == "Dark"

        def estimate_height(text, font_size=10, width=300):
            # Very rough estimate: average characters per line
            avg_char_width = font_size * 0.6
            approx_chars_per_line = width / avg_char_width
            lines = max(1, int(len(text) / approx_chars_per_line) + 1)
            line_height = sp(font_size) * 0.9
            return int(lines * line_height)

        with open(os.path.join(APP_PATH, "license_text.txt"), "r", encoding="utf-8") as file:
            lines = [line.rstrip("\n") for line in file]  # only remove trailing newline

        self.data = [
            {
                "text": line,
                "size": (None, estimate_height(line)),
                "color": (1, 1, 1, 1) if self.dark_mode else (0, 0, 0, 1),
            }
            for line in lines
        ]


class OSS_Licenses(RecycleView):
    # shows the open source licenses using a recycling view widget for better performance
    def __init__(self, **kwargs):
        super(OSS_Licenses, self).__init__(**kwargs)
        self.dark_mode = get_sys_theme() == "Dark"

        def estimate_height(text, font_size=10, width=300):
            # Very rough estimate: average characters per line
            avg_char_width = font_size * 0.6
            approx_chars_per_line = width / avg_char_width
            lines = max(1, int(len(text) / approx_chars_per_line) + 1)
            line_height = sp(font_size) * 0.9
            return int(lines * line_height)

        with open(os.path.join(APP_PATH, "oss_licenses.txt"), "r", encoding="utf-8") as file:
            lines = [line.rstrip("\n") for line in file]  # only remove trailing newline

        self.data = [
            {
                "text": line,
                "size": (None, estimate_height(line)),
                "color": (1, 1, 1, 1) if self.dark_mode else (0, 0, 0, 1),
            }
            for line in lines
        ]


# App class
class Andromr(MDApp):
    # Setup methods
    def build(self):
        self.setup()
        # set the app size to a phone size if on windows
        if platform == "win" or platform == "linux":
            Window.size = (350, 680)

        # load the file
        self.sm = Builder.load_file("main.kv")

        self.menu = MDDropdownMenu(
            items=[
                {
                    "viewclass": "OneLineListItem",
                    "text": "License",
                    "height": dp(48),
                    "on_release": lambda x="licensepage": self.change_screen(x),
                },
                {
                    "viewclass": "OneLineListItem",
                    "text": "About",
                    "height": dp(48),
                    "on_release": lambda x="osslicensepage": self.change_screen(x),
                },
            ]
        )
        # if the user hasn't agreed to the license
        if not appdata.agreed:
            # show him the screen
            self.sm.current = "licensepagebutton"

        return self.sm

    def setup(self):
        """
        Initializes all the attributes of the class needed
        """
        self.title = "Andromr"
        self.appdata = appdata

        # widgets
        self.last_screen = deque(maxlen=4)
        self.returnables = [
            "landing",
            "camera",
            "settings",
            "osslicensepage",
            "licensepage",
            "image_page",
        ]

        self.img_paths = []
        self.xml_paths = []

        # themes
        self.theme_cls.primary_palette = "LightGreen"
        self.theme_cls.theme_style = get_sys_theme()
        self.theme_cls.material_style = "M3"
        if platform == "android":
            self.bottom_pad = self.nav_bar_height_dp()
        else:
            self.bottom_pad = 0

    def on_start(self):
        Window.bind(on_keyboard=self.on_custom_back)

    def nav_bar_height_dp(self, offset=0, default=32) -> float:
        """
        Return navigation-bar height in *dp*.
        Otherwise the navigation bar might be overlapping with the buttons on the bottom
        """
        PythonActivity = autoclass("org.kivy.android.PythonActivity")
        activity = PythonActivity.mActivity
        resources = activity.getResources()
        res_id = resources.getIdentifier("navigation_bar_height", "dimen", "android")
        if res_id > 0:
            px = resources.getDimensionPixelSize(res_id)
            density = resources.getDisplayMetrics().density
            return (px / density) + offset
        return dp(default)

    def on_custom_back(self, window, key, scancode, codepoint, modifiers):
        print(f"Key pressed: {key}")  # Should print 27 for back button
        if key == 27:
            self.previous_screen()
            return True
        return False

    # UI methods
    def change_screen(self, screen_name, btn=None):
        """
        Change the screen displayed by Kivy
        Args:
            screen_name(str): name of the screen you want to change to
        """
        self.menu.dismiss()

        # record current screen
        if self.sm.current in self.returnables:
            self.last_screen.append(self.sm.current)

        # set new screen
        self.sm.current = screen_name

    def previous_screen(self, btn=None):
        """Switch to the previous screen"""
        if len(self.last_screen) != 0 and self.sm.current in self.returnables:
            self.sm.current = self.last_screen.pop()

    def show_info(self, text: str, title: str = ""):
        """
        show a pop-up with a button; better for longer texts
        Args:
            text(str): Text to display
            title(str): Defaults to empty; if used displays a title in the pop-up
        """
        self.dialog_information = MDDialog(
            title=title,
            text="",
            buttons=[
                MDFlatButton(text="OK", on_release=lambda dt: self.dialog_information.dismiss())
            ],
        )

        self.dialog_information.text = text
        self.dialog_information.title = title
        self.dialog_information.open()

    def show_toast(self, text: str):
        """
        show a simple text message on the bottom
        Args:
            text(str): Text wanted to be displayed
        """
        toast(text=text)

    def open_menu(self, button):
        self.menu.caller = button
        self.menu.open()

    def menu_callback(self):
        self.menu.dismiss()

    # Homr methods
    def start_inference(self):
        path = self.img_paths[0]
        del self.img_paths[0]
        # set the progress bar to 0
        appdata.progress = 0

        # go to progress page
        self.change_screen("progress")

        appdata.homr_running = True

        # start the ml thread and the progress thread seperatly from each other
        self.ml_thread = Thread(target=self.homr_call, args=(path,), daemon=True)
        self.progress_thread = Thread(
            target=self.root.get_screen("progress").update_progress_bar, daemon=True
        )
        self.ml_thread.start()
        self.progress_thread.start()

    def homr_call(self, path: str):
        """
        calls the homr (optical music recognition software) and returns when finished to the landing page
        Args:
            path(str): path to the image
            output_path(str): path where the musicxml is stored
        """
        # run homr with try-except on android
        # else we run it without for easier debugging
        if platform == "android":
            try:
                self._homr_call(path)
            except Exception as e:
                error_msg = f"An error occured during inference: {e}"
                Clock.schedule_once(lambda dt: self.show_info(text=error_msg))
                print(e)
        else:
            self._homr_call(path)

        if self.img_paths:
            self.start_inference()
        else:
            if len(self.xml_paths) >= 2:
                out_path = os.path.join(
                    XML_PATH, f"merged-music-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
                )

                merge_xmls(self.xml_paths, out_path)

                for file in self.xml_paths:
                    os.remove(file)
            Clock.schedule_once(lambda dt: self.change_screen("landing"))

    def _homr_call(self, path):
        return_path = homr(path)
        appdata.homr_running = False

        if self.root.get_screen("progress").ids.title.text == "":
            # if there's no user given title we give it a unique id based on time
            music_title = f"transcribed-music-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        else:
            # otherwise we set it to the users title
            music_title = str(self.root.get_screen("progress").ids.title.text)

        # rename the file
        os.rename(
            return_path,
            os.path.join(XML_PATH, f"{music_title}.musicxml"),
        )

        self.xml_paths.append(os.path.join(XML_PATH, f"{music_title}.musicxml"))

    def start_download(self, camera_page=False):
        self.dialog_download.dismiss()
        self.download_thread = Thread(target=download_weights, daemon=True)
        update = Thread(
            target=self.root.get_screen("downloadpage").update_download_bar,
            args=(camera_page,),
            daemon=True,
        )
        self.download_thread.start()
        update.start()
        Clock.schedule_once(lambda dt: self.change_screen("downloadpage"))

    def check_download_assets(self, camera_page=False):
        """
        If not all tflite models are downloaded it will create a Dialog informing the user
        that the App wants to download the models. If the user allows it, the app will switch
        to the downloadpage (which displays a progress bar).
        If all tflite models are downloaded it will directly switch to the camera screen.
        """
        if check_for_missing_models():
            self.dialog_download = MDDialog(
                text="You need to download assets before converting an image to .musicxml. You can also download later in the settings tab.",
                buttons=[
                    MDFlatButton(
                        id="cancel",
                        text="CANCEL",
                        on_release=lambda dt: self.dialog_download.dismiss(),
                    ),
                    MDFlatButton(
                        text="DOWNLOAD NOW",
                        on_release=lambda dt: self.start_download(camera_page),
                    ),
                ],
            )
            self.dialog_download.open()
        elif camera_page:
            self.change_screen("camera")
        else:
            self.show_toast("You already downloaded all assets")


if __name__ == "__main__":
    Andromr().run()
