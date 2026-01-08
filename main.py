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
from kivy.graphics import Line, Color
from kivy.uix.widget import Widget
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
from homr.transformer.encoder_inference import preload_cnn_encoder
from globals import APP_PATH, XML_PATH, appdata
from utils import crop_image_by_corners, get_sys_theme, downscale_cv2

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
    preload_cnn_encoder(num_threads=appdata.threads, use_gpu=appdata.gpu)
    preload_segnet(num_threads=appdata.threads, use_gpu=appdata.gpu)

else:
    # Custom placeholder widget for Desktop
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
        function(filename)


# Classes of Screens used by kivy
class LandingPage(Screen):
    pass


class CameraPage(Screen):
    # Unload the camera to stop recording
    # to improve gpu performance
    def on_leave(self):
        if platform == "android":
            self.ids.camera_pre._camera._release_camera()


class ProgressPage(Screen):
    pass


class SettingsPage(Screen):
    pass


class OSSLicensePage(Screen):
    pass


class LicensePage(Screen):
    pass


class LicensePageButton(Screen):
    pass


class EditImagePage(Screen):
    pass


class DownloadPage(Screen):
    pass


class MovableMDIconButton(MDIconButton):
    # custom button that is movable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_x, self.max_y = Window.system_size
        self.app = MDApp.get_running_app()

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            self.pos = (touch.x - self.width / 2, touch.y - self.height / 2)

            # update button positions (of unmovable)
            # get buttons
            btn0 = self.app.root.get_screen("image_page").ids.btn0
            btn1 = self.app.root.get_screen("image_page").ids.btn1
            btn2 = self.app.root.get_screen("image_page").ids.btn2
            btn3 = self.app.root.get_screen("image_page").ids.btn3

            # and set their position so they form a rectangle with btn0 and btn3
            btn1.pos = btn0.pos[0], btn3.pos[1]
            btn2.pos = btn3.pos[0], btn0.pos[1]

            # update lines
            line_drawer = self.app.root.get_screen("image_page").ids.line_drawer
            line_drawer.update_lines()


class UnmovableMDIconButton(MDIconButton):
    # custom button that is not movable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_x, self.max_y = Window.system_size

    def update_pos(self, x, y):
        self.pos = x, y


class LineDrawer(Widget):
    # draws lines between buttons
    # used in the EditImagePage
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app = MDApp.get_running_app()

    def update_lines(self):
        """
        updates the position of the lines by:
        1. removing lines
        2. creating a list with all positions
        3. drawing the line
        """
        self.canvas.clear()
        with self.canvas:
            Color(1, 0, 0, 1)  # Red
            # Get positions from buttons
            btns = [
                self.app.root.get_screen("image_page").ids.btn0,
                self.app.root.get_screen("image_page").ids.btn1,
                self.app.root.get_screen("image_page").ids.btn3,
                self.app.root.get_screen("image_page").ids.btn2,
            ]

            points = []
            for btn in btns:
                x = btn.center_x
                y = btn.center_y
                points.extend([x, y])

            # Close the shape by connecting last to first
            points.extend([btns[0].center_x, btns[0].center_y])

            Line(points=points, width=1.5)


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

        with open(os.path.join(APP_PATH, "license.txt"), "r", encoding="utf-8") as file:
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

        Window.bind(on_keyboard=self.on_custom_back)

        return self.sm

    def setup(self):
        """
        Initializes all the attributes of the class needed
        """
        self.title = "Andromr"
        self.appdata = appdata

        # create files list (used by the scorllview)
        self.files = os.listdir(XML_PATH)

        # widgets
        self.text_lables = [os.path.splitext(file)[0] for file in self.files]
        self.last_screen = deque(maxlen=10)
        self.returnables = [
            "landing",
            "camera",
            "settings",
            "osslicensepage",
            "licensepage",
            "image_page",
        ]

        # themes
        self.theme_cls.primary_palette = "LightGreen"
        self.theme_cls.theme_style = get_sys_theme()
        self.theme_cls.material_style = "M3"
        if platform == "android":
            self.bottom_pad = self.nav_bar_height_dp()
        else:
            self.bottom_pad = 0

    def on_start(self):
        # Update Scrollview on start
        self.update_scrollview()

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
        if key == 27:  # back gesture on android
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

        # update the scrollview on the landing page
        if screen_name == "landing":
            self.update_scrollview()

        if screen_name == "camera":
            self._restore_camera()

    def previous_screen(self, btn=None):
        """Switch to the previous screen"""
        if len(self.last_screen) != 0 and self.sm.current in self.returnables:
            self.sm.current = self.last_screen.pop()

    def update_scrollview(self):
        """Function that updates the scrollview on the landing page"""
        self.files = os.listdir(XML_PATH)
        self.text_lables = [os.path.splitext(file)[0] for file in self.files]
        scroll_box = self.sm.get_screen("landing").ids.scroll_box
        scroll_box.clear_widgets()

        for index, text in enumerate(self.text_lables):
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
                on_release=lambda func: self.confirm_delete(index),
                size_hint_x=None,
                pos_hint={"center_y": 0.5},
                theme_icon_color="Custom",
                icon_color=(1, 0, 0, 1),
                # ripple_scale=0
            )

            b_export = MDIconButton(
                icon="export-variant",
                on_release=lambda func: self.export_file(idx=index),
                size_hint_x=None,
                pos_hint={"center_y": 0.5},
                theme_icon_color="Custom",
                icon_color=(0, 1, 1, 1),
                # ripple_scale=0
            )

            row.add_widget(l_name)
            row.add_widget(b_delete)
            row.add_widget(b_export)
            scroll_box.add_widget(row)  # Add row instead of individual widgets

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

    def update_progress_bar(self):
        """
        Update the progress bar used while running homr
        """
        while appdata.homr_running:
            self.root.get_screen("progress").ids.progress_bar.value = appdata.homr_progress
            self.root.get_screen("progress").ids.progress_label.text = appdata.homr_state
            sleep(0.1)

    def update_download_bar(self, camera_page):
        """
        Update the progress bar used while downloading models
        """
        while appdata.download_running:
            self.root.get_screen("downloadpage").ids.download_bar.value = int(
                appdata.download_progress
            )
            self.root.get_screen("downloadpage").ids.download_label.text = str(
                appdata.downloaded_assets
            )
            sleep(0.02)
        if appdata.downloaded_assets == "failure":
            Clock.schedule_once(lambda dt: self.change_screen("landing"))
        elif camera_page:
            Clock.schedule_once(lambda dt: self.change_screen("camera"))
        else:
            Clock.schedule_once(lambda dt: self.change_screen("settings"))

    # Button click methods
    def export_file(self, idx, btn=None):
        """
        Save a file to Android External Storage
        Args:
            musicxml(bool): export as musicxml
            idx(int): index of the element in self.files

            Returns:
                None
        """
        # export (.musicxml)
        self.share_file(os.path.join(XML_PATH, self.files[idx]))

    def share_file(self, path: str):
        """
        Share a file from a file path using android share sheets.
        Based on https://github.com/Android-for-Python/share_send_example/
        Args:
            path(str): path to file located in app storage.
        """
        uri = SharedStorage().copy_to_shared(path)
        ShareSheet().share_file(uri)

    def confirm_delete(self, idx: int):
        """
        creates a special pop-up with two buttons; one is bound to delete an element in the scrollview, the other to cancel
        Args:
            idx(int): index of the element that we want to be deleted
        """
        self.dialog_delete = MDDialog(
            text="Are you sure you want to delete this scan? This cannot be undone.",
            buttons=[
                MDFlatButton(text="CANCEL", on_release=lambda dt: self.dialog_delete.dismiss()),
                MDFlatButton(text="CONFIRM", on_release=lambda func: self.delete_element(idx)),
            ],
        )
        self.dialog_delete.open()

    def delete_element(self, index: int):
        """
        Deletes a certain element of the scrollview
        Args:
            index(int): index of the element in the scrollview that should be deleted
        """
        os.remove(os.path.join(XML_PATH, self.files[index]))
        self.dialog_delete.dismiss()
        self.update_scrollview()

    def save_settings(self, num_threads, use_xnnpack, gpu, btn=None):
        if appdata.threads != num_threads or appdata.xnnpack != use_xnnpack or appdata.gpu != gpu:
            appdata.settings_changed = True
        else:
            appdata.settings_changed = False

        appdata.threads = int(num_threads)
        appdata.xnnpack = use_xnnpack
        appdata.gpu = gpu
        appdata.save_settings()
        self.change_screen("landing")

    def agree_license(self):
        """Function that is triggered when the user agreed to the license"""
        appdata.agreed = True
        appdata.save_settings()
        self.change_screen("landing")

    def open_menu(self, button):
        self.menu.caller = button
        self.menu.open()

    def menu_callback(self):
        self.menu.dismiss()

    # Crop Image methods
    def crop(self):
        """
        crop the displayed image based on the positions of the buttons
        """
        # get button positions
        pos_btns = [
            self.convert_to_img_pos(self.root.get_screen("image_page").ids.btn0.center),
            self.convert_to_img_pos(self.root.get_screen("image_page").ids.btn1.center),
            self.convert_to_img_pos(self.root.get_screen("image_page").ids.btn2.center),
            self.convert_to_img_pos(self.root.get_screen("image_page").ids.btn3.center),
        ]

        # call the crop function from utils.py
        crop_image_by_corners(self.img_path, pos_btns, self.img_path)
        self.start_inference(self.img_path)

    def convert_to_img_pos(self, pos):
        """
        Converts the on-screen coordinates to image coordinates

        Args:
            pos(float/int): position to convert

        Returns:
            position on image
        """

        # unpack values
        x, y = pos

        # depending on where there's space we need to subtract the space from the coordinate of the button
        if self.vertical_touch:
            y -= self.space_left
        else:
            x -= self.space_left

        # max: if the value is negative we want it to be 0
        # min: we obviously don't want the user to be able to make the image bigger than it is.
        # That's why the new side_length needs to be smaller than the side_length of the image
        x = max(0, min(x * self.scale, self.size[0]))
        y = max(0, min(y * self.scale, self.size[1]))

        return [
            x,
            abs(self.size[1] - y),
        ]  # the abs thing is a workaround because i read with (0,0) at bottom left while numpy/pillow thinks it's at(0, img_height)

    def set_cutter_btn(self, instance=None):
        """
        Displays the drag+drop buttons on image-corners
        """

        # get original size of the image
        img_x, img_y = self.size

        # get screen size
        win_x, win_y = Window.size

        # calculate on which size the image and the screen touch
        if img_x / win_x > img_y / win_y:
            # sides touch vertically
            self.vertical_touch = True
            self.scale = img_x / win_x

            dist_y = win_x * img_y / img_x
            dist_x = win_x

            # calculate how much space is left
            self.space_left = (win_y - dist_y) / 2  # half beause we only need one distance

            offset = 50  # buttons should be easily touchable. that's why there's an offset to the edge of the screen

            # move buttons to the correct point
            self.root.get_screen("image_page").ids.btn0.center = (
                offset,
                self.space_left + offset,
            )
            self.root.get_screen("image_page").ids.btn3.center = (
                dist_x - offset,
                self.space_left + dist_y - offset,
            )
            self.root.get_screen("image_page").ids.btn2.center = (
                offset,
                self.space_left + dist_y - offset,
            )
            self.root.get_screen("image_page").ids.btn1.center = (
                dist_x - offset,
                self.space_left + offset,
            )

        else:
            # sides touch horizontally - currently not working
            self.vertical_touch = False
            self.scale = img_y / win_y
            dist_x = win_y * img_x / img_y
            dist_y = win_y

            # calculate how much space is left
            self.space_left = (win_x - dist_x) / 2  # half because we need only one distance

            offset = 50
            self.root.get_screen("image_page").ids.btn0.center = (
                self.space_left - 25,
                0,
            )
            self.root.get_screen("image_page").ids.btn3.center = (
                self.space_left + dist_x - 25,
                dist_y - 50,
            )
            self.root.get_screen("image_page").ids.btn2.center = (
                self.space_left - 25,
                dist_y - 50,
            )
            self.root.get_screen("image_page").ids.btn1.center = (
                self.space_left + dist_x - 25,
                0,
            )

        # and display the lines
        line_drawer = self.root.get_screen("image_page").ids.line_drawer
        line_drawer.update_lines()

    # Camera methods
    def display_img(self):
        """
        displays the taken image in the image_box
        """
        # display screen to image_page
        self.change_screen("image_page")

        # remove the image loaded previously to the image box
        self.root.get_screen("image_page").ids.image_box.clear_widgets()

        # downscale image to save time during rendering
        buf, self.size, text_res = downscale_cv2(self.img_path, 0.25)

        # create texture from buffer
        texture = Texture.create(size=text_res, colorfmt="rgb")
        texture.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")

        # create Image widget
        img_widget = Image(fit_mode="contain")
        self.root.get_screen("image_page").ids.image_box.add_widget(img_widget)

        # and set texture
        img_widget.texture = texture

        # move the buttons to the correct location
        self.set_cutter_btn()

    def img_taken(self, img_path):
        """
        Function running after the image is taken, see android_camera_api.py
        Args:
            filename(str/Path): path to the image
        """
        self.img_path = img_path
        Clock.schedule_once(lambda dt: self.display_img())

    def capture(self, filename="test_cropped.jpg"):
        """Take an image"""
        take_picture(self.root.get_screen("camera").ids.camera_pre, self.img_taken, filename)

    # Homr methods
    def start_inference(self, path_to_image: str):
        # set the progress bar to 0
        appdata.progress = 0

        # go to progress page
        self.change_screen("progress")

        # reset text-field contents
        self.root.get_screen("progress").ids.title.text = ""
        appdata.homr_running = True

        # start the ml thread and the progress thread seperatly from each other
        self.ml_thread = Thread(target=self.homr_call, args=(path_to_image,), daemon=True)
        self.progress_thread = Thread(target=self.update_progress_bar, daemon=True)
        self.ml_thread.start()
        self.progress_thread.start()

    def homr_call(self, path: str):
        """
        calls the homr (optical music recognition software) and returns when finished to the landing page
        Args:
            path(str): path to the image
            output_path(str): path where the musicxml is stored
        """
        # run homr
        try:
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

            # and update self.files (needed for the scorllview)
            self.files = os.listdir(XML_PATH)
            self.text_lables = [os.path.splitext(file)[0] for file in self.files]

        except Exception as e:
            print(e)

        # switch to landing screen
        Clock.schedule_once(lambda dt: self.change_screen("landing"))

    def start_download(self, camera_page=False):
        self.dialog_download.dismiss()
        download = Thread(target=download_weights, daemon=True)
        update = Thread(target=self.update_download_bar, args=(camera_page,), daemon=True)
        download.start()
        update.start()
        Clock.schedule_once(lambda dt: self.change_screen("downloadpage"))

    def check_download_assets(self, camera_page=False):
        """
        If not all tflite models are downloaded it will create a Dialog informing the user
        that the App wants to download something. If the user allows to the app will switch
        to a Screen displaying a Progressbar.
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

    def _restore_camera(self, dt=None):
        """
        Restores the camerawidget
        """
        camera_screen = self.root.get_screen("camera")
        parent = camera_screen.ids.camera_pre.parent
        parent.remove_widget(camera_screen.ids.camera_pre)
        new_cam = KvCam(fit_mode="contain", play=True)
        camera_screen.ids.camera_pre = new_cam
        parent.add_widget(new_cam, index=0)


if __name__ == "__main__":
    Andromr().run()
