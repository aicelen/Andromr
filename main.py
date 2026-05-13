# main file of the app
# Andromr class is the main class of the app

# Kivy imports
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.animation import Animation
from kivy.factory import Factory
from kivy.properties import NumericProperty
from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDIconButton, MDFloatingActionButtonSpeedDial
from kivymd.uix.button.button import (
    MDFloatingBottomButton,
    MDFloatingLabel,
    MDFloatingRootButton,
)
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
from kivy.utils import get_color_from_hex
from kivymd.uix.slider import MDSlider
from kivymd.utils.set_bars_colors import set_bars_colors

# Built-in imports
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import os
from datetime import datetime
from time import perf_counter
from collections import deque
from itertools import zip_longest
from pathlib import Path

# Own imports
from homr.main import download_weights, homr, check_for_missing_models
from homr.relieur import merge_xmls
from homr.simple_logging import eprint
from validation.rate_validation_result import rate_folder
from globals import APP_PATH, XML_PATH, IMAGE_PATH, MODEL_STORAGE, appdata, APP_STORAGE
from utils import get_sys_theme, safe_filename, pdf_to_img, downscale_cv2

if platform == "android":
    from androidstorage4kivy import SharedStorage, ShareSheet  # type: ignore
    from jnius import autoclass  # type: ignore
    from android.activity import bind as activity_bind  # type: ignore
    from android_document_scanner import document_scan_result, start_document_scan

    Intent = autoclass("android.content.Intent")
    PythonActivity = autoclass("org.kivy.android.PythonActivity")
    BufferedInputStream = autoclass("java.io.BufferedInputStream")
    activity = PythonActivity.mActivity

else:
    from plyer import filechooser


class AlwaysHintSlider(MDSlider):
    """MDSlider variant that keeps the value hint visible when not dragging."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Clock.schedule_once(self._show_hint_box)

    def on_active(self, *args):
        super().on_active(*args)
        Clock.schedule_once(self._show_hint_box)

    def _show_hint_box(self, *args):
        if self.hint and "hint_box" in self.ids:
            self.ids.hint_box.opacity = 1


class MovableFloatingActionButtonSpeedDial(MDFloatingActionButtonSpeedDial):
    bottom_pad = NumericProperty(0)

    def on_bottom_pad(self, *args):
        self._update_pos_buttons(None, Window.width, Window.height)
        Clock.schedule_once(self._position_root_button)

    def on_anchor(self, *args):
        Clock.schedule_once(self._position_root_button)

    def on_parent(self, *args):
        if self.parent:
            self.parent.bind(size=self._position_root_button)
        Clock.schedule_once(self._position_root_button)

    def _position_root_button(self, root_button=None, *args):
        root_buttons = (
            [root_button]
            if isinstance(root_button, MDFloatingRootButton)
            else [widget for widget in self.children if isinstance(widget, MDFloatingRootButton)]
        )
        for root_button in root_buttons:
            root_button.y = dp(20 + self.bottom_pad)
            if self.anchor == "right":
                parent_width = (
                    self.parent.width if self.parent and self.parent.width else Window.width
                )
                root_button.x = parent_width - (root_button.width or dp(56)) - dp(20)

    def set_pos_root_button(self, instance_floating_root_button) -> None:
        def set_pos_root_button(*args):
            self._position_root_button(instance_floating_root_button)

        Clock.schedule_once(set_pos_root_button)

    def open_stack(self, instance_floating_root_button) -> None:
        for widget in self.children:
            if isinstance(widget, MDFloatingLabel):
                Animation.cancel_all(widget)

        if self.state != "open":
            y = 0
            label_position = dp(54)
            bottom_offset = dp(self.bottom_pad)
            anim_buttons_data = {}
            anim_labels_data = {}

            for widget in self.children:
                if isinstance(widget, MDFloatingBottomButton):
                    y += dp(56)
                    widget.y = widget.height + y + bottom_offset
                    if not self._anim_buttons_data:
                        anim_buttons_data[widget] = Animation(
                            opacity=1,
                            d=self.opening_time,
                            t=self.opening_transition,
                        )
                elif isinstance(widget, MDFloatingLabel):
                    label_position += dp(56)
                    if not self._label_pos_y_set:
                        widget.y = label_position + bottom_offset
                        widget.x = Window.width - widget.width - dp(86)
                    if not self._anim_labels_data:
                        anim_labels_data[widget] = Animation(opacity=1, d=self.opening_time)
                elif isinstance(widget, MDFloatingRootButton) and self.root_button_anim:
                    Animation(
                        rotate_value_angle=-45,
                        d=self.opening_time_button_rotation,
                        t=self.opening_transition_button_rotation,
                    ).start(widget)

            if anim_buttons_data:
                self._anim_buttons_data = anim_buttons_data
            if anim_labels_data and not self.hint_animation:
                self._anim_labels_data = anim_labels_data

            self.state = "open"
            self.dispatch("on_open")
            self.do_animation_open_stack(self._anim_buttons_data)
            self.do_animation_open_stack(self._anim_labels_data)
            if not self._label_pos_y_set:
                self._label_pos_y_set = True
        else:
            self.close_stack()


Factory.register("AlwaysHintSlider", cls=AlwaysHintSlider)
Factory.register(
    "MovableFloatingActionButtonSpeedDial",
    cls=MovableFloatingActionButtonSpeedDial,
)


# Classes of Screens used by kivy
class LandingPage(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.app = MDApp.get_running_app()
        self.data = {
            "Scanner": [
                "camera",
                "on_release",
                lambda x: self.app.check_download_assets(scanner=True),
            ],
            "Open File": [
                "file-upload",
                "on_release",
                lambda x: self.app.check_download_assets(file_chooser=True),
            ],
        }

    def on_enter(self):
        self.app.img_paths = []  # clean up the image path list
        self.app.xml_paths = []  # clean up the xml list

        # remove all images in the image folder
        image_paths = os.listdir(IMAGE_PATH)
        for path in image_paths:
            os.remove(os.path.join(IMAGE_PATH, path))
            print(f"Removed {path}")

        Clock.schedule_once(self.update_scrollview, 0)

    def update_scrollview(self, *args):
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
                on_release=lambda func, path=os.path.join(XML_PATH, file): self.confirm_delete(
                    path
                ),
                size_hint_x=None,
                pos_hint={"center_y": 0.5},
                theme_icon_color="Custom",
                icon_color=get_color_from_hex("#D32F2F"),
            )

            b_export = MDIconButton(
                icon="export-variant",
                on_release=lambda func, path=os.path.join(XML_PATH, file): self.export_file(path),
                size_hint_x=None,
                pos_hint={"center_y": 0.5},
                theme_icon_color="Custom",
                icon_color=self.app.theme_cls.text_color,
            )

            row.add_widget(l_name)
            row.add_widget(b_export)
            row.add_widget(b_delete)
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
        if platform == "android":
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


class ProgressPage(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.app = MDApp.get_running_app()

    def on_enter(self):
        self.ids.title.text = ""

    def update_progress_bar(self):
        """
        Update the progress bar used while running homr
        """
        self.update_progress_event = Clock.schedule_interval(
            lambda dt: self._update_progress_bar_status(), 0.1
        )

    def _update_progress_bar_status(self):
        # Update the UI
        self.ids.progress_bar.value = int(appdata.homr_progress)
        self.ids.progress_label.text = str(appdata.homr_state)
        # Check if the thread is finished
        if self.app.homr_future.done():
            # Stop the scheduled updates
            Clock.unschedule(self.update_progress_event)
            if self.app.homr_future.result()[0]:
                self.app.show_info(text=self.app.homr_future.result()[1])
            Clock.schedule_once(lambda dt: self.app.change_screen("landing"))


class SettingsPage(Screen):
    def on_leave(self):
        self.get_settings()

    def get_settings(self):
        """
        Gets the settings (number of threads) and saves them to a json.
        """
        num_threads = self.ids.slider_threads.value
        if appdata.threads != num_threads:
            appdata.settings_changed = True
        else:
            appdata.settings_changed = False

        appdata.threads = int(num_threads)
        appdata.save_settings()

    def verify_homr(self):
        app = MDApp.get_running_app()
        self.get_settings()
        need_download = app.check_download_assets(file_chooser=False, validation=True)
        if not need_download:
            app.start_inference(
                path_to_images=["test_data/entertainer/entertainer.png"],
                out_paths=["test_data/entertainer/entertainer.musicxml"],
                verify=True,
            )

    def confirm_delete_models(self):
        if os.listdir(MODEL_STORAGE):
            self.confirm = MDDialog(
                text="Are you sure you want to delete all models. This requires you to redownload all of them.",
                buttons=[
                    MDFlatButton(
                        id="cancel",
                        text="CANCEL",
                        on_release=lambda dt: self.confirm.dismiss(),
                    ),
                    MDFlatButton(
                        text="DELETE",
                        on_release=lambda dt: self.delete_models(),
                    ),
                ],
            )
            self.confirm.open()

        else:
            app = MDApp.get_running_app()
            app.show_toast(f"No models downloaded")

    def delete_models(self):
        self.confirm.dismiss()
        app = MDApp.get_running_app()
        models = os.listdir(MODEL_STORAGE)
        for model in models:
            os.remove(os.path.join(MODEL_STORAGE, model))
            eprint(f"Deleted {model}")

        app.show_toast(f"Models deleted")


class OSSLicensePage(Screen):
    pass


class LicensePage(Screen):
    pass


class PrivacyPolicyPage(Screen):
    pass


class LicensePageButton(Screen):
    def agree_license(self):
        """Function that is triggered when the user agreed to the license"""
        app = MDApp.get_running_app()
        app.change_screen("privacypolicypagebutton")


class PrivacyPolicyPageButton(Screen):
    def agree_license(self):
        """Function that is triggered when the user agreed to the privacy policy"""
        app = MDApp.get_running_app()
        appdata.agreed = True
        appdata.save_settings()
        app.change_screen("landing")


class DownloadPage(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.app = MDApp.get_running_app()

    def update_download_bar(self, scanner, file_chooser):
        """
        Start the periodic update of the progress bar.
        """
        # Schedule the update function to run 10 times per second
        self.update_download_event = Clock.schedule_interval(
            lambda dt: self._update_download_bar_status(scanner, file_chooser), 0.1
        )

    def _update_download_bar_status(self, scanner, file_chooser):
        # Update the UI
        self.ids.download_bar.value = int(appdata.download_progress)
        self.ids.download_label.text = str(appdata.downloaded_assets)

        # Check if the thread is finished
        if self.app.future.done():
            # Stop the scheduled updates
            Clock.unschedule(self.update_download_event)
            if self.app.future.result()[0]:
                self.app.show_info(text=self.app.future.result()[1], title="Error")
                Clock.schedule_once(lambda dt: self.app.change_screen("landing"))
            elif scanner:
                if platform == "android":
                    start_document_scan()
                else:
                    self.pick_file()

            elif file_chooser:
                self.app.pick_file()
            else:
                self.app.change_screen("settings")


class EditFilePage(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.app = MDApp.get_running_app()
        self.imgs_to_display = []
        self.data = {
            "Scanner": [
                "camera",
                "on_release",
                lambda x: self.app.check_download_assets(scanner=True),
            ],
            "Open File": [
                "file-upload",
                "on_release",
                lambda x: self.app.check_download_assets(file_chooser=True),
            ],
        }

    def on_leave(self, *args):
        self.imgs_to_display = []
        self.ids.gen_xml.close_stack()
        self.ids.image_box.clear_widgets()
        return super().on_leave(*args)

    def delete_image(self, img_idx):
        print(f"Deleting image {img_idx} form {self.imgs_to_display}")
        del self.imgs_to_display[img_idx]
        if not self.imgs_to_display:
            self.app.change_screen("landing")
        else:
            self.app.show_toast(f"Deleted Image {img_idx + 1}")
        self.ids.image_box.remove_widget(self.ids.image_box.slides[img_idx])

    def _display_imgs(self, paths):
        """
        Internal method that actually performs the UI updates
        """
        for path in paths:
            self.imgs_to_display.append(path)

            # downscale image to save time during rendering
            buf, _, text_res = downscale_cv2(path, 0.25)

            # create texture from buffer
            texture = Texture.create(size=text_res, colorfmt="rgb")
            texture.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")

            # create Image widget
            img_widget = Image(fit_mode="contain")
            self.ids.image_box.add_widget(img_widget)

            # and set texture
            img_widget.texture = texture


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


class PrivacyPolicy(RecycleView):
    # shows the privacy policy using a recycling view widget for better performance
    def __init__(self, **kwargs):
        super(PrivacyPolicy, self).__init__(**kwargs)
        self.dark_mode = get_sys_theme() == "Dark"

        def estimate_height(text, font_size=10, width=300):
            # Very rough estimate: average characters per line
            avg_char_width = font_size * 0.6
            approx_chars_per_line = width / avg_char_width
            lines = max(1, int(len(text) / approx_chars_per_line) + 1)
            line_height = sp(font_size) * 0.9
            return int(lines * line_height)

        with open(os.path.join(APP_PATH, "privacy_policy.txt"), "r", encoding="utf-8") as file:
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
                    "text": "Privacy Policy",
                    "height": dp(48),
                    "on_release": lambda x="privacypolicypage": self.change_screen(x),
                },
                {
                    "viewclass": "OneLineListItem",
                    "text": "Open-Source Licenses",
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
        self.last_screens = deque(maxlen=4)
        self.previous_screen = None
        self.returnables = [
            "landing",
            "settings",
            "osslicensepage",
            "licensepage",
            "privacypolicypage",
            "edit_file",
        ]

        # themes
        self.theme_cls.primary_palette = "Orange"
        self.theme_cls.primary_hue = "300"
        self.theme_cls.theme_style = get_sys_theme()
        self.theme_cls.material_style = "M3"
        if platform == "android":
            self.bottom_pad = self.nav_bar_height_dp()
            set_bars_colors(
                self.theme_cls.primary_color,  # status bar color
                self.theme_cls.primary_color,  # navigation bar color
                "Dark" if self.theme_cls.theme_style == "Light" else "Light",
            )
        else:
            self.bottom_pad = 0

    def on_start(self):
        Window.bind(on_keyboard=self.on_custom_back)
        if platform == "android":
            activity_bind(on_activity_result=self.activity_result)
        print("starting")

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
            self.switch_to_previous_screen()
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
            self.last_screens.append(self.sm.current)

        self.previous_screen = self.sm.current

        # set new screen
        self.sm.current = screen_name

    def switch_to_previous_screen(self, btn=None):
        """Switch to the previous screen"""
        if len(self.last_screens) != 0 and self.sm.current in self.returnables:
            self.sm.current = self.last_screens.pop()

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
        toast(text=text, length_long=True)

    def open_menu(self, button):
        self.menu.caller = button
        self.menu.open()

    def menu_callback(self):
        self.menu.dismiss()

    # Homr methods
    def start_inference(self, path_to_images: list, out_paths: list = [], verify: bool = False):
        # go to progress page
        self.change_screen("progress")
        executor = ThreadPoolExecutor(max_workers=1)
        self.homr_future = executor.submit(self.run_homr, path_to_images, out_paths, verify)

        update = Thread(target=self.root.get_screen("progress").update_progress_bar, daemon=True)
        update.start()

    def run_homr(self, paths: list, out_paths: list, verify: bool):
        t0 = perf_counter()
        out_paths_final = []
        for path, out_path in zip_longest(paths, out_paths):
            if out_path is None:
                out_path = os.path.join(
                    XML_PATH,
                    f"transcribed-music-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.musicxml",
                )

            if platform == "android":
                try:
                    homr(path=path, output_path=out_path)
                except Exception as e:
                    eprint(e)
                    text = f"An error occured during inference: {e}"
                    return True, text
            else:
                homr(path=path, output_path=out_path)
            
            if not verify:
                os.remove(path)  # remove images
            out_paths_final.append(out_path)  # used for merging xmls

        time = perf_counter() - t0

        if verify:
            try:
                validation_metrics, n_errors = rate_folder(
                    "test_data/entertainer", compare_all=True
                )

                ser = validation_metrics.total_ser * 100

                if ser < 3:
                    text = f"Results are great. The average difference was {ser} with a total of {n_errors} failures. It took {round(time, 2)}s"
                else:
                    text = f"Results are bad. The average difference was {ser} with a total of {n_errors} failures. Please report this on Github: github.com/aicelen/Andromr. It took {round(time, 2)}s"
                return True, text

            except Exception as e:
                text = f"An error occured during validation: {e}"
                return True, text
        else:
            # Merge xmls
            if len(out_paths_final) >= 2:
                out_path = os.path.join(
                    XML_PATH,
                    f"merged-music-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.musicxml",
                )

                merge_xmls(out_paths_final, out_path)

                for file in out_paths_final:
                    os.remove(file)

                out_paths_final = [out_path]

            # rename xml to user given name
            music_title = safe_filename(str(self.root.get_screen("progress").ids.title.text))
            if music_title:
                new_path = os.path.join(XML_PATH, f"{music_title}.musicxml")
                if not os.path.exists(new_path):
                    os.rename(out_paths_final[0], new_path)

            # and go back
            Clock.schedule_once(lambda dt: self.change_screen("landing"))

        return False, ""

    def start_download(self, scanner: bool = False, file_chooser: bool = False):
        self.dialog_download.dismiss()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = self.executor.submit(download_weights)
        update = Thread(
            target=self.root.get_screen("downloadpage").update_download_bar,
            args=(
                scanner,
                file_chooser,
            ),
            daemon=True,
        )
        update.start()
        Clock.schedule_once(lambda dt: self.change_screen("downloadpage"), 0.1)

    def check_download_assets(
        self, scanner: bool = False, file_chooser: bool = False, validation: bool = False
    ):
        """
        If not all models are downloaded it will create a Dialog informing the user
        that the App wants to download the models. If the user allows it, the app will switch
        to the downloadpage (which displays a progress bar).
        If all models are downloaded it will directly switch the scanner/file_chooser.
        """
        self.root.get_screen("landing").ids.gen_xml.close_stack()
        self.root.get_screen("edit_file").ids.gen_xml.close_stack()
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
                        on_release=lambda dt: self.start_download(scanner, file_chooser),
                    ),
                ],
            )
            self.dialog_download.open()
            return True
        elif scanner:
            if platform == "android":
                start_document_scan()
            else:
                self.pick_file()
        elif file_chooser:
            self.pick_file()
        elif not validation:
            self.show_toast("You already downloaded all assets")
        return False

    def pick_file(self):
        if platform == "android":
            intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.setType("*/*")
            intent.putExtra(Intent.EXTRA_MIME_TYPES, ["image/*", "application/pdf"])
            intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, True)
            intent.addCategory(Intent.CATEGORY_OPENABLE)
            PythonActivity.mActivity.startActivityForResult(intent, 42)

        else:
            file_paths = filechooser.open_file(title="Select your document", multiple=True)
            Clock.schedule_once(lambda dt: self.change_screen("edit_file"))
            Clock.schedule_once(
                lambda dt: self.root.get_screen("edit_file")._display_imgs(file_paths), 0
            )

    def activity_result(self, request_code, result_code, data):
        if request_code == 42 and result_code == -1 and data:
            self.pick_file_result(data)
        elif request_code == 43 and result_code == -1 and data:
            file_paths = document_scan_result(data)
            Clock.schedule_once(lambda dt: self.change_screen("edit_file"))
            Clock.schedule_once(
                lambda dt: self.root.get_screen("edit_file")._display_imgs(file_paths), 0
            )
        else:
            eprint(
                f"An error occured in activity result: {request_code} {result_code} {bool(data)}"
            )

    def pick_file_result(self, data):
        storage = SharedStorage()
        file_paths = []

        def add_picked_path(path):
            if Path(path).suffix.lower() == ".pdf":
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
                file_paths.extend(pdf_to_img(path, f"converted-pdf-{timestamp}"))
            else:
                file_paths.append(path)

        clip = data.getClipData()
        if clip is not None:
            # Multiple files selected
            for i in range(clip.getItemCount()):
                uri = clip.getItemAt(i).getUri()
                path = storage._copy_uri_to_cache(uri)
                if path:
                    add_picked_path(path)
        else:
            # Single file selected
            uri = data.getData()
            path = storage._copy_uri_to_cache(uri)
            if path:
                add_picked_path(path)

        eprint(f"Picked files: {file_paths}")
        Clock.schedule_once(lambda dt: self.change_screen("edit_file"))
        Clock.schedule_once(
            lambda dt: self.root.get_screen("edit_file")._display_imgs(file_paths), 0
        )


if __name__ == "__main__":
    Andromr().run()
