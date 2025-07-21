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


# Other imports
from pathlib import Path
from threading import Thread
import os
import numpy as np
import cv2
from time import sleep
from PIL import Image as PILImage
from datetime import datetime

# Imports from own code
from get_theme import get_sys_theme
from oemer import main as oemer
from globals import APP_PATH, appdata
from add_measure_type import add_measure_type
from save_file import save_to_external_storage
from android_camera_api import take_picture
from utils import rotate_image, convert_musicxml_to_midi, crop_image_by_corners


# Ask for permissions on android
if platform == "android":
    from jnius import autoclass # pylint: disable=import-error # type: ignore
    from android.permissions import request_permissions, Permission, check_permission  # pylint: disable=import-error # type: ignore
    required_permissions = [Permission.CAMERA]
    
    def has_all_permissions():
        return all(check_permission(perm) for perm in required_permissions)

    # Request permissions if not already granted
    if not has_all_permissions():
        request_permissions(required_permissions)
        while not has_all_permissions():
            sleep(0.1)
            print('waiting for permissions...')



# Classes of Screens used by kivy
class LandingPage(Screen):
    pass

class CameraPage(Screen):
    pass

class ProgressPage(Screen):
    pass

class SettingsPage(Screen):
    pass

class LicensePage(Screen):
    pass

class TermsPage(Screen):
    pass

class TermsPageButton(Screen):
    pass

class EditImagePage(Screen):
    pass


# Custom widget classes
class KvCam(Camera):
    CameraInfo = autoclass('android.hardware.Camera$CameraInfo')
    resolution = (640, 480) #960, 720
    index = CameraInfo.CAMERA_FACING_BACK

    def on_tex(self, *l):
        if self._camera._buffer is None:
            return None

        super(KvCam, self).on_tex(*l)
        self.texture = Texture.create(size=np.flip(self.resolution), colorfmt='rgb')
        frame = self.frame_from_buf()
        self.frame_to_screen(frame)

    def frame_from_buf(self):
        w, h = self.resolution
        frame = np.frombuffer(self._camera._buffer.tostring(), 'uint8').reshape((h + h // 2, w))
        frame_bgr = cv2.cvtColor(frame, 93)
        if self.index:
            return np.flip(np.rot90(frame_bgr, 1), 1)
        else:
            return np.rot90(frame_bgr, 3)

    def frame_to_screen(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        flipped = np.flip(frame_rgb, 0)
        buf = flipped.tobytes()
        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

class MovableMDIconButton(MDIconButton):
    # custom button that is movable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_x, self.max_y = Window.system_size
        self.app = MDApp.get_running_app()

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            self.pos = (touch.x - self.width / 2, touch.y - self.height / 2)
            
            #update button positions (of unmovable)
            #get buttons
            btn0 = self.app.root.get_screen("image_page").ids.btn0
            btn1 = self.app.root.get_screen("image_page").ids.btn1
            btn2 = self.app.root.get_screen("image_page").ids.btn2
            btn3 = self.app.root.get_screen("image_page").ids.btn3

            #and set their position so they form a rectangle with btn0 and btn3
            btn1.pos = btn0.pos[0], btn3.pos[1]
            btn2.pos = btn3.pos[0], btn0.pos[1]

            #update lines
            line_drawer = self.app.root.get_screen("image_page").ids.line_drawer
            line_drawer.update_lines()

class UnmovableMDIconButton(MDIconButton):
    # custom button that is not movable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_x, self.max_y = Window.system_size

    def update_pos(self, x, y):
        self.pos = x,y

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
            btns = [self.app.root.get_screen('image_page').ids.btn0,
                    self.app.root.get_screen('image_page').ids.btn1,
                    self.app.root.get_screen('image_page').ids.btn3,
                    self.app.root.get_screen('image_page').ids.btn2]

            points = []
            for btn in btns:
                x = btn.center_x
                y = btn.center_y
                points.extend([x, y])

            # Close the shape by connecting last to first
            points.extend([btns[0].center_x, btns[0].center_y])

            Line(points=points, width=1.5)

class Terms_Conditions(RecycleView):
    # shows the terms and conditions using a recycling view widget for better performance
    def __init__(self, **kwargs):
        super(Terms_Conditions, self).__init__(**kwargs)

        def estimate_height(text, font_size=10, width=300, padding=20):
            # Very rough estimate: average characters per line
            avg_char_width = font_size * 0.6
            approx_chars_per_line = width / avg_char_width
            lines = max(1, int(len(text) / approx_chars_per_line) + 1)
            line_height = sp(font_size) * 1.2
            return int(lines * line_height) + padding


        with open(Path(f"{APP_PATH}/data/terms_and_conditions.txt"), 'r') as file:
            lines = [line.strip() for line in file if line.strip()]
        
        
        self.data = [{'text': line, 'size': (None, estimate_height(line))} for line in lines]

class Licenses(RecycleView):
    # shows the open source licenses using a recycling view widget for better performance
    def __init__(self, **kwargs):
        super(Licenses, self).__init__(**kwargs)

        def estimate_height(text, font_size=10, width=300, padding=20):
            # Very rough estimate: average characters per line
            avg_char_width = font_size * 0.6
            approx_chars_per_line = width / avg_char_width
            lines = max(1, int(len(text) / approx_chars_per_line) + 1)
            line_height = sp(font_size) * 1.2
            return int(lines * line_height) + padding



        with open(Path(f"{APP_PATH}/data/licenses.txt"), 'r') as file:
            lines = [line.strip() for line in file if line.strip()]

        
        
        self.data = [{'text': line, 'size': (None, estimate_height(line))} for line in lines]

# App class
class Andromr(MDApp):
    def build(self):
        self.setup()
        #set the app size to a phone size if on windows
        if platform == 'win':
            Window.size = (350, 680)

        # load the file
        self.sm = Builder.load_file("main.kv")

        # if the user hasn't agreed to the terms and conditions
        if not appdata.agreed:
            # show him the screen
            self.sm.current = 'termspagebutton'

        return self.sm


    def setup(self):
        """
        Initializes all the attributes of the class needed
        """
        self.title = "Andromr"
        self.step_size = appdata.step_size

        # generate folders
        os.makedirs(Path(f"{APP_PATH}/data/generated_xmls"), exist_ok=True)
        os.makedirs(Path(f"{APP_PATH}/data/generated_midi"), exist_ok=True)  

        # create files list (used by the scorllview)
        self.files = os.listdir(Path(f"{APP_PATH}/data/generated_xmls")) # list of paths to generated .musicxml

        self.text_lables = [os.path.splitext(file)[0] for file in self.files]

        try:
            self.theme_cls.theme_style = get_sys_theme()
        except:
            self.theme_cls.theme_style = 'Light' # default theme

        self.theme_cls.material_style = "M3" # m3 looks cool

        #set variables from globals to use in the .kv file
        self.step_size = appdata.step_size


    def change_screen(self, screen_name):
        """
        Change the screen displayed by Kivy
        Args:
            screen_name(str): name of the screen you want to change to
        """

        # start the camera
        if screen_name == "camera":
            self.sm.get_screen('camera').ids.camera_pre.play = True
        else:
            self.sm.get_screen('camera').ids.camera_pre.play = False

        #set screen
        self.sm.current = screen_name

        # update the scrollview on the landing page
        if screen_name == "landing":
            self.update_scrollview()




    

    def agree_t_c(self):
        '''Function that is triggered when the user agreed to the terms and conditions'''
        appdata.agreed = True
        appdata.save_settings()
        self.change_screen('landing')



    def update_scrollview(self):
        """Function that updates the scrollview on the landing page"""

        self.files = os.listdir(Path(f"{APP_PATH}/data/generated_xmls"))
        self.text_lables =  [os.path.splitext(file)[0] for file in self.files]
        scroll_box = self.sm.get_screen('landing').ids.scroll_box
        scroll_box.clear_widgets()

        for index, text in enumerate(self.text_lables):
            # Create a horizontal box layout
            row = MDBoxLayout(
                orientation="horizontal",
                size_hint_y=None,
                height=dp(50),
                spacing=10  # Adjust spacing between label and button
            )

            l_name = MDLabel(
                text=text,
                size_hint_x=0.9,  # Make label take most space
                size_hint_y=None,
                height=dp(50),
                halign="center"
            )

            b_delete = MDIconButton(
                icon="delete-outline",
                on_release=lambda func: self.confirm_delete(index),
                size_hint_x=None,
                pos_hint={'center_y': 0.5},
                theme_icon_color="Custom",
                icon_color=(1,0,0,1),
                #ripple_scale=0
                
            )

            b_export = MDIconButton(
                icon="export-variant",
                on_release=lambda func: self.export_option(index),
                size_hint_x=None,
                pos_hint={'center_y': 0.5},
                theme_icon_color="Custom",
                icon_color=(0, 1, 1, 1),
                #ripple_scale=0
            )

            row.add_widget(l_name)
            row.add_widget(b_delete)
            row.add_widget(b_export)
            scroll_box.add_widget(row)  # Add row instead of individual widgets
    

    def crop(self):
        """
        crop the displayed image based on the positions of the buttons
        """
        
        # get button positions
        pos_btns = [self.convert_to_img_pos(self.root.get_screen('image_page').ids.btn0.center), 
                    self.convert_to_img_pos(self.root.get_screen('image_page').ids.btn1.center), 
                    self.convert_to_img_pos(self.root.get_screen('image_page').ids.btn2.center), 
                    self.convert_to_img_pos(self.root.get_screen('image_page').ids.btn3.center)
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
        x = max(0, min(x*self.scale, self.size[0]))
        y = max(0, min(y*self.scale, self.size[1]))

        return [x, abs(self.size[1]-y)] #the abs thing is a workaround because i read with (0,0) at bottom left while numpy/pillow thinks it's at(0, img_height) 

        
    def set_cutter_btn(self, instance=None):
        """
        Displays the drag+drop buttons on image-corners
        """

        #get original size of the image
        img_x, img_y = self.size

        #get screen size
        win_x, win_y = Window.size
        
        #calculate on which size the image and the screen touch
        if img_x/win_x > img_y/win_y:
            #sides touch vertically
            self.vertical_touch = True
            self.scale = img_x/win_x

            dist_y = win_x*img_y/img_x
            dist_x = win_x

            #calculate how much space is left
            self.space_left = (win_y - dist_y)/2 #half beause we only need one distance

            
            offset = 50 # buttons should be easily touchable. that's why there's an offset to the edge of the screen

            #move buttons to the correct point
            self.root.get_screen('image_page').ids.btn0.center = (offset, self.space_left+offset)
            self.root.get_screen('image_page').ids.btn3.center = (dist_x-offset, self.space_left+dist_y-offset) 
            self.root.get_screen('image_page').ids.btn2.center = (offset, self.space_left+dist_y-offset) 
            self.root.get_screen('image_page').ids.btn1.center = (dist_x-offset, self.space_left+offset)

        else:
            #sides touch horizontally - currently not working
            self.vertical_touch = False
            self.scale = img_y/win_y
            dist_x = win_y*img_x/img_y
            dist_y = win_y


            #calculate how much space is left
            self.space_left = (win_x - dist_x)/2 # half because we need only one distance

            offset = 50
            self.root.get_screen('image_page').ids.btn0.center = (self.space_left-25, 0)
            self.root.get_screen('image_page').ids.btn3.center = (self.space_left+dist_x-25, dist_y-50)
            self.root.get_screen('image_page').ids.btn2.center = (self.space_left-25, dist_y-50)
            self.root.get_screen('image_page').ids.btn1.center = (self.space_left+dist_x-25, 0)

        #and display the lines
        line_drawer = self.root.get_screen("image_page").ids.line_drawer
        line_drawer.update_lines()


    def display_img(self):
        """
        displays an image in the image_box
        """
        # display screen to image_page
        self.change_screen('image_page')

        # remove the image loaded previously to the image box
        self.root.get_screen('image_page').ids.image_box.clear_widgets()

        # get size of loaded image
        img = PILImage.open(self.img_path)
        self.size = img.size

        #display image in image_box
        img_widget = Image(source=self.img_path)
        img_widget.reload()
        self.root.get_screen('image_page').ids.image_box.add_widget(img_widget)

        # move the buttons to the correct location
        self.set_cutter_btn()


    def export_file(self, musicxml, idx):
        """
        Save a file to Android External Storage
        Args:
            musicxml(bool): export as musicxml
            idx(int): index of the element in self.files
        
            Returns:
                None
        """
        if musicxml:
            #export (.musicxml)
            self.show_toast(save_to_external_storage(f"{APP_PATH}/data/generated_xmls/{self.files[idx]}"))
        
        else:
            #convert to .mid
            if not os.path.exists(f"{APP_PATH}/data/generated_midi/{self.files[idx]}.midi"):
                convert_musicxml_to_midi(f"{APP_PATH}/data/generated_xmls/{self.files[idx]}.musicxml", f"{APP_PATH}/data/generated_midi/{self.files[idx]}.mid")
            #export (.mid)
            self.show_toast(save_to_external_storage(f"{APP_PATH}/data/generated_midi/{self.files[idx]}.mid"))
        
        self.dialog_export.dismiss()


    def show_toast(self, text: str):
        """
        show a simple text message on the bottom
        Args:
            text(str): Text wanted to be displayed
        """
        toast(text, True, 80, 200, 0)
    

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
                MDFlatButton(
                    text="OK", 
                    on_release=lambda dt: self.dialog_information.dismiss()
                )
            ]
        )
        
        self.dialog_information.text = text
        self.dialog_information.title = title
        self.dialog_information.open()


    def confirm_delete(self, idx: int):
        """
        creates a special pop-up with two buttons; one is bound to delete an element in the scrollview, the other to cancel
        Args:
            idx(int): index of the element that we want to be deleted
        """
        self.dialog_delete = MDDialog(
            text="Are you sure you want to delete this scan? This cannot be undone.",
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    on_release=lambda dt: self.dialog_delete.dismiss()
                ),
                MDFlatButton(
                    text="CONFIRM",
                    on_release=lambda func: self.delete_element(idx)
                ),
            ]
        )
        self.dialog_delete.open()
    
    def export_option(self, idx: int):
        """
        give option to export as .musicxml or .mid
        Args:
            idx(int): index of the element that we want to be deleted
        """
        self.dialog_export = MDDialog(
            title="Export as",
            text="",
            buttons=[
                MDFlatButton(
                    text="musicxml",
                    on_release=lambda func: self.export_file(True, idx) #prior: idx=idx
                ),
                MDFlatButton(
                    text="midi",
                    on_release=lambda func: self.export_file(False, idx)
                ),
            ]
        )
        self.dialog_export.open()
    
    def start_inference(self, path_to_image: str):
        # set the progress bar to 0
        appdata.progress = 0

        # go to progress page
        self.change_screen("progress")

        # reset text-field contents
        self.root.get_screen('progress').ids.title.text = ""
        self.root.get_screen('progress').ids.division.text = ""
        self.root.get_screen('progress').ids.beat.text = ""

        # start the ml thread and the progress thread seperatly from each other
        self.ml_thread = Thread(target=self.oemer_backend_call, args=(path_to_image, f"{APP_PATH}/data/generated_xmls"), daemon=True)
        self.progress_thread = Thread(target=self.update_progress_bar, daemon=True)
        self.ml_thread.start()
        self.progress_thread.start()


    def oemer_backend_call(self, path: str, output_path: str):
        """
        calls the oemer (optical music recognition software) and returns when finished to the landing page
        Args:
            path(str): path to the image
            output_path(str): path where the musicxml is stored

        """
        try:
            # run oemer
            return_path = oemer(path, appdata.use_gpu, output_path)


            if self.root.get_screen('progress').ids.title.text == "":
                # if there's no user given title we give it a unique id based on time
                music_title = f"transcribed-music-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
            else:
                # otherwise we set it to the users title
                music_title = str(self.root.get_screen('progress').ids.title.text)

            # get user inputs of the beat and the division
            beat = self.root.get_screen('progress').ids.beat.text
            division = self.root.get_screen('progress').ids.division.text

            # if we got valid integers
            if beat.isdigit() and division.isdigit():
                # we can add them
                add_measure_type(return_path, beat, division)

            # rename the file
            os.rename(return_path, os.path.join(APP_PATH, "data", "generated_xmls", f"{music_title}.musicxml"))

            # and update self.files (needed for the scorllview)
            self.files = os.listdir(Path(f"{APP_PATH}/data/generated_xmls"))
            self.text_lables =  [os.path.splitext(file)[0] for file in self.files]
        
        except Exception as e:
            # if anything fails during transcription it prints the error
            print(e)
            # and switches back to the landing screen as if nothing ever happend

        # set the progress to 100 to stop the while loop in self.update_progress_bar()
        appdata.progress = 100

        # switch to landing screen
        Clock.schedule_once(lambda dt:self.change_screen("landing"))

    def update_progress_bar(self):
        """
        Update the progress bar used while running oemer
        """
        while appdata.progress != 100:
            self.root.get_screen('progress').ids.progress_bar.value = appdata.progress
            self.root.get_screen('progress').ids.progress_label.text = f"{round(appdata.progress, 1)}%"
            sleep(0.1) # we dont need this to update every frame



    #takes all the values in the settings tab and hands them over to global variables
    def apply_settings(self, use_gpu: bool, step_size: int):
        """
        Applies the given values to the variables used and saves them to a .pkl file
        Args:
            use_gpu(bool): use gpu or not
            step_size(int): describes the distance the immage section travels after each input to the tflite models
        """

        # user sees quality setting ranging from 1 to 5; needs to be converted to step_size
        quality_to_step_size = {
            1: 256, # fastest
            2: 224,
            3: 192,
            4: 160,
            5: 128 # slowest
        }
        appdata.step_size = quality_to_step_size[step_size]
        appdata.use_gpu = use_gpu
        appdata.save_settings()
        self.change_screen("landing")
    
    def step_size_to_quality(self, step_size):
        """
        converts the step_size to quality to display it to the user
        """
        quality_from_step_size = {
            256: 1,
            224: 2,
            192: 3,
            160: 4,
            128: 5
        }
        return quality_from_step_size[step_size]

    def delete_element(self, index: int):
        """
        Deletes a certain element of the scrollview
        Args:
            index(int): index of the element in the scrollview that should be deleted
        """
        os.remove(Path(f"{APP_PATH}/data/generated_xmls/{self.files[index]}"))
        self.dialog_delete.dismiss()
        self.update_scrollview()

    def img_taken(self, img_path):
        """
        Function running after the image is taken, see android_camera_api.py
        Args:
            filename(str/Path): path to the image
        """
        rotate_image(img_path, img_path) # we need to rotate the image; android stores them 270° rotated (clockwise)
        self.img_path = img_path
        Clock.schedule_once(lambda dt: self.display_img())

    
    def capture(self, filename='taken_img.png'): # if you wonder why it saves to png; oemer "hates" jpg compression artifacts, everything below 90 messed things up  
        '''Take an image'''
        try:
            os.remove(filename)
        except:
            print("couldn't be removed")
        take_picture(self.root.get_screen('camera').ids.camera_pre, self.img_taken, filename)


    def on_start(self):
        # Runs from the start of Kivy
        self.update_scrollview()
    
    def on_stop(self):
        # When leaving we should save everything we ajusted in the settigns
        appdata.save_settings()

if __name__ == "__main__":
    Andromr().run()