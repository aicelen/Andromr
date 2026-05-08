import os
import json
from pathlib import Path
from kivy import platform
from homr.simple_logging import eprint
import shutil

APP_PATH = os.getcwd()

if platform == "android":
    from android.storage import app_storage_path  # type: ignore
    from android import mActivity # type: ignore

    def get_android_storage() -> str:
        """
        android.app_storage_path is deprecated and unsafe. Therefore 
        we need to migrate the files to the newer getExternalFilesDir.
        Returns path to the used App_Storage
        """
        old_path = app_storage_path()

        context = mActivity.getApplicationContext()
        result = context.getExternalFilesDir(None)
        if result is None:
            eprint("Using old path")
            return old_path # external storage not available, can't migrate
        new_path = str(result.toString())

        dirs_to_migrate = ["models", "images", "musicxml"]
        if os.path.exists(os.path.join(new_path, "saved_settings.json")) or not os.path.exists(os.path.join(old_path, "saved_settings.json")):
            eprint("Already migrated. Using new path")
            return new_path
        try:
            for directory in dirs_to_migrate:
                cur_old_dir = os.path.join(old_path, directory)
                cur_new_dir = os.path.join(new_path, directory)
                shutil.copytree(cur_old_dir, cur_new_dir, dirs_exist_ok=True)
                shutil.rmtree(cur_old_dir)

            shutil.move(os.path.join(old_path, "saved_settings.json"), new_path)
            eprint("using new path")
            return new_path

        except Exception as e:
            eprint(f'Migration failed: {e}')
            # Clean up any partial copy
            if os.path.exists(new_path):
                shutil.rmtree(new_path)
        eprint("Using old path")
        return old_path

    APP_STORAGE = os.path.join(get_android_storage())
else:
    APP_STORAGE = os.path.join(APP_PATH, "data")

MODEL_STORAGE = os.path.join(APP_STORAGE, "models")
IMAGE_PATH = os.path.join(APP_STORAGE, "images")
XML_PATH = os.path.join(APP_STORAGE, "musicxml")

os.makedirs(XML_PATH, exist_ok=True)
os.makedirs(IMAGE_PATH, exist_ok=True)
os.makedirs(MODEL_STORAGE, exist_ok=True)


class AppData:
    def __init__(self):
        self.settings_file_path = os.path.join(APP_STORAGE, "saved_settings.json")

        # Default settings
        self.default_settings = {"threads": 2, "xnnpack": False, "agreed": False, "gpu": False}

        # Hashes
        self.hashes = {
            "segnet_308_int8.zip": "2c7ba2ad87a20f11b5122ce76cb244167ba67c2d0be962ac55f746f6ef03f377",
            "encoder_331_int8.zip": "75ddeefb4402cb95f0454e2cc1e31305463020598cd16fdf48b2461c7388a796",
            "decoder_331_int8.zip": "f327d851ed674935f2c83804ce4806ae51793a422138e2b8d59bf96e97e49db1"
        }

        # Load data
        self._load_settings()

        # Initialize other attributes
        self.homr_running = True
        self.homr_progress = 0
        self.homr_state = "Segmenting"

        self.download_running = True
        self.download_progress = 0
        self.downloaded_assets = "0/4"
        self.settings_changed = False

    def _load_settings(self):
        if os.path.exists(self.settings_file_path):
            # Load file normally
            try:
                with open(self.settings_file_path, "r") as f:
                    settings = json.load(f)

                # Validate and use loaded settings
                self.threads = settings.get("threads", self.default_settings["threads"])
                self.xnnpack = settings.get("xnnpack", self.default_settings["xnnpack"])
                self.agreed = settings.get("agreed", self.default_settings["agreed"])
                self.gpu = settings.get("gpu", self.default_settings["gpu"])
                return

            except Exception as e:
                eprint(f"An error occured during json loading. The file is probably corrupted. Creating a new file")
                pass
        
        # Create new file using defaults
        Path(self.settings_file_path).parent.mkdir(parents=True, exist_ok=True)
        self.threads = self.default_settings["threads"]
        self.xnnpack = self.default_settings["xnnpack"]
        self.agreed = self.default_settings["agreed"]
        self.gpu = self.default_settings["gpu"]
        self.save_settings()

    def save_settings(self):
        settings_data = {
            "threads": self.threads,
            "xnnpack": self.xnnpack,
            "agreed": self.agreed,
            "gpu": self.gpu,
        }

        with open(self.settings_file_path, "w") as f:
            json.dump(settings_data, f, indent=2)


appdata = AppData()
