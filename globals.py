from os import path
import json
from pathlib import Path

APP_PATH = path.dirname(path.realpath(__file__))  # get path of my app


class AppData:
    def __init__(self):
        self.settings_file = Path(APP_PATH + "/data/saved_settings.json")
        # Default settings
        self.default_settings = {"threads": 2, "xnnpack": False, "agreed": False, "gpu": False}

        # Load data
        self._load_settings()

        # Initialize other attributes
        self.homr_running = True
        self.homr_progress = 0
        self.homr_state = "Segmentation"

        self.download_running = True
        self.download_progress = 0
        self.downloaded_assets = "0/4"
        self.settings_changed = False

    def _load_settings(self):
        if self.settings_file.exists():
            # Load file normally
            with open(self.settings_file, "r") as f:
                settings = json.load(f)

            # Validate and use loaded settings
            self.threads = settings.get("threads", self.default_settings["threads"])
            self.xnnpack = settings.get("xnnpack", self.default_settings["xnnpack"])
            self.agreed = settings.get("agreed", self.default_settings["agreed"])
            self.gpu = settings.get("gpu", self.default_settings["gpu"])

        else:
            # Create new file using defaults
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
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

        with open(self.settings_file, "w") as f:
            json.dump(settings_data, f, indent=2)


appdata = AppData()
