from os import path
from pickle import load, dump
from pathlib import Path

APP_PATH = path.dirname(path.realpath(__file__)) # get path of my app


class AppData():
    def __init__(self):
        # load data
        if path.exists(Path(APP_PATH+'/data/saved_settings.pkl')):
            with open(Path(APP_PATH+'/data/saved_settings.pkl'), 'rb') as f:
                saved_settings = load(f)

        else:
            #if not existing: create file
            print('error')
            with open(Path(APP_PATH+'/data/saved_settings.pkl'), 'wb') as f:
                dump([
                    2, # Number of threads
                    False, # Read to license
                    False, # Use xnnpack
                    ], f)

            #than we load it
            with open(Path(APP_PATH+'/data/saved_settings.pkl'), 'rb') as f:
                saved_settings = load(f)

        self.threads = saved_settings[0]
        self.xnnpack= saved_settings[1]
        self.agreed = saved_settings[2]

        self.progress = 0
        self.download_running = True
        self.download_progress = 0
        self.downloaded_assets = "0/4"

    def save_settings(self):
        with open(Path(APP_PATH+'/data/saved_settings.pkl'), 'wb') as f:
            dump([self.threads, self.xnnpack, self.agreed], f)


appdata = AppData()