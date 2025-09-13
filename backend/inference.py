import numpy as np
from numpy import ndarray
from globals import APP_PATH
from backend.model import TensorFlowModel
import json
from PIL import Image
import os

class PredictSymbols():
    """
    Class to predict symbols using tflite models.

        **Attributes**:
            model_name(str): which model to use; choose between "sfn", "clef", "rests", "rests_above8"

        **Methods**:
            predict(region: ndarray) -> str:
                Predicts a symbol from the given 2D np array 
    """
    
    def __init__(self, model_name):
        self.model = TensorFlowModel()
        self.model.load(os.path.join(APP_PATH, f"{model_name}.tflite"))
        
        with open(os.path.join(APP_PATH, "backend", "tflite_models", f"{model_name}.json")) as f:
            self.class_map = json.load(f)


    def predict(self, region: ndarray):
        """
        Predict a symbol using tensorflowlite.
        Args:
            region(ndarray): 2d (black-white) array containing the symbol
        
        Returns:
            symbol(string): name of the symbol; depending on the model
        """
        image_of_region = Image.fromarray(region.astype(np.uint8))
        region = np.array(image_of_region.resize((40, 70), Image.NEAREST))
        region = region*255
        region = np.expand_dims(region, axis=0)  # [height, width, channels] -> [1, height, width, channels]
        region = np.expand_dims(region, axis=-1)  # [1, height, width] -> [1, height, width, 1]

        pred = self.model.run_inference(region.astype(np.float32))
        return self.class_map[str(np.argmax(pred))]