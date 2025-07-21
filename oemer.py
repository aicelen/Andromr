#others
import os
import cv2
import numpy as np

#ml
from typing import Any, Optional, Tuple
from typing import Tuple
from PIL import Image
import backend.layers as layers
from backend.staffline_extraction import extract as staff_extract
from backend.notehead_extraction import extract as note_extract
from backend.note_group_extraction import extract as group_extract
from backend.symbol_extraction import extract as symbol_extract
from backend.rhythm_extraction import extract as rhythm_extract
from backend.build_system import MusicXMLBuilder
from backend.model import TensorFlowModel
from time import perf_counter
from globals import appdata, APP_PATH


#functions for ml (oemer)
def main(img_path, use_gpu, out_path):
    clear_data()
    mxl_path, raw_data = extract(img_path, use_gpu, out_path)
    clear_data()
    return(mxl_path)

def inference(
    model_name: str, 
    img_path: str, 
    use_gpu: bool,
    manual_th: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:


    model = TensorFlowModel()
    model.load(os.path.join(APP_PATH, model_name))

    batch_size = 1

    input_shape =  model.get_input_shape()
    output_shape =  model.get_output_shape()
    input_shape[0] = batch_size
    model.resize_input(input_shape)

    # Collect data
    image_pil = Image.open(img_path)
    if "GIF" != image_pil.format:
        # Tricky workaround to avoid random mistery transpose when loading with 'Image'.
        image_cv = cv2.imread(img_path)
        image_pil = Image.fromarray(image_cv)

    image_pil = image_pil.convert("RGB")
    print(image_pil.size)
    image = np.array(resize_image(image_pil), dtype=np.float32)
    win_size = input_shape[1]

    data = []
    for y in range(0, image.shape[0], appdata.step_size):
        if y + win_size > image.shape[0]:
            y = image.shape[0] - win_size
        for x in range(0, image.shape[1], appdata.step_size):
            if x + win_size > image.shape[1]:
                x = image.shape[1] - win_size
            
            data.append(image[y:y+win_size, x:x+win_size])

    #set progress increment for progress bar in kivy gui
    if model_name == "unet_big.tflite":
        #budget: 35%
        progress_inc = 40/len(data) #66 is the number i calibrated for
    else:
        #budget: 50%
        progress_inc = 52/len(data)

    # Predict
    prediction = []
    for idx in range(0, len(data), batch_size):
        print(f"{idx+1}/{len(data)} (step: {batch_size})")
        batch = np.array(data[idx:idx+batch_size])
        t0 = perf_counter()
        out = model.run_inference(batch)
        print(perf_counter()- t0)
        prediction.append(out)
        appdata.progress += progress_inc
        

    # Merge prediction patches
    output_shape = image.shape[:2] + (output_shape[-1],)
    out = np.zeros(output_shape, dtype=np.float32)
    mask = np.zeros(output_shape, dtype=np.float32)
    hop_idx = 0
    for y in range(0, image.shape[0], appdata.step_size):
        if y + win_size > image.shape[0]:
            y = image.shape[0] - win_size
        for x in range(0, image.shape[1], appdata.step_size):
            if x + win_size > image.shape[1]:
                x = image.shape[1] - win_size
            batch_idx = hop_idx // batch_size #here / ()++
            remainder = hop_idx % batch_size
            hop = prediction[batch_idx][remainder]
            out[y:y+win_size, x:x+win_size] += hop
            mask[y:y+win_size, x:x+win_size] += 1
            hop_idx += 1

    out /= mask
    if manual_th is None:
        class_map = np.argmax(out, axis=-1)
    else:
        assert len(manual_th) == output_shape[-1]-1, f"{manual_th}, {output_shape[-1]}"
        class_map = np.zeros(out.shape[:2] + (len(manual_th),))
        for idx, th in enumerate(manual_th):
            class_map[..., idx] = np.where(out[..., idx+1]>th, 1, 0)

    return class_map, out

def resize_image(image: Image.Image):
    # Estimate target size with number of pixels.
    # Best number would be 3M~4.35M pixels.
    w, h = image.size
    pis = w * h
    if 3000000 <= pis <= 435000:
        return image
    lb = 3000000 / pis
    ub = 4350000 / pis
    ratio = pow((lb + ub) / 2, 0.5)
    tar_w = round(ratio * w)
    tar_h = round(ratio * h)
    print(tar_w, tar_h)
    return image.resize((tar_w, tar_h))

def clear_data() -> None:
    lls = layers.list_layers()
    for l in lls:
        layers.delete_layer(l)


def generate_pred(img_path: str, use_gpu) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("Extracting staffline and symbols")
    staff_symbols_map, _ = inference(
        "unet_big.tflite",
        img_path,
        use_gpu
    )
    staff = np.where(staff_symbols_map==1, 1, 0)
    symbols = np.where(staff_symbols_map==2, 1, 0)

    print("Extracting layers of different symbols")
    #symbol_thresholds = [0.5, 0.4, 0.4]
    sep, _ = inference(
        "seg_net.tflite",
        img_path,
        use_gpu,
        manual_th=None
    )
    stems_rests = np.where(sep==1, 1, 0)
    notehead = np.where(sep==2, 1, 0)
    clefs_keys = np.where(sep==3, 1, 0)
    # stems_rests = sep[..., 0]
    # notehead = sep[..., 1]
    # clefs_keys = sep[..., 2]

    return staff, symbols, stems_rests, notehead, clefs_keys


def polish_symbols(rgb_black_th=300):
    img = layers.get_layer('original_image')
    sym_pred = layers.get_layer('symbols_pred')

    img = Image.fromarray(img).resize((sym_pred.shape[1], sym_pred.shape[0]))
    arr = np.sum(np.array(img), axis=-1)
    arr = np.where(arr < rgb_black_th, 1, 0)  # Filter background
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    arr = cv2.dilate(cv2.erode(arr.astype(np.uint8), ker), ker)  # Filter staff lines
    mix = np.where(sym_pred+arr>1, 1, 0)
    return mix


def register_notehead_bbox(bboxes):
    symbols = layers.get_layer('symbols_pred')
    layer = layers.get_layer('bboxes')
    for (x1, y1, x2, y2) in bboxes:
        yi, xi = np.where(symbols[y1:y2, x1:x2]>0)
        yi += y1
        xi += x1
        layer[yi, xi] = np.array([x1, y1, x2, y2])
    return layer


def register_note_id() -> None:
    symbols = layers.get_layer('symbols_pred')
    layer = layers.get_layer('note_id')
    notes = layers.get_layer('notes')
    for idx, note in enumerate(notes):
        x1, y1, x2, y2 = note.bbox
        yi, xi = np.where(symbols[y1:y2, x1:x2]>0)
        yi += y1
        xi += x1
        layer[yi, xi] = idx
        notes[idx].id = idx


def extract(img_path, use_gpu, out_path) -> str:
    # Make predictions
    staff, symbols, stems_rests, notehead, clefs_keys = generate_pred(str(img_path), use_gpu)
        
    # Load the original image, resize to the same size as prediction.

    image_pil = Image.open(str(img_path))
    if "GIF" != image_pil.format:
        image = cv2.imread(str(img_path))
    else:
        gif_image = image_pil.convert('RGB')
        gif_img_arr = np.array(gif_image)
        image = gif_img_arr[:, :, ::-1].copy()
    
    print(f"reisze{staff.shape[1], staff.shape[0]}")

    image = cv2.resize(image, (staff.shape[1], staff.shape[0]))
    appdata.progress += 2


    
    # Register predictions
    symbols = symbols + clefs_keys + stems_rests
    symbols[symbols>1] = 1
    layers.register_layer("stems_rests_pred", stems_rests)
    layers.register_layer("clefs_keys_pred", clefs_keys)
    layers.register_layer("notehead_pred", notehead)
    layers.register_layer("symbols_pred", symbols)
    layers.register_layer("staff_pred", staff)
    layers.register_layer("original_image", image)
    appdata.progress += 1

    # ---- Extract staff lines and group informations ---- #
    print("Extracting stafflines")
    staffs, zones = staff_extract()
    layers.register_layer("staffs", staffs)  # Array of 'Staff' instances
    layers.register_layer("zones", zones)  # Range of each zones, array of 'range' object.
    appdata.progress += 1

    # ---- Extract noteheads ---- #
    print("Extracting noteheads")
    notes = note_extract()

    # Array of 'NoteHead' instances.
    layers.register_layer('notes', np.array(notes))

    # Add a new layer (w * h), indicating note id of each pixel.
    layers.register_layer('note_id', np.zeros(symbols.shape, dtype=np.int64)-1)
    register_note_id()


    # ---- Extract groups of note ---- #
    print("Grouping noteheads")
    groups, group_map = group_extract()
    layers.register_layer('note_groups', np.array(groups))
    layers.register_layer('group_map', group_map)
    appdata.progress += 1

    # ---- Extract symbols ---- #
    print("Extracting symbols")
    barlines, clefs, sfns, rests = symbol_extract()
    layers.register_layer('barlines', np.array(barlines))
    layers.register_layer('clefs', np.array(clefs))
    layers.register_layer('sfns', np.array(sfns))
    layers.register_layer('rests', np.array(rests))

    appdata.progress += 1


    # ---- Parse rhythm ---- #
    print("Extracting rhythm types")
    rhythm_extract()

    appdata.progress += 1


    # ---- Build MusicXML ---- #
    print("Building MusicXML document")
    basename = os.path.basename(img_path).replace(".jpg", "").replace(".png", "")
    builder = MusicXMLBuilder(title=basename.capitalize())
    raw_data = builder.build()


    #sort(information)
    #mark_unsure
    xml = builder.to_musicxml()

    # ---- Write out the MusicXML ---- #
    if not out_path.endswith(".musicxml"):
        # Take the output path as the folder.
        out_path = os.path.join(out_path, basename+".musicxml")

    with open(out_path, "wb") as ff:
        ff.write(xml)
    appdata.progress += 1
    return out_path, raw_data