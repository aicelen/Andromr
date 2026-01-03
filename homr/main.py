import glob
import os
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np
from kivy.utils import platform

from homr import color_adjust, download_utils
from homr.autocrop import autocrop
from homr.bar_line_detection import (
    detect_bar_lines,
    prepare_bar_line_image,
)
from homr.bounding_boxes import (
    BoundingEllipse,
    RotatedBoundingBox,
    create_bounding_ellipses,
    create_rotated_bounding_boxes,
)
from homr.brace_dot_detection import (
    find_braces_brackets_and_grand_staff_lines,
    prepare_brace_dot_image,
)
from homr.debug import Debug
from homr.model import InputPredictions, MultiStaff
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.noise_filtering import filter_predictions
from homr.note_detection import add_notes_to_staffs, combine_noteheads_with_stems
from homr.resize import resize_image
from homr.segmentation.config import segnet_path_tflite
from homr.segmentation.inference_segnet import extract
from homr.simple_logging import eprint
from homr.staff_detection import break_wide_fragments, detect_staff, make_lines_stronger
from homr.staff_parsing import parse_staffs
from homr.staff_position_save_load import load_staff_positions, save_staff_positions
from homr.transformer.configs import default_config
from homr.type_definitions import NDArray

from globals import appdata

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class PredictedSymbols:
    def __init__(
        self,
        noteheads: list[BoundingEllipse],
        staff_fragments: list[RotatedBoundingBox],
        clefs_keys: list[RotatedBoundingBox],
        stems_rest: list[RotatedBoundingBox],
        bar_lines: list[RotatedBoundingBox],
    ) -> None:
        self.noteheads = noteheads
        self.staff_fragments = staff_fragments
        self.clefs_keys = clefs_keys
        self.stems_rest = stems_rest
        self.bar_lines = bar_lines


def get_predictions(
    original: NDArray, preprocessed: NDArray, img_path: str, enable_cache: bool
) -> InputPredictions:
    result = extract(preprocessed, img_path, step_size=320, use_cache=enable_cache)
    original_image = cv2.resize(original, (result.staff.shape[1], result.staff.shape[0]))
    preprocessed_image = cv2.resize(preprocessed, (result.staff.shape[1], result.staff.shape[0]))
    return InputPredictions(
        original=original_image,
        preprocessed=preprocessed_image,
        notehead=result.notehead.astype(np.uint8),
        symbols=result.symbols.astype(np.uint8),
        staff=result.staff.astype(np.uint8),
        clefs_keys=result.clefs_keys.astype(np.uint8),
        stems_rest=result.stems_rests.astype(np.uint8),
    )


def replace_extension(path: str, new_extension: str) -> str:
    return os.path.splitext(path)[0] + new_extension


def load_and_preprocess_predictions(
    image_path: str, enable_debug: bool, enable_cache: bool
) -> tuple[InputPredictions, Debug]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to read " + image_path)
    image = autocrop(image)
    image = resize_image(image)
    preprocessed, _background = color_adjust.color_adjust(image, 40)
    predictions = get_predictions(image, preprocessed, image_path, enable_cache)
    debug = Debug(predictions.original, image_path, enable_debug)
    debug.write_image("color_adjust", preprocessed)

    predictions = filter_predictions(predictions, debug)

    predictions.staff = make_lines_stronger(predictions.staff, (1, 2))
    debug.write_threshold_image("staff", predictions.staff)
    debug.write_threshold_image("symbols", predictions.symbols)
    debug.write_threshold_image("stems_rest", predictions.stems_rest)
    debug.write_threshold_image("notehead", predictions.notehead)
    debug.write_threshold_image("clefs_keys", predictions.clefs_keys)
    return predictions, debug


def predict_symbols(debug: Debug, predictions: InputPredictions) -> PredictedSymbols:
    eprint("Creating bounds for noteheads")
    noteheads = create_bounding_ellipses(predictions.notehead, min_size=(4, 4))
    eprint("Creating bounds for staff_fragments")
    staff_fragments = create_rotated_bounding_boxes(
        predictions.staff, skip_merging=True, min_size=(5, 1), max_size=(10000, 100)
    )

    eprint("Creating bounds for clefs_keys")
    clefs_keys = create_rotated_bounding_boxes(
        predictions.clefs_keys, min_size=(20, 40), max_size=(1000, 1000)
    )
    eprint("Creating bounds for stems_rest")
    stems_rest = create_rotated_bounding_boxes(predictions.stems_rest)
    eprint("Creating bounds for bar_lines")
    bar_line_img = prepare_bar_line_image(predictions.stems_rest)
    debug.write_threshold_image("bar_line_img", bar_line_img)
    bar_lines = create_rotated_bounding_boxes(bar_line_img, skip_merging=True, min_size=(1, 5))

    return PredictedSymbols(noteheads, staff_fragments, clefs_keys, stems_rest, bar_lines)


@dataclass
class ProcessingConfig:
    enable_debug: bool
    enable_cache: bool
    write_staff_positions: bool
    read_staff_positions: bool
    selected_staff: int


def process_image(
    image_path: str,
    config: ProcessingConfig,
    xml_generator_args: XmlGeneratorArguments,
) -> None:
    eprint("Processing " + image_path)
    xml_file = replace_extension(image_path, ".musicxml")
    debug_cleanup: Debug | None = None
    try:
        if config.read_staff_positions:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to read " + image_path)
            image = resize_image(image)
            debug = Debug(image, image_path, config.enable_debug)
            staff_position_files = replace_extension(image_path, ".txt")
            multi_staffs = load_staff_positions(
                debug, image, staff_position_files, config.selected_staff
            )
        else:
            multi_staffs, image, debug = detect_staffs_in_image(image_path, config)
        debug_cleanup = debug

        appdata.homr_state = "Transforming"
        appdata.homr_progress = 1

        result_staffs = parse_staffs(
            debug, multi_staffs, image, selected_staff=config.selected_staff
        )

        eprint("Writing XML", result_staffs)
        xml = generate_xml(xml_generator_args, result_staffs, "")  # "" for the empty title
        xml.write(xml_file)

        eprint("Finished parsing " + str(len(result_staffs)) + " staves")
        teaser_file = replace_extension(image_path, "_teaser.png")
        if config.write_staff_positions:
            staff_position_files = replace_extension(image_path, ".txt")
            save_staff_positions(multi_staffs, image.shape, staff_position_files)
        debug.write_teaser(teaser_file, multi_staffs)
        debug.clean_debug_files_from_previous_runs()

        eprint("Result was written to", xml_file)
    except:
        if os.path.exists(xml_file):
            os.remove(xml_file)
        raise
    finally:
        if debug_cleanup is not None:
            debug_cleanup.clean_debug_files_from_previous_runs()
    return xml_file


def detect_staffs_in_image(
    image_path: str, config: ProcessingConfig
) -> tuple[list[MultiStaff], NDArray, Debug]:
    appdata.homr_state = "Segementing"
    appdata.homr_progress = 1

    predictions, debug = load_and_preprocess_predictions(
        image_path, config.enable_debug, config.enable_cache
    )

    appdata.homr_state = "Extracting"
    appdata.homr_progress = 1

    symbols = predict_symbols(debug, predictions)

    symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)
    debug.write_bounding_boxes("staff_fragments", symbols.staff_fragments)
    eprint("Found " + str(len(symbols.staff_fragments)) + " staff line fragments")

    noteheads_with_stems = combine_noteheads_with_stems(symbols.noteheads, symbols.stems_rest)
    debug.write_bounding_boxes_alternating_colors("notehead_with_stems", noteheads_with_stems)
    eprint("Found " + str(len(noteheads_with_stems)) + " noteheads")
    if len(noteheads_with_stems) == 0:
        raise Exception("No noteheads found")

    appdata.homr_progress = 7

    average_note_head_height = float(
        np.median([notehead.notehead.size[1] for notehead in noteheads_with_stems])
    )
    eprint("Average note head height: " + str(average_note_head_height))

    all_noteheads = [notehead.notehead for notehead in noteheads_with_stems]
    all_stems = [note.stem for note in noteheads_with_stems if note.stem is not None]
    bar_lines_or_rests = [
        line
        for line in symbols.bar_lines
        if not line.is_overlapping_with_any(all_noteheads)
        and not line.is_overlapping_with_any(all_stems)
    ]
    bar_line_boxes = detect_bar_lines(bar_lines_or_rests, average_note_head_height)
    debug.write_bounding_boxes_alternating_colors("bar_lines", bar_line_boxes)
    eprint("Found " + str(len(bar_line_boxes)) + " bar lines")

    debug.write_bounding_boxes(
        "anchor_input", symbols.staff_fragments + bar_line_boxes + symbols.clefs_keys
    )

    appdata.homr_progress = 10

    staffs = detect_staff(
        debug, predictions.staff, symbols.staff_fragments, symbols.clefs_keys, bar_line_boxes
    )

    appdata.homr_progress = 90

    if len(staffs) == 0:
        raise Exception("No staffs found")
    debug.write_bounding_boxes_alternating_colors("staffs", staffs)

    brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
    debug.write_threshold_image("brace_dot", brace_dot_img)
    brace_dot = create_rotated_bounding_boxes(brace_dot_img, skip_merging=True, max_size=(100, -1))

    notes = add_notes_to_staffs(
        staffs, noteheads_with_stems, predictions.symbols, predictions.notehead
    )

    multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, brace_dot)
    eprint(
        "Found",
        len(multi_staffs),
        "connected staffs (after merging grand staffs, multiple voices): ",
        [len(staff.staffs) for staff in multi_staffs],
    )

    debug.write_all_bounding_boxes_alternating_colors("notes", multi_staffs, notes)

    return multi_staffs, predictions.preprocessed, debug


def get_all_image_files_in_folder(folder: str) -> list[str]:
    image_files = []
    for ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
        image_files.extend(glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True))
    without_teasers = [
        img
        for img in image_files
        if "_teaser" not in img
        and "_debug" not in img
        and "_staff" not in img
        and "_tesseract" not in img
    ]
    return sorted(without_teasers)


def download_weights() -> None:
    try:
        base_url = "https://github.com/aicelen/Andromr/releases/download/v1.0/"
        missing_models = check_for_missing_models()
        if len(missing_models) == 0:
            return

        appdata.downloaded_assets = f"Downloaded 0 of {len(missing_models)}"

        eprint("Downloading", len(missing_models), "models - this is only required once")
        for idx, model in enumerate(missing_models):
            base_name = os.path.basename(model).split(".")[0]
            eprint(f"Downloading {base_name}")
            try:
                zip_name = base_name + ".zip"
                download_url = base_url + zip_name
                downloaded_zip = os.path.join(os.path.dirname(model), zip_name)
                download_utils.download_file(download_url, downloaded_zip)
                if platform == 'android':
                    destination_dir = os.path.dirname(model)
                download_utils.unzip_file(downloaded_zip, destination_dir)
            finally:
                if os.path.exists(downloaded_zip):
                    os.remove(downloaded_zip)

            appdata.downloaded_assets = f"Downloaded {idx + 1} of {len(missing_models)}"

    except Exception as e:
        print(e)
        appdata.downloaded_assets = "failure"

    appdata.download_running = False
    return

def check_for_missing_models() -> list:
    """
    Checks for missing models and returns a list with all the links to the missing models.
    """
    models = [
        segnet_path_tflite,
        default_config.filepaths.encoder_cnn_path_tflite,
        default_config.filepaths.encoder_transformer_path,
        default_config.filepaths.decoder_path,
    ]
    missing_models = [model for model in models if not os.path.exists(model)]
    return missing_models


def homr(path, cache=False):
    t0 = perf_counter()
    config = ProcessingConfig(False, False, False, False, -1)
    xml_generator_args = XmlGeneratorArguments(False, False, False)
    out_path = process_image(path, config, xml_generator_args)
    eprint(f"Homr took {perf_counter() - t0} seconds.")
    return out_path


if __name__ == "__main__":
    homr("taken_img.jpg", cache=False)
