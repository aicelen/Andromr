from PIL import Image
import os
from music21 import converter, midi
import numpy as np
import cv2

def rotate_image(input_path, output_path):
    """
    Rotates an image 90 degrees clockwise and saves it.
    Swaps dimensions (e.g., 2000x1000 becomes 1000x2000).
    
    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path where the rotated image will be saved.
    """
    try:
        # load
        img = cv2.imread(input_path)
        
        # rotate
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
        # save
        cv2.imwrite(output_path, rotated_img)

    except Exception as e:
        print(f"Error: {e}")


def convert_musicxml_to_midi(input_file, output_file=None):
    """
    Convert a MusicXML file to MIDI format
    
    Args:
        input_file (str): Path to the input MusicXML file
        output_file (str, optional): Path for the output MIDI file. 
                                   If None, uses input filename with .mid extension
    
    Returns:
        None
    """
    # Load the MusicXML file
    score = converter.parse(input_file)
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = base_name + '.mid'
    
    # Convert to MIDI and save
    midi_file = midi.translate.music21ObjectToMidiFile(score)
    midi_file.open(output_file, 'wb')
    midi_file.write()
    midi_file.close()
    

def crop_image_by_corners(path, points, output_path):
    """
    Crops an image based on up to 4 points
    Args:
        path(str): Path to input image
        points(List[int or float]): Points to cut from
        output_path(str): Path to save the image
    
    Returns:
        None
    """
    img = Image.open(path)
    arr = np.array(img)

    # Find bounding box of the 4 points
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    
    min_x = int(round(min(x_coords)))
    min_y = int(round(min(y_coords)))
    max_x = int(round(max(x_coords)))
    max_y = int(round(max(y_coords)))

    # Crop the image
    arr = arr[min_y:max_y, min_x:max_x]
    img = Image.fromarray(arr)

    # Save the cropped image
    img.save(output_path)