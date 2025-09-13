from enum import Enum, auto

from backend.dense_dataset_definitions import DENSE_DATASET_DEFINITIONS as DEF


CLASS_CHANNEL_LIST = [
    DEF.STAFF + DEF.LEDGERLINE,
    DEF.NOTEHEADS_SOLID + [38],
    DEF.NOTEHEADS_HOLLOW + [42],
    DEF.NOTEHEADS_WHOLE + [46],
    DEF.FLAG_DOWN + DEF.FLAG_UP + [59, 65],
    DEF.BEAM + DEF.DOT,
    DEF.BARLINE_BETWEEN + DEF.STEM,
    DEF.ALL_ACCIDENTALS,
    DEF.ALL_KEYS,
    DEF.ALL_RESTS + [163],
    DEF.TUPETS,
    DEF.SLUR_AND_TIE,
    DEF.ALL_CLEFS + DEF.NUMBERS,
    DEF.TIME_SIGNATURE_SUBSET
]

CLASS_CHANNEL_MAP = {
    color: idx+1
    for idx, colors in enumerate(CLASS_CHANNEL_LIST)
    for color in colors
}

CHANNEL_NUM = len(CLASS_CHANNEL_LIST) + 2


class NoteHeadConstant:
    NOTEHEAD_MORPH_WIDTH_FACTOR = 0.5 #0.444444  # Width to unit size factor when morphing
    NOTEHEAD_MORPH_HEIGHT_FACTOR = 0.4 #0.37037  # Height to unit size factor when morphing
    NOTEHEAD_SIZE_RATIO = 1.285714  # width/height

    STEM_WIDTH_UNIT_RATIO = 0.272727  # Ratio of stem's width to unit size
    STEM_HEIGHT_UNIT_RATIO = 4  # Ratio of stem's height to unit size

    CLEF_ZONE_WIDTH_UNIT_RATIO = 4.5406916
    CLEF_WIDTH_UNIT_RATIO = 3.2173913
    SMALL_CLEF_WIDTH_UNIT_RATIO = 2.4347826

    STAFFLINE_WIDTH_UNIT_RATIO = 0.261
