import copy
import os
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path


def clef_attributes(clef) -> dict[str, str]:
    attributes = {}
    sign = clef.find("sign")
    line = clef.find("line")
    octave = clef.find("clef-octave-change")
    if sign is not None and sign.text:
        attributes["Sign"] = sign.text
    if line is not None and line.text:
        attributes["Line"] = line.text
    if octave is not None and octave.text:
        attributes["ClefOctaveChange"] = octave.text
    return attributes


def time_attributes(time) -> dict[str, str]:
    attributes = {}
    beats = time.find("beats")
    beat_type = time.find("beat-type")
    interchangeable = time.find("interchangeable")
    senza_misura = time.find("senza-misura")
    if beats is not None and beats.text:
        attributes["Beats"] = beats.text
    if beat_type is not None and beat_type.text:
        attributes["BeatType"] = beat_type.text
    if interchangeable is not None and interchangeable.text:
        attributes["Interchangeable"] = interchangeable.text
    if senza_misura is not None and senza_misura.text:
        attributes["SenzaMisura"] = senza_misura.text
    return attributes


def key_attributes(key) -> dict[str, str]:
    attributes = {}
    fifths = key.find("fifths")
    key_alter = key.find("key-alter")
    mode = key.find("mode")
    if fifths is not None and fifths.text:
        attributes["Fifths"] = fifths.text
    if key_alter is not None and key_alter.text:
        attributes["KeyAlter"] = key_alter.text
    if mode is not None and mode.text:
        attributes["Mode"] = mode.text
    return attributes


def get_part_attributes(part) -> dict[str, object]:
    attributes = {}
    for measure in part.findall("measure"):
        for attrib in measure.findall("attributes"):
            div = attrib.find("divisions")
            key = attrib.find("key")
            xtime = attrib.find("time")
            clef = attrib.find("clef")
            if div is not None and div.text:
                attributes["Divisions"] = div.text
            if key is not None:
                attributes["Key"] = key_attributes(key)
            if xtime is not None:
                attributes["Time"] = time_attributes(xtime)
            if clef is not None:
                attributes["Clef"] = clef_attributes(clef)
    return attributes


def preprocess_measure(measure, last_attributes: dict[str, object], debug: bool = False) -> None:
    for attrib in measure.findall("attributes"):
        div = attrib.find("divisions")
        if div is not None and last_attributes.get("Divisions") == div.text:
            attrib.remove(div)
            if debug:
                print(f"Remove division at measure {measure.get('number')}")

        key = attrib.find("key")
        if key is not None and last_attributes.get("Key") == key_attributes(key):
            attrib.remove(key)
            if debug:
                print(f"Remove key at measure {measure.get('number')}")

        xtime = attrib.find("time")
        if xtime is not None and last_attributes.get("Time") == time_attributes(xtime):
            attrib.remove(xtime)
            if debug:
                print(f"Remove time at measure {measure.get('number')}")

        clef = attrib.find("clef")
        if clef is not None and last_attributes.get("Clef") == clef_attributes(clef):
            attrib.remove(clef)
            if debug:
                print(f"Remove clef at measure {measure.get('number')}")

        if len(attrib) == 0:
            measure.remove(attrib)


def merge_xmls(
    concat: tuple[str],
    path_out: str,
    debug=False,
) -> tuple:
    sorted_list = get_file_list(concat, debug=debug)
    if not sorted_list:
        return None, 0, 0

    main_file = sorted_list[0]
    print(f"Starting with {main_file}")
    tree = ET.parse(main_file)
    root = tree.getroot()
    main_parts = root.findall("part")
    last_parts_attributes = [get_part_attributes(part) for part in main_parts]

    for f in sorted_list[1:]:
        if debug:
            print(f"Processing {f}")
        new_root = ET.parse(f).getroot()

        for i, part1 in enumerate(main_parts):
            current_len = len(part1.findall("measure"))
            if debug:
                print(f"Main part has {current_len} measures")

            new_parts = new_root.findall("part")
            if i >= len(new_parts):
                continue

            for measure in new_parts[i].findall("measure"):
                new_measure = copy.deepcopy(measure)
                new_number = str(int(measure.get("number")) + current_len)
                new_measure.set("number", new_number)

                if measure.get("number") == "1":
                    preprocess_measure(new_measure, last_parts_attributes[i], debug=debug)

                part1.append(new_measure)
                if debug:
                    print(f"Added measure {new_number}, part {i + 1}")

            last_parts_attributes[i] = get_part_attributes(part1)

    ET.indent(tree, space="  ")
    tree.write(path_out, encoding="utf-8", xml_declaration=True)
    return None, 0, 0


def get_file_list(
    concat: tuple[str],
    debug=False,
) -> list[str]:
    sorted_list = []

    for pattern in concat:
        if not Path(pattern).suffix:
            pattern += "*.musicxml"

        matched_files = list(glob(pattern))

        if len(matched_files) == 0 and debug:
            print(f"No file found for {pattern}")

        for fichier in matched_files:
            if not os.path.exists(fichier):
                if debug:
                    print(f"The file {fichier} does not exist.")
                return None
            if os.path.isdir(fichier):
                if debug:
                    print(f"{fichier} is a directory.")
                return None

            sorted_list.append(fichier)
    if len(sorted_list) == 0:
        print(f"No files found for {concat}")
        return None
    return sorted(sorted_list)


if __name__ == "__main__":
    merge_xmls(
        ["test_relieur.musicxml", "test_relieur.musicxml"], "test_relieur_merged.musicxml"
    )
