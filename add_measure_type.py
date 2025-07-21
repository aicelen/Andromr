import xml.etree.ElementTree as ET


def add_measure_type(xml_path, beats, beat_type):
    """
    Adds the measure type to the musicxml (e.g. 4/4)
    Args:
        xml_path(str): Path to xml_path
        beats(int): 1st part of measure type
        beat_type(int): 2nd part of measure type
    """

    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find MusicXML namespace (if present)
    ns = ""
    if "}" in root.tag:
        ns = root.tag.split("}")[0] + "}"

    # Find first <measure> element
    first_measure = root.find(f".//{ns}measure")
    if first_measure is None:
        print("Error: No <measure> element found in the file.")
        return False

    # Find or create attributes in the first measure
    attributes = first_measure.find(f"./{ns}attributes")
    if attributes is None:
        attributes = ET.SubElement(first_measure, f"{ns}attributes")

    # Remove existing time signature, if present
    existing_time = attributes.find(f"./{ns}time")
    if existing_time is not None:
        attributes.remove(existing_time)

    # Add new time signature
    time_elem = ET.SubElement(attributes, f"{ns}time")
    ET.SubElement(time_elem, f"{ns}beats").text = beats
    ET.SubElement(time_elem, f"{ns}beat-type").text = beat_type

    # Save the modified file
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)