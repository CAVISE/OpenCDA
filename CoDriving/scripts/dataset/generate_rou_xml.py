import argparse
import xml.etree.cElementTree as ET
from random import randint
from xml.dom import minidom

directions = [
    "left_up",
    "left_right",
    "left_down",
    "right_up",
    "right_left",
    "right_down",
    "down_up",
    "down_left",
    "down_right",
    "up_down",
    "up_left",
    "up_right",
]


def write_pretty_xml(element_tree, file_name):
    """Format xml"""
    rough_string = ET.tostring(element_tree.getroot(), "utf-8")
    
    parsed = minidom.parseString(rough_string)
    pretty_xml = parsed.toprettyxml(indent="  ")

    with open(file_name, "w", encoding="utf-8") as f:
        f.write(pretty_xml)


def generate(n: int, filename: str):
    routes = ET.Element("routes")

    ET.SubElement(
        routes,
        "vType",
        id="vehicle.seat.leon",
        accel="2.5",
        decel="4.5",
        sigma="0.5",
        length="5",
        minGap="2.5",
        maxSpeed="40",
        guiShape="passenger",
    )
    ET.SubElement(routes, "route", id="left_up", edges="E4 -E2 -E1 -E5")
    ET.SubElement(routes, "route", id="left_right", edges="E4 -E2 -E3 -E6")
    ET.SubElement(routes, "route", id="left_down", edges="E4 -E2 E0 -E7")

    ET.SubElement(routes, "route", id="right_up", edges="E6 E3 -E1 -E5")
    ET.SubElement(routes, "route", id="right_left", edges="E6 E3 E2 -E4")
    ET.SubElement(routes, "route", id="right_down", edges="E6 E3 E0 -E7")

    ET.SubElement(routes, "route", id="down_up", edges="E7 -E0 -E1 -E5")
    ET.SubElement(routes, "route", id="down_left", edges="E7 -E0 E2 -E4")
    ET.SubElement(routes, "route", id="down_right", edges="E7 -E0 -E3 -E6")

    ET.SubElement(routes, "route", id="up_down", edges="E5 E1 E0 -E7")
    ET.SubElement(routes, "route", id="up_left", edges="E5 E1 E2 -E4")
    ET.SubElement(routes, "route", id="up_right", edges="E5 E1 -E3 -E6")

    t = 1
    for i in range(n):
        r = randint(0, len(directions) - 1)
        ET.SubElement(
            routes,
            "vehicle",
            id=f"{i}",
            type="vehicle.seat.leon",
            route=directions[r],
            depart=f"{t}",
        )
        t += randint(1, 7)

    tree = ET.ElementTree(routes)
    write_pretty_xml(tree, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-n", "--number", type=int, help="Number of items to be generated", default=100
    )
    parser.add_argument(
        "-f", "--filename", type=str, help="Name of output file", default="gen.rou.xml"
    )
    
    args = parser.parse_args()
    n = args.number
    filename = args.filename
    generate(n, filename)
