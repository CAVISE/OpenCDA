"""
Generate SUMO networks from OpenDRIVE files.

This script converts OpenDRIVE files to SUMO network format using netconvert,
then manually inserts traffic light landmarks retrieved from the OpenDRIVE data.
"""

import argparse
import bisect
import collections
import logging
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional, List, Set, Tuple

import lxml.etree as ET

import os
import sys

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import carla
import sumolib


class SumoTopology(object):
    """
    Topology representation of a SUMO network.

    Provides methods to query and navigate the topology of a SUMO network,
    including road connections, junctions, and lane mappings between OpenDRIVE
    and SUMO coordinate systems.

    Parameters
    ----------
    topology : dict
        Dictionary mapping (road_id, lane_id) to sets of successor
        (road_id, lane_id) tuples for standard roads.
    paths : dict
        Dictionary containing junction path information mapping
        (road_id, lane_id) to sets of ((from_edge, from_lane), (to_edge, to_lane)).
    odr2sumo_ids : dict
        Dictionary mapping OpenDRIVE (road_id, lane_id) to sets of
        SUMO (edge_id, lane_index) tuples.

    Attributes
    ----------
    _topology : dict
        Dictionary mapping (road_id, lane_id) to sets of successor (road_id, lane_id) tuples.
    _paths : dict
        Dictionary containing junction path information.
    _odr2sumo_ids : dict
        Dictionary mapping OpenDRIVE (road_id, lane_id) to SUMO (edge_id, lane_index) tuples.
    """

    def __init__(
        self,
        topology: Dict[Tuple[str, int], Set[Tuple[str, int]]],
        paths: Dict[Tuple[str, int], Set[Tuple[Tuple[str, int], Tuple[str, int]]]],
        odr2sumo_ids: Dict[Tuple[str, int], Set[Tuple[str, int]]],
    ):
        # Contains only standard roads.
        self._topology = topology
        # Contaions only roads that belong to a junction.
        self._paths = paths
        # Mapped ids between sumo and opendrive.
        self._odr2sumo_ids = odr2sumo_ids

    # http://sumo.sourceforge.net/userdoc/Networks/Import/OpenDRIVE.html#dealing_with_lane_sections
    def get_sumo_id(self, odr_road_id: str, odr_lane_id: int, s: float = 0) -> Optional[Tuple[str, int]]:
        """
        Get SUMO edge and lane index from OpenDRIVE road and lane IDs.

        Parameters
        ----------
        odr_road_id : str
            OpenDRIVE road ID.
        odr_lane_id : int
            OpenDRIVE lane ID.
        s : float, optional
            Position along the road to select the correct SUMO edge when
            split into multiple edges due to lane sections. Default is 0.

        Returns
        -------
        sumo_id : tuple of (str, int) or None
            Tuple of (sumo_edge_id, sumo_lane_index), or None if not found.
        """
        if (odr_road_id, odr_lane_id) not in self._odr2sumo_ids:
            return None

        sumo_ids: List[Tuple[str, int]] = list(self._odr2sumo_ids[(odr_road_id, odr_lane_id)])

        if (len(sumo_ids)) == 1:
            return sumo_ids[0]

        else:
            # Ensures that all the related sumo edges belongs to the same
            # opendrive road but to different lane sections.
            assert len(set([edge.split(".", 1)[0] for edge, _ in sumo_ids])) == 1

            s_coords = [float(edge.split(".", 1)[1]) for edge, _ in sumo_ids]

            sorted_pairs = sorted(zip(s_coords, sumo_ids))
            sorted_s_coords, sorted_sumo_ids = zip(*sorted_pairs)
            index = bisect.bisect_left(sorted_s_coords, s, lo=1) - 1
            return sorted_sumo_ids[index]

    def is_junction(self, odr_road_id: str, odr_lane_id: int) -> bool:
        """
        Check if an OpenDRIVE road/lane pair belongs to a junction.

        Parameters
        ----------
        odr_road_id : str
            OpenDRIVE road ID.
        odr_lane_id : int
            OpenDRIVE lane ID.

        Returns
        -------
        is_junction : bool
            True if the pair belongs to a junction, False otherwise.
        """
        return (odr_road_id, odr_lane_id) in self._paths

    def get_successors(self, sumo_edge_id: str, sumo_lane_index: int) -> List[Tuple[str, int]]:
        """
        Get successor edges for a SUMO edge/lane pair.

        Parameters
        ----------
        sumo_edge_id : str
            SUMO edge ID.
        sumo_lane_index : int
            SUMO lane index.

        Returns
        -------
        successors : list of tuple
            List of (edge_id, lane_index) tuples for successor edges.
            Empty list if the edge is a junction.
        """
        if self.is_junction(sumo_edge_id, sumo_lane_index):
            return []

        return list(self._topology.get((sumo_edge_id, sumo_lane_index), set()))

    def get_incoming(self, odr_road_id: str, odr_lane_id: int) -> List[Tuple[str, int]]:
        """
        Get incoming edges for a junction path.

        Parameters
        ----------
        odr_road_id : str
            OpenDRIVE road ID.
        odr_lane_id : int
            OpenDRIVE lane ID.

        Returns
        -------
        incoming : list of tuple
            List of (edge_id, lane_index) tuples for incoming edges.
            Empty list if not a junction.
        """
        if not self.is_junction(odr_road_id, odr_lane_id):
            return []

        result = set([(connection[0][0], connection[0][1]) for connection in self._paths[(odr_road_id, odr_lane_id)]])
        return list(result)

    def get_path_connectivity(self, odr_road_id: str, odr_lane_id: int) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
        """
        Get full path connectivity for a junction.

        Parameters
        ----------
        odr_road_id : str
            OpenDRIVE road ID.
        odr_lane_id : int
            OpenDRIVE lane ID.

        Returns
        -------
        connections : list of tuple
            List of ((from_edge, from_lane), (to_edge, to_lane)) tuples.
            Empty list if not a junction.
        """
        return list(self._paths.get((odr_road_id, odr_lane_id), set()))


def build_topology(sumo_net: sumolib.net.Net) -> SumoTopology:
    """
    Build a SumoTopology object from a SUMO network.

    This function processes a SUMO network to extract its topology, including
    road connections, lane mappings, and junction information.

    Parameters
    ----------
    sumo_net : sumolib.net.Net
        The SUMO network object to process.

    Returns
    -------
    SumoTopology
        An object representing the topology of the SUMO network.

    Raises
    ------
    RuntimeError
        If the SUMO network contains lanes without original OpenDRIVE IDs.
    """
    # --------------------------
    # OpenDrive->Sumo mapped ids
    # --------------------------
    # Only takes into account standard roads.

    odr2sumo_ids: Dict[Tuple[str, int], Set[Tuple]] = {}
    for edge in sumo_net.getEdges():
        for lane in edge.getLanes():
            if lane.getParam("origId") is None:
                raise RuntimeError(
                    'Sumo lane {} does not have "origId" parameter. '
                    "Make sure that the --output.original-names parameter is "
                    "active when running netconvert.".format(lane.getID())
                )

            if len(lane.getParam("origId").split()) > 1:
                logging.warning("[Building topology] Sumo net contains joined opendrive roads.")

            for odr_id in lane.getParam("origId").split():
                odr_road_id, odr_lane_id = odr_id.split("_")
                if (odr_road_id, int(odr_lane_id)) not in odr2sumo_ids:
                    odr2sumo_ids[(odr_road_id, int(odr_lane_id))] = set()
                odr2sumo_ids[(odr_road_id, int(odr_lane_id))].add((edge.getID(), lane.getIndex()))

    # -----------
    # Connections
    # -----------
    topology: Dict[Tuple[str, int], Set[Tuple]] = {}
    paths: Dict[Tuple[str, int], Set[Tuple]] = {}

    for from_edge in sumo_net.getEdges():
        for to_edge in sumo_net.getEdges():
            connections = from_edge.getConnections(to_edge)
            for connection in connections:
                from_ = connection.getFromLane()
                to_ = connection.getToLane()
                from_edge_id, from_lane_index = from_.getEdge().getID(), from_.getIndex()
                to_edge_id, to_lane_index = to_.getEdge().getID(), to_.getIndex()

                if (from_edge_id, from_lane_index) not in topology:
                    topology[(from_edge_id, from_lane_index)] = set()

                topology[(from_edge_id, from_lane_index)].add((to_edge_id, to_lane_index))

                # Checking if the connection is an opendrive path.
                conn_odr_ids = connection.getParam("origId")
                if conn_odr_ids is not None:
                    if len(conn_odr_ids.split()) > 1:
                        logging.warning("[Building topology] Sumo net contains joined opendrive paths.")

                    for odr_id in conn_odr_ids.split():
                        odr_road_id, odr_lane_id = odr_id.split("_")
                        if (odr_road_id, int(odr_lane_id)) not in paths:
                            paths[(odr_road_id, int(odr_lane_id))] = set()

                        paths[(odr_road_id, int(odr_lane_id))].add(((from_edge_id, from_lane_index), (to_edge_id, to_lane_index)))

    return SumoTopology(topology, paths, odr2sumo_ids)


class SumoTrafficLight(object):
    """
    Traffic light representation for SUMO networks.

    Holds all necessary data to define a traffic light in SUMO, including
    connections, signal phases, and parameters for landmark association.

    Parameters
    ----------
    tlid : str
        Traffic light ID.
    program_id : str, optional
        Program ID for the traffic light.
    offset : int, optional
        Time offset for the program.
    tltype : str, optional
        Traffic light type.

    Attributes
    ----------
    id : str
        Traffic light ID.
    program_id : str
        Program ID.
    offset : int
        Time offset.
    type : str
        Traffic light type.
    phases : list of Phase
        List of signal phases.
    parameters : set of tuple
        Set of (link_index, landmark_id) parameter pairs.
    connections : set of Connection
        Set of controlled connections.
    """

    DEFAULT_DURATION_GREEN_PHASE = 42
    DEFAULT_DURATION_YELLOW_PHASE = 3
    DEFAULT_DURATION_RED_PHASE = 3

    Phase = collections.namedtuple("Phase", "duration state min_dur max_dur next name")
    Connection = collections.namedtuple("Connection", "tlid from_road to_road from_lane to_lane link_index")

    def __init__(self, tlid: str, program_id: str = "0", offset: int = 0, tltype: str = "static"):
        self.id = tlid
        self.program_id = program_id
        self.offset = offset
        self.type = tltype

        self.phases: List[SumoTrafficLight.Phase] = []
        self.parameters: Set[Tuple[int, str]] = set()
        self.connections: Set[SumoTrafficLight.Connection] = set()

    @staticmethod
    def generate_tl_id(from_edge: str, to_edge: str) -> str:
        """
        Generate traffic light ID from junction connectivity.

        Parameters
        ----------
        from_edge : str
            Incoming edge ID.
        to_edge : str
            Outgoing edge ID.

        Returns
        -------
        tl_id : str
            Traffic light ID in format "from_edge:to_edge".
        """
        return "{}:{}".format(from_edge, to_edge)

    @staticmethod
    def generate_default_program(tl: "SumoTrafficLight") -> None:
        """
        Generate default program for the given sumo traffic light

        Creates a simple program with green-yellow-red phases for each
        incoming road, cycling through them sequentially.

        Parameters
        ----------
        tl : SumoTrafficLight
            Traffic light to generate program for.
        """
        incoming_roads = [connection.from_road for connection in tl.connections]
        for road in set(incoming_roads):
            phase_green = ["r"] * len(tl.connections)
            phase_yellow = ["r"] * len(tl.connections)
            phase_red = ["r"] * len(tl.connections)

            for connection in tl.connections:
                if connection.from_road == road:
                    phase_green[connection.link_index] = "g"
                    phase_yellow[connection.link_index] = "y"

            tl.add_phase(SumoTrafficLight.DEFAULT_DURATION_GREEN_PHASE, "".join(phase_green))
            tl.add_phase(SumoTrafficLight.DEFAULT_DURATION_YELLOW_PHASE, "".join(phase_yellow))
            tl.add_phase(SumoTrafficLight.DEFAULT_DURATION_RED_PHASE, "".join(phase_red))

    def add_phase(self, duration: int, state: str, min_dur: int = -1, max_dur: int = -1, next_phase: Optional[int] = None, name: str = "") -> None:
        """
         Add a signal phase to the traffic light program.

        Parameters
        ----------
        duration : int
            Phase duration in seconds.
        state : str
            Signal state string
        min_dur : int, optional
            Minimum duration for actuated control.
        max_dur : int, optional
            Maximum duration for actuated control.
        next_phase : int, optional
            Index of next phase.
        name : str, optional
            Phase name. Default is empty string
        """
        self.phases.append(SumoTrafficLight.Phase(duration, state, min_dur, max_dur, next_phase, name))

    def add_parameter(self, key: int, value: str) -> None:
        """
        Add a parameter to the traffic light.

        Parameters
        ----------
        key : int
            Link index.
        value : str
            Landmark ID.
        """
        self.parameters.add((key, value))

    def add_connection(self, connection: "SumoTrafficLight.Connection") -> None:
        """
        Add a controlled connection to the traffic light.

        Parameters
        ----------
        connection : Connection
            Connection namedtuple to add.
        """
        self.connections.add(connection)

    def add_landmark(self, landmark_id: str, tlid: str, from_road: str, to_road: str, from_lane: int, to_lane: int, link_index: int = -1) -> bool:
        """
        Add an OpenDRIVE landmark to the traffic light.

        Parameters
        ----------
        landmark_id : str
            OpenDRIVE landmark ID.
        tlid : str
            Traffic light ID.
        from_road : str
            Incoming edge ID.
        to_road : str
            Outgoing edge ID.
        from_lane : int
            Incoming lane index.
        to_lane : int
            Outgoing lane index.
        link_index : int, optional
            Link index. Default is -1 (auto-assign).

        Returns
        -------
        success : bool
            True if landmark was successfully added, False if duplicate.
        """
        if link_index == -1:
            link_index = len(self.connections)

        def is_same_connection(c1: SumoTrafficLight.Connection, c2: SumoTrafficLight.Connection) -> bool:
            return c1.from_road == c2.from_road and c1.to_road == c2.to_road and c1.from_lane == c2.from_lane and c1.to_lane == c2.to_lane

        connection = SumoTrafficLight.Connection(tlid, from_road, to_road, from_lane, to_lane, link_index)
        if any([is_same_connection(connection, c) for c in self.connections]):
            logging.warning("Different landmarks controlling the same connection. Only one will be included.")
            return False

        self.add_connection(connection)
        self.add_parameter(link_index, landmark_id)
        return True

    def to_xml(self) -> ET.Element:
        """
        Convert traffic light to XML element for SUMO network file.

        Returns
        -------
        xml_tag : lxml.etree.Element
            XML element representing the traffic light logic.
        """
        info = {"id": self.id, "type": self.type, "programID": self.program_id, "offset": str(self.offset)}

        xml_tag = ET.Element("tlLogic", info)
        for phase in self.phases:
            ET.SubElement(xml_tag, "phase", {"state": phase.state, "duration": str(phase.duration)})
        for parameter in sorted(self.parameters, key=lambda x: x[0]):
            ET.SubElement(xml_tag, "param", {"key": "linkSignalID:" + str(parameter[0]), "value": str(parameter[1])})

        return xml_tag


def _netconvert_carla_impl(xodr_file: str, output: str, tmpdir: str, guess_tls: bool = False) -> None:
    """
    Internal implementation of the netconvert_carla function.

    This function handles the actual conversion process from OpenDRIVE to SUMO format,
    including temporary file management and error handling.

    Parameters
    ----------
    xodr_file : str
        Path to the input OpenDRIVE file.
    output : str
        Path where the output SUMO network file will be saved.
    tmpdir : str
        Temporary directory for intermediate files.
    guess_tls : bool, optional
        If True, attempts to guess traffic lights at intersections. Default is False.

    Raises
    ------
    RuntimeError
        If the conversion process fails.
    """
    # ----------
    # netconvert
    # ----------
    basename = os.path.splitext(os.path.basename(xodr_file))[0]
    tmp_sumo_net = os.path.join(tmpdir, basename + ".net.xml")

    try:
        basedir = os.path.dirname(os.path.realpath(__file__))
        result = subprocess.call(
            [
                "netconvert",
                "--opendrive",
                xodr_file,
                "--output-file",
                tmp_sumo_net,
                "--geometry.min-radius.fix",
                "--geometry.remove",
                "--opendrive.curve-resolution",
                "1",
                "--opendrive.import-all-lanes",
                "--type-files",
                os.path.join(basedir, "data/opendrive_netconvert.typ.xml"),
                # Necessary to link odr and sumo ids.
                "--output.original-names",
                # Discard loading traffic lights as them will be inserted
                # manually afterwards.
                "--tls.discard-loaded",
                "true",
            ]
        )
    except subprocess.CalledProcessError:
        raise RuntimeError("There was an error when executing netconvert.")
    else:
        if result != 0:
            raise RuntimeError("There was an error when executing netconvert.")

    # --------
    # Sumo net
    # --------
    sumo_net = sumolib.net.readNet(tmp_sumo_net)
    sumo_topology = build_topology(sumo_net)

    # ---------
    # Carla map
    # ---------
    with open(xodr_file, "r") as f:
        carla_map = carla.Map("netconvert", str(f.read()))

    # ---------
    # Landmarks
    # ---------
    tls = {}  # {tlsid: SumoTrafficLight}

    landmarks = carla_map.get_all_landmarks_of_type("1000001")
    for landmark in landmarks:
        if landmark.name == "":
            # This is a workaround to avoid adding traffic lights without controllers.
            logging.warning("Landmark %s has not a valid name.", landmark.name)
            continue

        road_id = str(landmark.road_id)
        for from_lane, to_lane in landmark.get_lane_validities():
            for lane_id in range(from_lane, to_lane + 1):
                if lane_id == 0:
                    continue

                wp = carla_map.get_waypoint_xodr(landmark.road_id, lane_id, landmark.s)
                if wp is None:
                    logging.warning(
                        "Could not find waypoint for landmark {} (road_id: {}, lane_id: {}, s:{}".format(
                            landmark.id, landmark.road_id, lane_id, landmark.s
                        )
                    )
                    continue

                # When the landmark belongs to a junction, we place te traffic light at the
                # entrance of the junction.
                if wp.is_junction and sumo_topology.is_junction(road_id, lane_id):
                    tlid = str(wp.get_junction().id)
                    if tlid not in tls:
                        tls[tlid] = SumoTrafficLight(tlid)
                    tl = tls[tlid]

                    if guess_tls:
                        for from_edge, from_lane in sumo_topology.get_incoming(road_id, lane_id):
                            successors = sumo_topology.get_successors(from_edge, from_lane)
                            for to_edge, to_lane in successors:
                                tl.add_landmark(landmark.id, tl.id, from_edge, to_edge, from_lane, to_lane)

                    else:
                        connections = sumo_topology.get_path_connectivity(road_id, lane_id)
                        for from_, to_ in connections:
                            from_edge, from_lane = from_
                            to_edge, to_lane = to_

                            tl.add_landmark(landmark.id, tl.id, from_edge, to_edge, from_lane, to_lane)

                # When the landmarks does not belong to a junction (i.e., belongs to a std road),
                # we place the traffic light between that std road and its successor.
                elif not wp.is_junction and not sumo_topology.is_junction(road_id, lane_id):
                    from_id = sumo_topology.get_sumo_id(road_id, lane_id, landmark.s)
                    if from_id is None:
                        logging.warning("No SUMO id found for road %s lane %s.", road_id, lane_id)
                        continue
                    from_edge, from_lane = from_id

                    for to_edge, to_lane in sumo_topology.get_successors(from_edge, from_lane):
                        tlid = SumoTrafficLight.generate_tl_id(from_edge, to_edge)
                        if tlid not in tls:
                            tls[tlid] = SumoTrafficLight(tlid)
                        tl = tls[tlid]

                        tl.add_landmark(landmark.id, tl.id, from_edge, to_edge, from_lane, to_lane)

                else:
                    logging.warning("Landmark %s could not be added.", landmark.id)

    # ---------------
    # Modify sumo net
    # ---------------
    parser = ET.XMLParser(remove_blank_text=True)
    tree = ET.parse(tmp_sumo_net, parser)
    root = tree.getroot()

    for tl in tls.values():
        SumoTrafficLight.generate_default_program(tl)
        edges_tags: Any = tree.xpath("//edge")
        if not edges_tags:
            raise RuntimeError("No edges found in sumo net.")
        root.insert(root.index(edges_tags[-1]) + 1, tl.to_xml())

        for connection in tl.connections:
            tags: Any = tree.xpath(
                '//connection[@from="{}" and @to="{}" and @fromLane="{}" and @toLane="{}"]'.format(
                    connection.from_road, connection.to_road, connection.from_lane, connection.to_lane
                )
            )

            if tags:
                if len(tags) > 1:
                    logging.warning(
                        "Found repeated connections from={} to={} fromLane={} toLane={}.".format(
                            connection.from_road, connection.to_road, connection.from_lane, connection.to_lane
                        )
                    )

                tags[0].set("tl", str(connection.tlid))
                tags[0].set("linkIndex", str(connection.link_index))
            else:
                logging.warning(
                    "Not found connection from={} to={} fromLane={} toLane={}.".format(
                        connection.from_road, connection.to_road, connection.from_lane, connection.to_lane
                    )
                )

    tree.write(output, pretty_print=True, encoding="UTF-8", xml_declaration=True)


def netconvert_carla(xodr_file: str, output: str, guess_tls: bool = False) -> None:
    """
    Generate a SUMO network file from an OpenDRIVE file.

    This function converts an OpenDRIVE file to a SUMO network file, with optional
    traffic light detection at intersections.

    Parameters
    ----------
    xodr_file : str
        Path to the input OpenDRIVE file (*.xodr).
    output : str
        Path where the output SUMO network file will be saved (*.net.xml).
    guess_tls : bool, optional
        If True, attempts to guess traffic lights at intersections. Default is False.

    Notes
    -----
    This function requires SUMO to be installed and the SUMO_HOME environment
    variable to be properly set.
    """
    try:
        tmpdir = tempfile.mkdtemp()
        _netconvert_carla_impl(xodr_file, output, tmpdir, guess_tls)

    finally:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("xodr_file", help="opendrive file (*.xodr")
    argparser.add_argument("--output", "-o", default="net.net.xml", type=str, help="output file (default: net.net.xml)")
    argparser.add_argument("--guess-tls", action="store_true", help="guess traffic lights at intersections (default: False)")
    args = argparser.parse_args()

    netconvert_carla(args.xodr_file, args.output, args.guess_tls)
