"""
Script to generate sumo nets based on opendrive files. Internally,
it uses netconvert to generatet he net and inserts, manually, the traffic
light landmarks retrieved from the opendrive.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import logging
import shutil
import subprocess
import tempfile
from typing import Any, TypeAlias

import os
import sys


ET: Any = None
carla: Any = None
sumolib: Any = None

RoadLaneId: TypeAlias = tuple[str, int]
ConnectionPath: TypeAlias = tuple[RoadLaneId, RoadLaneId]
TopologyMap: TypeAlias = dict[RoadLaneId, set[RoadLaneId]]
PathsMap: TypeAlias = dict[RoadLaneId, set[ConnectionPath]]
OdrToSumoIdsMap: TypeAlias = dict[RoadLaneId, set[RoadLaneId]]


def _load_lxml_etree() -> Any:
    """
    Load lxml lazily so argparse can run without conversion dependencies.
    """
    global ET
    if ET is None:
        import lxml.etree as etree_module

        ET = etree_module
    return ET


def _load_carla() -> Any:
    """
    Load CARLA lazily so CLI help works even outside the CARLA environment.
    """
    global carla
    if carla is None:
        import carla as carla_module

        carla = carla_module
    return carla


def _load_sumolib() -> Any:
    """
    Load sumolib lazily so argparse can run before SUMO_HOME is configured.
    """
    global sumolib
    if sumolib is None:
        if "SUMO_HOME" not in os.environ:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        sumo_tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
        if sumo_tools_path not in sys.path:
            sys.path.append(sumo_tools_path)

        import sumolib as sumolib_module

        sumolib = sumolib_module
    return sumolib


def _load_conversion_dependencies() -> None:
    """
    Load dependencies required by both netconvert and CARLA landmark handling.
    """
    _load_lxml_etree()
    _load_carla()
    _load_sumolib()


class SumoTopology(object):
    """
    This object holds the topology of a sumo net. Internally, the information
    is structured as follows:

    Parameters
    ----------
    topology : {
            (road_id, lane_id): [(successor_road_id, succesor_lane_id), ...], ...}
        - paths: {
            (road_id, lane_id): [
                ((in_road_id, in_lane_id), (out_road_id, out_lane_id)), ...
            ], ...}
        - odr2sumo_ids: {
            (odr_road_id, odr_lane_id): [(sumo_edge_id, sumo_lane_id), ...], ...}
    """

    def __init__(self, topology: TopologyMap, paths: PathsMap, odr2sumo_ids: OdrToSumoIdsMap) -> None:
        # Contains only standard roads.
        self._topology = topology
        # Contaions only roads that belong to a junction.
        self._paths = paths
        # Mapped ids between sumo and opendrive.
        self._odr2sumo_ids = odr2sumo_ids

    # http://sumo.sourceforge.net/userdoc/Networks/Import/OpenDRIVE.html#dealing_with_lane_sections
    def get_sumo_id(self, odr_road_id: str, odr_lane_id: int, s: float = 0) -> RoadLaneId | None:
        """
        Returns the pair (sumo_edge_id, sumo_lane index) corresponding to the
        provided odr pair. The argument 's' allows selecting the better sumo
        edge when it has been split into different edges due to different odr
        lane sections.
        """
        if (odr_road_id, odr_lane_id) not in self._odr2sumo_ids:
            return None

        sumo_ids = list(self._odr2sumo_ids[(odr_road_id, odr_lane_id)])

        if (len(sumo_ids)) == 1:
            return sumo_ids[0]

        else:
            # Ensures that all the related sumo edges belongs to the same
            # opendrive road but to different lane sections.
            assert set([edge.split(".", 1)[0] for edge, _ in sumo_ids]) == 1

            s_coords = [float(edge.split(".", 1)[1]) for edge, _ in sumo_ids]
            sorted_pairs = sorted(zip(s_coords, sumo_ids))
            s_coords_sorted, sumo_ids_sorted = zip(*sorted_pairs)
            index = bisect.bisect_left(s_coords_sorted, s, lo=1) - 1
            return sumo_ids_sorted[index]

    def is_junction(self, odr_road_id: str, odr_lane_id: int) -> bool:
        """
        Checks whether the provided pair (odr_road_id, odr_lane_id) belongs to
         a junction.
        """
        return (odr_road_id, odr_lane_id) in self._paths

    def get_successors(self, sumo_edge_id: str, sumo_lane_index: int) -> list[RoadLaneId]:
        """
        Returns the successors (standard roads) of the provided pair
         (sumo_edge_id, sumo_lane_index)
        """
        if self.is_junction(sumo_edge_id, sumo_lane_index):
            return []

        return list(self._topology.get((sumo_edge_id, sumo_lane_index), set()))

    def get_incoming(self, odr_road_id: str, odr_lane_id: int) -> list[RoadLaneId]:
        """
        If the pair (odr_road_id, odr_lane_id) belongs to a junction,
        returns the incoming edges of the path. Otherwise,
        return and empty list.
        """
        if not self.is_junction(odr_road_id, odr_lane_id):
            return []

        result = set([(connection[0][0], connection[0][1]) for connection in self._paths[(odr_road_id, odr_lane_id)]])
        return list(result)

    def get_path_connectivity(self, odr_road_id: str, odr_lane_id: int) -> list[ConnectionPath]:
        """
        Returns incoming and outgoing roads of the
        pair (odr_road_id, odr_lane_id). If the provided
        pair not belongs to a junction, returns an empty list.
        """
        return list(self._paths.get((odr_road_id, odr_lane_id), set()))


def build_topology(sumo_net: sumolib.net.Net) -> SumoTopology:
    """
    Builds sumo topology.
    """
    # --------------------------
    # OpenDrive->Sumo mapped ids
    # --------------------------
    # Only takes into account standard roads.

    odr2sumo_ids: OdrToSumoIdsMap = {}
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
    topology: TopologyMap = {}
    paths: PathsMap = {}

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
    SumoTrafficLight holds all the necessary data to define a traffic light in sumo:

        * connections (tlid, from_road, to_road, from_lane, to_lane, link_index).
        * phases (duration, state, min_dur, max_dur, nex, name).
        * parameters.
    """

    DEFAULT_DURATION_GREEN_PHASE = 42
    DEFAULT_DURATION_YELLOW_PHASE = 3
    DEFAULT_DURATION_RED_PHASE = 3

    Phase = collections.namedtuple("Phase", "duration state min_dur max_dur next name")
    Connection = collections.namedtuple("Connection", "tlid from_road to_road from_lane to_lane link_index")

    def __init__(self, tlid: str, program_id: str = "0", offset: int = 0, tltype: str = "static") -> None:
        self.id = tlid
        self.program_id = program_id
        self.offset = offset
        self.type = tltype

        self.phases: list[SumoTrafficLight.Phase] = []
        self.parameters: set[tuple[int, object]] = set()
        self.connections: set[SumoTrafficLight.Connection] = set()

    @staticmethod
    def generate_tl_id(from_edge: str, to_edge: str) -> str:
        """
        Generates sumo traffic light id based on the junction connectivity.
        """
        return "{}:{}".format(from_edge, to_edge)

    @staticmethod
    def generate_default_program(tl: "SumoTrafficLight") -> None:
        """
        Generates a default program for the given sumo traffic light
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

    def add_phase(
        self,
        duration: int,
        state: str,
        min_dur: int = -1,
        max_dur: int = -1,
        next_phase: int | None = None,
        name: str = "",
    ) -> None:
        """
        Adds a new phase.
        """
        self.phases.append(SumoTrafficLight.Phase(duration, state, min_dur, max_dur, next_phase, name))

    def add_parameter(self, key: int, value: object) -> None:
        """
        Adds a new parameter.
        """
        self.parameters.add((key, value))

    def add_connection(self, connection: "SumoTrafficLight.Connection") -> None:
        """
        Adds a new connection.
        """
        self.connections.add(connection)

    def add_landmark(
        self,
        landmark_id: object,
        tlid: str,
        from_road: str,
        to_road: str,
        from_lane: int,
        to_lane: int,
        link_index: int = -1,
    ) -> bool:
        """
        Adds a new landmark.

        Returns True if the landmark is successfully included. Otherwise, returns False.
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

    def to_xml(self) -> Any:
        etree = _load_lxml_etree()
        info = {"id": self.id, "type": self.type, "programID": self.program_id, "offset": str(self.offset)}

        xml_tag = etree.Element("tlLogic", info)
        for phase in self.phases:
            etree.SubElement(xml_tag, "phase", {"state": phase.state, "duration": str(phase.duration)})
        for parameter in sorted(self.parameters, key=lambda parameter: parameter[0]):
            etree.SubElement(xml_tag, "param", {"key": "linkSignalID:" + str(parameter[0]), "value": str(parameter[1])})

        return xml_tag


def _netconvert_carla_impl(xodr_file: str, output: str, tmpdir: str, guess_tls: bool = False) -> None:
    """
    Implements netconvert carla.
    """
    _load_conversion_dependencies()

    # ----------
    # netconvert
    # ----------
    basename = os.path.splitext(os.path.basename(xodr_file))[0]
    tmp_sumo_net = os.path.join(tmpdir, basename + ".net.xml")

    try:
        basedir = os.path.dirname(os.path.realpath(__file__))
        netconvert_cmd = [
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
        ]
        type_file = os.path.join(basedir, "data/opendrive_netconvert.typ.xml")
        if os.path.exists(type_file):
            netconvert_cmd.extend(["--type-files", type_file])
        else:
            logging.warning(
                "SUMO OpenDRIVE type file was not found at %s. Running netconvert without custom type definitions.",
                type_file,
            )

        netconvert_cmd.extend(
            [
                # Necessary to link odr and sumo ids.
                "--output.original-names",
                # Discard loading traffic lights as them will be inserted
                # manually afterwards.
                "--tls.discard-loaded",
                "true",
            ]
        )
        result = subprocess.call(netconvert_cmd)
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
    tls: dict[str, SumoTrafficLight] = {}  # {tlsid: SumoTrafficLight}

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
                    sumo_id = sumo_topology.get_sumo_id(road_id, lane_id, landmark.s)
                    if sumo_id is None:
                        logging.warning(
                            "Could not map OpenDRIVE lane (%s, %s) to a SUMO lane for landmark %s.",
                            road_id,
                            lane_id,
                            landmark.id,
                        )
                        continue
                    from_edge, from_lane = sumo_id

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
    etree = _load_lxml_etree()
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(tmp_sumo_net, parser)
    root = tree.getroot()

    for tl in tls.values():
        SumoTrafficLight.generate_default_program(tl)
        edges_tags = tree.xpath("//edge")
        if not edges_tags:
            raise RuntimeError("No edges found in sumo net.")
        root.insert(root.index(edges_tags[-1]) + 1, tl.to_xml())

        for connection in tl.connections:
            tags = tree.xpath(
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

    output_dir = os.path.dirname(os.path.abspath(output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    tree.write(output, pretty_print=True, encoding="UTF-8", xml_declaration=True)


def netconvert_carla(xodr_file: str, output: str, guess_tls: bool = False) -> str:
    """
    Generates sumo net.

        :param xodr_file: opendrive file (*.xodr)
        :param output: output file (*.net.xml)
        :param guess_tls: guess traffic lights at intersections.
        :returns: path to the generated sumo net.
    """
    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp()
        _netconvert_carla_impl(xodr_file, output, tmpdir, guess_tls)

    finally:
        if tmpdir is not None and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

    return output


def netconvert_carla_from_opendrive(xodr_content: str, output: str, guess_tls: bool = False, map_name: str = "carla_map") -> str:
    """
    Generates a SUMO net from an OpenDRIVE string.

        :param xodr_content: opendrive map content.
        :param output: output file (*.net.xml)
        :param guess_tls: guess traffic lights at intersections.
        :param map_name: name used only for temporary conversion files.
        :returns: path to the generated sumo net.
    """
    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp()
        tmp_xodr_file = os.path.join(tmpdir, map_name + ".xodr")
        with open(tmp_xodr_file, "w", encoding="utf-8") as f:
            f.write(xodr_content)

        _netconvert_carla_impl(tmp_xodr_file, output, tmpdir, guess_tls)

    finally:
        if tmpdir is not None and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

    return output


def get_opendrive_from_carla(carla_host: str, carla_port: int, carla_timeout: float) -> str:
    """
    Retrieves the currently loaded CARLA map as OpenDRIVE.
    """
    carla_module = _load_carla()
    client = carla_module.Client(carla_host, carla_port)
    client.set_timeout(carla_timeout)
    world = client.get_world()
    return str(world.get_map().to_opendrive())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("xodr_file", nargs="?", help="opendrive file (*.xodr)")
    argparser.add_argument("--output", "-o", default="net.net.xml", type=str, help="output file (default: net.net.xml)")
    argparser.add_argument("--guess-tls", action="store_true", help="guess traffic lights at intersections (default: False)")
    argparser.add_argument(
        "--from-carla",
        action="store_true",
        help="read OpenDRIVE from the currently loaded CARLA map instead of an xodr file",
    )
    argparser.add_argument("--carla-host", type=str, default="carla", help="IP address or hostname of the CARLA server (default: 'carla')")
    argparser.add_argument("--carla-port", type=int, default=2000, help="Port of the CARLA server (default: 2000)")
    argparser.add_argument("--carla-timeout", type=float, default=30.0, help="Timeout of the CARLA server response in seconds (default: 30.0)")
    args = argparser.parse_args()

    if args.from_carla:
        if args.xodr_file is not None:
            argparser.error("xodr_file cannot be used together with --from-carla")
        opendrive = get_opendrive_from_carla(args.carla_host, args.carla_port, args.carla_timeout)
        netconvert_carla_from_opendrive(opendrive, args.output, args.guess_tls)
    else:
        if args.xodr_file is None:
            argparser.error("xodr_file is required unless --from-carla is used")
        netconvert_carla(args.xodr_file, args.output, args.guess_tls)
