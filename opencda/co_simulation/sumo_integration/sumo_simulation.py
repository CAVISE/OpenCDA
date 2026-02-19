"""This module is responsible for the management of the sumo simulation."""

import collections
import enum
import logging
import os

import carla  # pylint: disable=import-error
import sumolib  # pylint: disable=import-error
import traci  # pylint: disable=import-error

from opencda.co_simulation.sumo_integration.constants import INVALID_ACTOR_ID

import lxml.etree as ET  # pylint: disable=import-error

logger = logging.getLogger("cavise.sumo_simulation")

# ==================================================================================================
# -- sumo definitions ------------------------------------------------------------------------------
# ==================================================================================================


# https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html#signal_state_definitions
class SumoSignalState(object):
    """
    SumoSignalState contains the different traffic light states.
    """

    RED = "r"
    YELLOW = "y"
    GREEN = "G"
    GREEN_WITHOUT_PRIORITY = "g"  # noqa: DC01
    GREEN_RIGHT_TURN = "s"  # noqa: DC01
    RED_YELLOW = "u"  # noqa: DC01
    OFF_BLINKING = "o"  # noqa: DC01
    OFF = "O"


# https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html#abstract_vehicle_class
class SumoActorClass(enum.Enum):
    """
    SumoActorClass enumerates the different sumo actor classes.
    """

    IGNORING = "ignoring"  # noqa: DC01
    PRIVATE = "private"  # noqa: DC01
    EMERGENCY = "emergency"  # noqa: DC01
    AUTHORITY = "authority"  # noqa: DC01
    ARMY = "army"  # noqa: DC01
    VIP = "vip"  # noqa: DC01
    PEDESTRIAN = "pedestrian"  # noqa: DC01
    PASSENGER = "passenger"  # noqa: DC01
    HOV = "hov"  # noqa: DC01
    TAXI = "taxi"  # noqa: DC01
    BUS = "bus"  # noqa: DC01
    COACH = "coach"  # noqa: DC01
    DELIVERY = "delivery"  # noqa: DC01
    TRUCK = "truck"  # noqa: DC01
    TRAILER = "trailer"  # noqa: DC01
    MOTORCYCLE = "motorcycle"  # noqa: DC01
    MOPED = "moped"  # noqa: DC01
    BICYCLE = "bicycle"  # noqa: DC01
    EVEHICLE = "evehicle"  # noqa: DC01
    TRAM = "tram"  # noqa: DC01
    RAIL_URBAN = "rail_urban"  # noqa: DC01
    RAIL = "rail"  # noqa: DC01
    RAIL_ELECTRIC = "rail_electric"  # noqa: DC01
    RAIL_FAST = "rail_fast"  # noqa: DC01
    SHIP = "ship"  # noqa: DC01
    CUSTOM1 = "custom1"  # noqa: DC01
    CUSTOM2 = "custom2"  # noqa: DC01


SumoActor = collections.namedtuple("SumoActor", "type_id vclass transform signals extent color")

# ==================================================================================================
# -- sumo traffic lights ---------------------------------------------------------------------------
# ==================================================================================================


class SumoTLLogic(object):
    """
    SumoTLLogic holds the data relative to a traffic light in sumo.
    """

    def __init__(self, tlid, states, parameters):
        self.tlid = tlid
        self.states = states

        self._landmark2link = {}
        self._link2landmark = {}
        for link_index, landmark_id in parameters.items():
            # Link index information is added in the parameter as 'linkSignalID:x'
            link_index = int(link_index.split(":")[1])

            if landmark_id not in self._landmark2link:
                self._landmark2link[landmark_id] = []
            self._landmark2link[landmark_id].append((tlid, link_index))
            self._link2landmark[(tlid, link_index)] = landmark_id

    def get_number_signals(self):
        """
        Returns number of internal signals of the traffic light.
        """
        if len(self.states) > 0:
            return len(self.states[0])
        return 0

    def get_all_signals(self):
        """
        Returns all the signals of the traffic light.
            :returns list: [(tlid, link_index), (tlid, link_index), ...]
        """
        return [(self.tlid, i) for i in range(self.get_number_signals())]

    def get_all_landmarks(self):
        """
        Returns all the landmarks associated with this traffic light.
        """
        return self._landmark2link.keys()

    def get_associated_signals(self, landmark_id):
        """
        Returns all the signals associated with the given landmark.
            :returns list: [(tlid, link_index), (tlid, link_index), ...]
        """
        return self._landmark2link.get(landmark_id, [])


class SumoTLManager(object):
    """
    SumoTLManager is responsible for the management of the sumo traffic lights (i.e., keeps control
    of the current program, phase, ...)
    """

    def __init__(self):
        self._tls = {}  # {tlid: {program_id: SumoTLLogic}
        self._current_program = {}  # {tlid: program_id}
        self._current_phase = {}  # {tlid: index_phase}

        for tlid in traci.trafficlight.getIDList():
            self.subscribe(tlid)

            self._tls[tlid] = {}
            for tllogic in traci.trafficlight.getAllProgramLogics(tlid):
                states = [phase.state for phase in tllogic.getPhases()]
                parameters = tllogic.getParameters()
                tl = SumoTLLogic(tlid, states, parameters)
                self._tls[tlid][tllogic.programID] = tl

            # Get current status of the traffic lights.
            self._current_program[tlid] = traci.trafficlight.getProgram(tlid)
            self._current_phase[tlid] = traci.trafficlight.getPhase(tlid)

        self._off = False

    @staticmethod
    def subscribe(tlid):
        """
        Subscribe the given traffic ligth to the following variables:

            * Current program.
            * Current phase.
        """
        traci.trafficlight.subscribe(
            tlid,
            [
                traci.constants.TL_CURRENT_PROGRAM,
                traci.constants.TL_CURRENT_PHASE,
            ],
        )

    @staticmethod
    def unsubscribe(tlid):
        """
        Unsubscribe the given traffic ligth from receiving updated information each step.
        """
        traci.trafficlight.unsubscribe(tlid)

    def get_all_signals(self):
        """
        Returns all the traffic light signals.
        """
        signals = set()
        for tlid, program_id in self._current_program.items():
            signals.update(self._tls[tlid][program_id].get_all_signals())
        return signals

    def get_all_landmarks(self):
        """
        Returns all the landmarks associated with a traffic light in the simulation.
        """
        landmarks = set()
        for tlid, program_id in self._current_program.items():
            landmarks.update(self._tls[tlid][program_id].get_all_landmarks())
        return landmarks

    def get_all_associated_signals(self, landmark_id):
        """
        Returns all the signals associated with the given landmark.
            :returns list: [(tlid, link_index), (tlid, link_index), ...]
        """
        signals = set()
        for tlid, program_id in self._current_program.items():
            signals.update(self._tls[tlid][program_id].get_associated_signals(landmark_id))
        return signals

    def get_state(self, landmark_id):
        """
        Returns the traffic light state of the signals associated with the given landmark.
        """
        states = set()
        for tlid, link_index in self.get_all_associated_signals(landmark_id):
            current_program = self._current_program[tlid]
            current_phase = self._current_phase[tlid]

            tl = self._tls[tlid][current_program]
            states.update(tl.states[current_phase][link_index])

        if len(states) == 1:
            return states.pop()
        elif len(states) > 1:
            logger.warning(f"Landmark {landmark_id} is associated with signals with different states")
            return SumoSignalState.RED
        else:
            return None

    def set_state(self, landmark_id, state):
        """
        Updates the state of all the signals associated with the given landmark.
        """
        for tlid, link_index in self.get_all_associated_signals(landmark_id):
            traci.trafficlight.setLinkState(tlid, link_index, state)
        return True

    def switch_off(self):
        """
        Switch off all traffic lights.
        """
        for tlid, link_index in self.get_all_signals():
            traci.trafficlight.setLinkState(tlid, link_index, SumoSignalState.OFF)
        self._off = True

    def tick(self):
        """
        Tick to traffic light manager
        """
        if self._off is False:
            for tl_id in traci.trafficlight.getIDList():
                results = traci.trafficlight.getSubscriptionResults(tl_id)
                current_program = results[traci.constants.TL_CURRENT_PROGRAM]
                current_phase = results[traci.constants.TL_CURRENT_PHASE]

                if current_program != "online":
                    self._current_program[tl_id] = current_program
                    self._current_phase[tl_id] = current_phase


# ==================================================================================================
# -- sumo simulation -------------------------------------------------------------------------------
# ==================================================================================================


def _get_sumo_net(cfg_file):
    """
    Returns sumo net.

    This method reads the sumo configuration file and retrieve the sumo net filename to create the
    net.
    """
    cfg_file = os.path.join(os.getcwd(), cfg_file)

    tree = ET.parse(cfg_file)
    tag = tree.find(".//net-file")
    if tag is None:
        return None

    net_file = os.path.join(os.path.dirname(cfg_file), tag.get("value"))
    logger.debug(f"Reading net file: {net_file}")

    sumo_net = sumolib.net.readNet(net_file)
    return sumo_net


class SumoSimulation(object):
    """
    SumoSimulation is responsible for the management of the sumo simulation.
    """

    def __init__(self, cfg_file, step_length, host=None, port=None, sumo_gui=False, client_order=1):
        if sumo_gui is True:
            sumolib.checkBinary("sumo-gui")
        else:
            sumolib.checkBinary("sumo")

        if host is None or port is None:
            logger.error("Error in sumo section of scenario YAML config.")
        else:
            logger.info(f"Connection to sumo server. Host: {host} Port: {port}")
            traci.init(host=host, port=port)

        traci.setOrder(client_order)

        # Retrieving net from configuration file.
        self.net = _get_sumo_net(cfg_file)

        # Creating a random route to be able to spawn carla actors.
        traci.route.add("carla_route", [traci.edge.getIDList()[0]])

        # Variable to asign an id to new added actors.
        self._sequential_id = 0  # noqa: DC05

        # Structures to keep track of the spawned and destroyed vehicles at each time step.
        self.spawned_actors = set()
        self.destroyed_actors = set()

        # Traffic light manager.
        self.traffic_light_manager = SumoTLManager()

    @property
    def traffic_light_ids(self):
        return self.traffic_light_manager.get_all_landmarks()

    @staticmethod
    def subscribe(actor_id):
        """
        Subscribe the given actor to the following variables:

            * Type.
            * Vehicle class.
            * Color.
            * Length, Width, Height.
            * Position3D (i.e., x, y, z).
            * Angle, Slope.
            * Speed.
            * Lateral speed.
            * Signals.
        """
        traci.vehicle.subscribe(
            actor_id,
            [
                traci.constants.VAR_TYPE,
                traci.constants.VAR_VEHICLECLASS,
                traci.constants.VAR_COLOR,
                traci.constants.VAR_LENGTH,
                traci.constants.VAR_WIDTH,
                traci.constants.VAR_HEIGHT,
                traci.constants.VAR_POSITION3D,
                traci.constants.VAR_ANGLE,
                traci.constants.VAR_SLOPE,
                traci.constants.VAR_SPEED,
                traci.constants.VAR_SPEED_LAT,
                traci.constants.VAR_SIGNALS,
            ],
        )

    @staticmethod
    def unsubscribe(actor_id):
        """
        Unsubscribe the given actor from receiving updated information each step.
        """
        traci.vehicle.unsubscribe(actor_id)

    def get_net_offset(self):
        """
        Accessor for sumo net offset.
        """
        if self.net is None:
            return (0, 0)
        return self.net.getLocationOffset()

    @staticmethod
    def get_actor(actor_id):
        """
        Accessor for sumo actor.
        """
        results = traci.vehicle.getSubscriptionResults(actor_id)

        type_id = results[traci.constants.VAR_TYPE]
        vclass = SumoActorClass(results[traci.constants.VAR_VEHICLECLASS])
        color = results[traci.constants.VAR_COLOR]

        length = results[traci.constants.VAR_LENGTH]
        width = results[traci.constants.VAR_WIDTH]
        height = results[traci.constants.VAR_HEIGHT]

        location = list(results[traci.constants.VAR_POSITION3D])
        rotation = [results[traci.constants.VAR_SLOPE], results[traci.constants.VAR_ANGLE], 0.0]
        transform = carla.Transform(carla.Location(location[0], location[1], location[2]), carla.Rotation(rotation[0], rotation[1], rotation[2]))

        signals = results[traci.constants.VAR_SIGNALS]
        extent = carla.Vector3D(length / 2.0, width / 2.0, height / 2.0)

        return SumoActor(type_id, vclass, transform, signals, extent, color)

    def spawn_actor(self, type_id, id, color=None):
        """
        Spawns a new actor.

            :param type_id: vtype to be spawned.
            :param color: color attribute for this specific actor.
            :return: actor id if the actor is successfully spawned. Otherwise, INVALID_ACTOR_ID.
        """
        try:
            traci.vehicle.add(id, "carla_route", typeID=type_id)
        except traci.exceptions.TraCIException as error:
            logger.error(f"Spawn sumo actor failed: {error}")
            return INVALID_ACTOR_ID

        if color is not None:
            color = color.split(",")
            traci.vehicle.setColor(id, color)

        self._sequential_id += 1  # noqa: DC05

        return id

    @staticmethod
    def destroy_actor(actor_id):
        """
        Destroys the given actor.
        """
        # traci.vehicle.remove(actor_id)

        if actor_id in traci.vehicle.getIDList():
            traci.vehicle.remove(actor_id)
        else:
            logger.warning(f"Tried to remove nonexistent SUMO actor: {actor_id}")

    def get_traffic_light_state(self, landmark_id):
        """
        Accessor for traffic light state.

        If the traffic ligth does not exist, returns None.
        """
        return self.traffic_light_manager.get_state(landmark_id)

    def switch_off_traffic_lights(self):
        """
        Switch off all traffic lights.
        """
        self.traffic_light_manager.switch_off()

    def synchronize_vehicle(self, vehicle_id, transform, signals=None):
        """
        Updates vehicle state.

            :param vehicle_id: id of the actor to be updated.
            :param transform: new vehicle transform (i.e., position and rotation).
            :param signals: new vehicle signals.
            :return: True if successfully updated. Otherwise, False.
        """
        loc_x, loc_y = transform.location.x, transform.location.y
        yaw = transform.rotation.yaw

        traci.vehicle.moveToXY(vehicle_id, "", 0, loc_x, loc_y, angle=yaw, keepRoute=2)
        if signals is not None:
            traci.vehicle.setSignals(vehicle_id, signals)
        return True

    def synchronize_traffic_light(self, landmark_id, state):
        """
        Updates traffic light state.

            :param landmark_id: id of the traffic light to be updated (logic id, link index).
            :param state: new traffic light state.
            :return: True if successfully updated. Otherwise, False.
        """
        self.traffic_light_manager.set_state(landmark_id, state)

    def tick(self):
        """
        Tick to sumo simulation.
        """
        traci.simulationStep()
        self.traffic_light_manager.tick()

        # Update data structures for the current frame.
        self.spawned_actors = set(traci.simulation.getDepartedIDList())
        self.destroyed_actors = set(traci.simulation.getArrivedIDList())

    @staticmethod
    def close():
        """
        Closes traci client.
        """
        traci.close()
