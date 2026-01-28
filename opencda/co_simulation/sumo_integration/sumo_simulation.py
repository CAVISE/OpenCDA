"""This module is responsible for the management of the sumo simulation."""

import collections
import enum
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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
    
    Attributes
    ----------
    RED : str
        Red light state.
    YELLOW : str
        Yellow light state.
    GREEN : str
        Green light state.
    GREEN_WITHOUT_PRIORITY : str
        Green without priority state.
    GREEN_RIGHT_TURN : str
        Green right turn state.
    RED_YELLOW : str
        Red-yellow light state.
    OFF_BLINKING : str
        Off blinking state.
    OFF : str
        Off state.
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
    
    Attributes
    ----------
    IGNORING : str
        Ignoring class.
    PRIVATE : str
        Private vehicle class.
    EMERGENCY : str
        Emergency vehicle class.
    AUTHORITY : str
        Authority vehicle class.
    ARMY : str
        Army vehicle class.
    VIP : str
        VIP vehicle class.
    PEDESTRIAN : str
        Pedestrian class.
    PASSENGER : str
        Passenger vehicle class.
    HOV : str
        High-occupancy vehicle class.
    TAXI : str
        Taxi class.
    BUS : str
        Bus class.
    COACH : str
        Coach class.
    DELIVERY : str
        Delivery vehicle class.
    TRUCK : str
        Truck class.
    TRAILER : str
        Trailer class.
    MOTORCYCLE : str
        Motorcycle class.
    MOPED : str
        Moped class.
    BICYCLE : str
        Bicycle class.
    EVEHICLE : str
        Electric vehicle class.
    TRAM : str
        Tram class.
    RAIL_URBAN : str
        Urban rail class.
    RAIL : str
        Rail class.
    RAIL_ELECTRIC : str
        Electric rail class.
    RAIL_FAST : str
        Fast rail class.
    SHIP : str
        Ship class.
    CUSTOM1 : str
        Custom class 1.
    CUSTOM2 : str
        Custom class 2.
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
    
    Parameters
    ----------
    tlid : str
        Traffic light ID.
    states : List[str]
        List of traffic light states.
    parameters : Dict[str, Any]
        Dictionary of traffic light parameters.
        
    Attributes
    ----------
    tlid : str
        Traffic light ID.
    states : List[str]
        List of traffic light states.
     _landmark2link : Dict[str, List[Tuple[str, str]]]
        Maps landmark IDs to lists of (traffic light ID, link index) tuples.
    _link2landmark : Dict[Tuple[str, str], str]
        Maps (traffic light ID, link index) tuples to landmark IDs.
    """

    def __init__(self, tlid: str, states: List[str], parameters: Dict[str, Any]):
        self.tlid = tlid
        self.states = states

        self._landmark2link: Dict[str, List[Tuple[str, int]]] = {} #NOTE using List[Tuple[str, int] provide mistakes below but using List[Tuple[str, str]]] also provide mistakes
        self._link2landmark: Dict[Tuple[str, str], str] = {}
        for link_index, landmark_id in parameters.items():
            # Link index information is added in the parameter as 'linkSignalID:x'
            link_index = int(link_index.split(":")[1]) #NOTE Incompatible types in assignment (expression has type "int", variable has type "str")

            if landmark_id not in self._landmark2link:
                self._landmark2link[landmark_id] = []
            self._landmark2link[landmark_id].append((tlid, link_index))
            self._link2landmark[(tlid, link_index)] = landmark_id

    def get_number_signals(self) -> int:
        """
        Returns number of internal signals of the traffic light.
        """
        if len(self.states) > 0:
            return len(self.states[0])
        return 0

    def get_all_signals(self) ->  List[Tuple[str, int]]:
        """
        Returns all the signals of the traffic light.
            :returns list: [(tlid, link_index), (tlid, link_index), ...]
        """
        return [(self.tlid, i) for i in range(self.get_number_signals())]

    def get_all_landmarks(self) -> Any:
        """
        Get number of internal signals of the traffic light.

        Returns
        -------
        Dict_keys
            Number of signals.
        """
        return self._landmark2link.keys()

    def get_associated_signals(self, landmark_id: str) -> List[Tuple[str, int]]:
        """
        Get all the landmarks associated with this traffic light.
        
        Returns
        -------
        dict_keys
            Dictionary keys containing all landmark IDs.
        """
        return self._landmark2link.get(landmark_id, [])


class SumoTLManager(object):
    """
    SumoTLManager is responsible for the management of the sumo traffic lights (i.e., keeps control
    of the current program, phase, ...)
    """

    def __init__(self) -> None:
        self._tls: Dict[str, Dict[str, Any]] = {}  # {tlid: {program_id: SumoTLLogic}
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
    def subscribe(tlid: str) -> None:
        """
        Subscribe the given traffic light to receive updates.
        
        Subscribes to the following variables:
        - Current program
        - Current phase
        
        Parameters
        ----------
        tlid : str
            Traffic light ID.
        """
        traci.trafficlight.subscribe(
            tlid,
            [
                traci.constants.TL_CURRENT_PROGRAM,
                traci.constants.TL_CURRENT_PHASE,
            ],
        )

    @staticmethod
    def unsubscribe(tlid: str) -> None:
        """
        Unsubscribe the given traffic light from receiving updates.
        
        Parameters
        ----------
        tlid : str
            Traffic light ID.
        """
        traci.trafficlight.unsubscribe(tlid)

    def get_all_signals(self) -> Set[Tuple[str, int]]:
        """
        Get all the traffic light signals.
        
        Returns
        -------
        Set[Tuple[str, int]]
            Set of tuples containing (tlid, link_index).
        """
        signals = set()
        for tlid, program_id in self._current_program.items():
            signals.update(self._tls[tlid][program_id].get_all_signals())
        return signals

    def get_all_landmarks(self) -> Set[str]:
        """
        Get all the landmarks associated with traffic lights in the simulation.
        
        Returns
        -------
        Set[str]
            Set of landmark IDs.
        """
        landmarks = set()
        for tlid, program_id in self._current_program.items():
            landmarks.update(self._tls[tlid][program_id].get_all_landmarks())
        return landmarks

    def get_all_associated_signals(self, landmark_id: str) -> Set[Tuple[str, int]]:
        """
        Get all the signals associated with the given landmark.
        
        Parameters
        ----------
        landmark_id : str
            Landmark ID.
        
        Returns
        -------
        Set[Tuple[str, int]]
            Set of tuples containing (tlid, link_index).
        """
        signals = set()
        for tlid, program_id in self._current_program.items():
            signals.update(self._tls[tlid][program_id].get_associated_signals(landmark_id))
        return signals

    def get_state(self, landmark_id: str) -> Optional[str]:
        """
        Get the traffic light state of the signals associated with the given landmark.
        
        Parameters
        ----------
        landmark_id : str
            Landmark ID.
        
        Returns
        -------
        Optional[str]
            Traffic light state, or None if no associated signals found.
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

    def set_state(self, landmark_id: str, state: str) -> bool:
        """
        Update the state of all the signals associated with the given landmark.
        
        Parameters
        ----------
        landmark_id : str
            Landmark ID.
        state : str
            New traffic light state.
        
        Returns
        -------
        bool
            True if successfully updated.
        """
        for tlid, link_index in self.get_all_associated_signals(landmark_id):
            traci.trafficlight.setLinkState(tlid, link_index, state)
        return True

    def switch_off(self) -> None:
        """
        Switch off all traffic lights.
        """
        for tlid, link_index in self.get_all_signals():
            traci.trafficlight.setLinkState(tlid, link_index, SumoSignalState.OFF)
        self._off = True

    def tick(self) -> None:
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


def _get_sumo_net(cfg_file: str) -> Optional[sumolib.net.Net]:
    """
    Get sumo net from configuration file.
    
    This method reads the sumo configuration file and retrieves the sumo net
    filename to create the net.
    
    Parameters
    ----------
    cfg_file : str
        Path to sumo configuration file.
    
    Returns
    -------
    Optional[sumolib.net.Net]
        Sumo net object, or None if net file not found.
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
    
    Parameters
    ----------
    cfg_file : str
        Path to sumo configuration file.
    step_length : float
        Simulation step length.
    host : Optional[str], optional
        Host address for TraCI connection.
    port : Optional[int], optional
        Port number for TraCI connection.
    sumo_gui : bool, optional
        Whether to use sumo-gui, by default False.
    client_order : int, optional
        Client order for TraCI connection, by default 1.
        
    Attributes
    ----------
    net : Optional[sumolib.net.Net]
        Sumo network object.
    spawned_actors : Set[str]
        Set of actor IDs spawned in current step.
    destroyed_actors : Set[str]
        Set of actor IDs destroyed in current step.
    traffic_light_manager : SumoTLManager
        Traffic light manager instance.
    """

    def __init__(
        self,
        cfg_file: str,
        step_length: float,
        host: Optional[str] = None,
        port: Optional[int] = None,
        sumo_gui: bool = False,
        client_order: int = 1,
    ):
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
        self.spawned_actors: Set[str] = set()
        self.destroyed_actors: Set[str] = set()

        # Traffic light manager.
        self.traffic_light_manager = SumoTLManager()

    @property
    def traffic_light_ids(self)-> Set[str]:
        """
        Get all traffic light IDs.
        
        Returns
        -------
        Set[str]
            Set of traffic light landmark IDs.
        """
        return self.traffic_light_manager.get_all_landmarks()

    @staticmethod
    def subscribe(actor_id: str) -> None:
        """
        Subscribe the given actor to receive updates.
        
        Subscribes to the following variables:
        - Type
        - Vehicle class
        - Color
        - Length, Width, Height
        - Position3D (i.e., x, y, z)
        - Angle, Slope
        - Speed
        - Lateral speed
        - Signals
        
        Parameters
        ----------
        actor_id : str
            Actor ID.
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
    def unsubscribe(actor_id: str) -> None:
        """
        Unsubscribe the given actor from receiving updates.
        
        Parameters
        ----------
        actor_id : str
            Actor ID.
        """
        traci.vehicle.unsubscribe(actor_id)

    def get_net_offset(self)-> Tuple[float, float]:
        """
        Get sumo net offset.
        
        Returns
        -------
        Tuple[float, float]
            Net offset as (x, y) coordinates.
        """
        if self.net is None:
            return (0, 0)
        return self.net.getLocationOffset()

    @staticmethod
    def get_actor(actor_id: str) -> SumoActor:
        """
        Get sumo actor information.
        
        Parameters
        ----------
        actor_id : str
            Actor ID.
        
        Returns
        -------
        SumoActor
            Named tuple containing actor information.
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

    def spawn_actor(self, type_id: str, id: str, color: Optional[str]=None) -> Union[str, int]:
        """
        Spawn a new actor.
        
        Parameters
        ----------
        type_id : str
            Vehicle type ID to be spawned.
        id : str
            Actor ID.
        color : Optional[str]
            Color attribute for this specific actor.
        
        Returns
        -------
        str
            Actor ID if successfully spawned, otherwise INVALID_ACTOR_ID.
        """
        try:
            traci.vehicle.add(id, "carla_route", typeID=type_id)
        except traci.exceptions.TraCIException as error:
            logger.error(f"Spawn sumo actor failed: {error}")
            return INVALID_ACTOR_ID #NOTE Incompatible return value type

        if color is not None:
            color = color.split(",") #NOTE Incompatible types in assignment
            traci.vehicle.setColor(id, color)

        self._sequential_id += 1  # noqa: DC05

        return id

    @staticmethod
    def destroy_actor(actor_id: str) -> None:
        """
        Destroy the given actor.
        
        Parameters
        ----------
        actor_id : str
            Actor ID.
        """
        # traci.vehicle.remove(actor_id)

        if actor_id in traci.vehicle.getIDList():
            traci.vehicle.remove(actor_id)
        else:
            logger.warning(f"Tried to remove nonexistent SUMO actor: {actor_id}")

    def get_traffic_light_state(self, landmark_id: str) -> Optional[str]:
        """
        Get traffic light state.
        
        Parameters
        ----------
        landmark_id : str
            Landmark ID.
        
        Returns
        -------
        Optional[str]
            Traffic light state, or None if traffic light does not exist.
        """
        return self.traffic_light_manager.get_state(landmark_id)

    def switch_off_traffic_lights(self) -> None:
        """
        Switch off all traffic lights.
        """
        self.traffic_light_manager.switch_off()

    def synchronize_vehicle(self, vehicle_id: str, transform: carla.Transform, signals: Optional[int]=None) -> bool:
        """
        Update vehicle state.
        
        Parameters
        ----------
        vehicle_id : str
            ID of the actor to be updated.
        transform : carla.Transform
            New vehicle transform (i.e., position and rotation).
        signals : Optional[int], optional
            New vehicle signals.
        
        Returns
        -------
        bool
            True if successfully updated, False otherwise.
        """
        loc_x, loc_y = transform.location.x, transform.location.y
        yaw = transform.rotation.yaw

        traci.vehicle.moveToXY(vehicle_id, "", 0, loc_x, loc_y, angle=yaw, keepRoute=2)
        if signals is not None:
            traci.vehicle.setSignals(vehicle_id, signals)
        return True

    def synchronize_traffic_light(self, landmark_id: str, state: str) -> None:
        """
        Update traffic light state.
        
        Parameters
        ----------
        landmark_id : str
            ID of the traffic light to be updated.
        state : str
            New traffic light state.
        """
        self.traffic_light_manager.set_state(landmark_id, state)

    def tick(self) -> None:
        """
        Tick to sumo simulation.
        """
        traci.simulationStep()
        self.traffic_light_manager.tick()

        # Update data structures for the current frame.
        self.spawned_actors = set(traci.simulation.getDepartedIDList())
        self.destroyed_actors = set(traci.simulation.getArrivedIDList())

    @staticmethod
    def close() -> None:
        """
        Closes traci client.
        """
        traci.close()
