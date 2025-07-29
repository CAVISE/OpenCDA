.. class:: CodrivingModelManager(model_name, pretrained, nodes, excluded_nodes)

   Manages the co-driving GNN model, handles trajectory updates and vehicle control in the SUMO simulation.

   :param model_name: Name of the model to load from the CoDriving.models package.
   :type model_name: str
   :param pretrained: Path to the pretrained model weights (.pt or .pth file).
   :type pretrained: str
   :param nodes: List of network nodes (crossroads).
   :type nodes: list
   :param excluded_nodes: List of nodes to exclude from processing.
   :type excluded_nodes: list


.. method:: _load_yaw()

   Loads yaw angle dictionary from a serialized pickle file.

   :return: Dictionary mapping routes to yaw data arrays.
   :rtype: dict


.. method:: _get_nearest_node(pos)

   Returns the closest network node to the given position.

   :param pos: A 2D position vector [x, y].
   :type pos: numpy.ndarray
   :return: The nearest node object from `self.nodes`.
   :rtype: object


.. method:: _import_model()

   Dynamically imports and returns the model class from the CoDriving.models module.

   :return: Imported model class.
   :rtype: type
   :raises ModuleNotFoundError: If the model module cannot be found.


.. method:: make_trajs()

   Main update loop that:

   - Updates vehicle trajectories.
   - Encodes scenario features.
   - Runs the GNN model.
   - Controls vehicle movements based on model predictions.


.. method:: update_trajs()

   Updates the trajectory dictionary for all currently visible vehicles.

   :return: Updated dictionary mapping vehicle IDs to their trajectory history.
   :rtype: dict


.. method:: get_intention_from_vehicle_id(vehicle_id)

   Parses the vehicle ID to infer its intended maneuver (left, right, straight).

   :param vehicle_id: Vehicle identifier in the format `from_to_index`.
   :type vehicle_id: str
   :return: Maneuver intention ('left', 'straight', 'right').
   :rtype: str
   :raises Exception: If the vehicle ID is malformed or unrecognized.


.. method:: encoding_scenario_features()

   Encodes scenario features (position, speed, yaw, intention) for all active vehicles near the intersection.

   :return: Tuple of feature matrix and list of target agent IDs.
   :rtype: tuple[numpy.ndarray, list]


.. staticmethod:: transform_sumo2carla(states)

   In-place transformation from SUMO to CARLA coordinate space.

   Notes:
     - SUMO uses +y to +x for angles; CARLA uses +x to +y.
     - CARLA uses a left-handed coordinate system.

   :param states: State array of shape (N,) or (N, D), where D >= 4.
   :type states: numpy.ndarray
   :raises NotImplementedError: If input shape is invalid.


.. staticmethod:: rotation_matrix_back(yaw)

   Constructs a 2D rotation matrix to rotate vectors back from agent-relative to global coordinates.

   :param yaw: Agent heading angle in radians.
   :type yaw: float
   :return: A 2x2 rotation matrix.
   :rtype: numpy.ndarray


.. staticmethod:: get_yaw(vehicle_id, pos, yaw_dict)

   Retrieves yaw angle based on vehicle position and precomputed yaw dictionary.

   :param vehicle_id: The ID of the vehicle.
   :type vehicle_id: str
   :param pos: Current position of the vehicle.
   :type pos: numpy.ndarray
   :param yaw_dict: Dictionary of yaw values for different routes.
   :type yaw_dict: dict
   :return: Yaw angle in radians.
   :rtype: float


.. staticmethod:: get_intention_vector(intention='straight')

   Converts a maneuver string into a one-hot encoded intention vector.

   :param intention: One of 'left', 'straight', 'right', or 'null'.
   :type intention: str
   :return: One-hot encoded vector of size 3.
   :rtype: numpy.ndarray
   :raises NotImplementedError: If the intention is invalid.
