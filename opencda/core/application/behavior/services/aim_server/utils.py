import numpy as np
import math

from .models import Location, Rotation, Transform


def rotation_matrix_back(yaw):
    """
    Rotate back.
    https://en.wikipedia.org/wiki/Rotation_matrix#Non-standard_orientation_of_the_coordinate_system
    """
    rotation = np.array([[np.cos(-np.pi / 2 + yaw), -np.sin(-np.pi / 2 + yaw)], [np.sin(-np.pi / 2 + yaw), np.cos(-np.pi / 2 + yaw)]])
    return rotation


def get_intention_vector(intention: str = "straight") -> np.ndarray:
    """
    Return a 3-bit one-hot format intention vector.
    """
    intention_feature = np.zeros(3)
    if intention == "left":
        intention_feature[0] = 1
    elif intention == "straight":
        intention_feature[1] = 1
    elif intention == "right":
        intention_feature[2] = 1
    elif intention == "null":
        pass  # return zero array
    else:
        raise NotImplementedError
    return intention_feature


def get_end(start, intention):
    """
    Determines the direction of exit from the turn given its start and vehicle intention.

    :param start: direction from which CAV enters the intersection
    :param intention: direction of turning relative to CAV movement
    :return: end
    """
    match intention:
        case "right":
            match start:
                case "up":
                    return "left"
                case "right":
                    return "up"
                case "down":
                    return "right"
                case "left":
                    return "down"
        case "left":
            match start:
                case "up":
                    return "right"
                case "right":
                    return "down"
                case "down":
                    return "left"
                case "left":
                    return "up"
        case "straight":
            match start:
                case "up":
                    return "down"
                case "right":
                    return "left"
                case "down":
                    return "up"
                case "left":
                    return "right"


def get_carla_transform(in_sumo_transform, extent):
    """
    Returns carla transform based on sumo transform.
    """
    in_location = in_sumo_transform.location
    in_rotation = in_sumo_transform.rotation

    # From front-center-bumper to center (sumo reference system).
    # (http://sumo.sourceforge.net/userdoc/Purgatory/Vehicle_Values.html#angle)
    yaw = -1 * in_rotation.yaw + 90
    pitch = in_rotation.pitch
    out_location = (
        in_location.x - math.cos(math.radians(yaw)) * extent.x,
        in_location.y - math.sin(math.radians(yaw)) * extent.x,
        in_location.z - math.sin(math.radians(pitch)) * extent.x,
    )
    out_rotation = (in_rotation.pitch, in_rotation.yaw, in_rotation.roll)

    # Transform to carla reference system (left-handed system).
    out_transform = Transform(
        Location(out_location[0], -out_location[1], out_location[2]), Rotation(out_rotation[0], out_rotation[1] - 90, out_rotation[2])
    )

    return out_transform


def get_sumo_transform(in_carla_transform, extent):
    """
    Returns sumo transform based on carla transform.
    """
    in_location = in_carla_transform.location
    in_rotation = in_carla_transform.rotation

    # From center to front-center-bumper (carla reference system).
    yaw = -1 * in_rotation.yaw
    pitch = in_rotation.pitch
    out_location = (
        in_location.x + math.cos(math.radians(yaw)) * extent.x,
        in_location.y - math.sin(math.radians(yaw)) * extent.x,
        in_location.z - math.sin(math.radians(pitch)) * extent.x,
    )
    out_rotation = (in_rotation.pitch, in_rotation.yaw, in_rotation.roll)

    # Transform to sumo reference system.
    out_transform = Transform(
        Location(out_location[0], -out_location[1], out_location[2]), Rotation(out_rotation[0], out_rotation[1] + 90, out_rotation[2])
    )

    return out_transform
