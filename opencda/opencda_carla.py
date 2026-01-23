"""
Used to reduce the dependency on CARLA api by mocking them in the same structure.
"""


class Vector3D(object):
    """
    Represents a 3D vector and provides useful helper functions.

    Parameters
    ----------
    x : float, optional
        The value of the first axis. Default is 0.
    y : float, optional
        The value of the second axis. Default is 0.
    z : float, optional
        The value of the third axis. Default is 0.

    Attributes
    ----------
    x : float
        The value of the first axis.
    y : float
        The value of the second axis.
    z : float
        The value of the third axis.
    """

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x, self.y, self.z = float(x), float(y), float(z)


class Location(Vector3D):
    """
    Stores a 3D location, and provides useful helper methods.

    Parameters
    ----------
    x : float, optional
        The value of the x-axis. Default is 0.
    y : float, optional
        The value of the y-axis. Default is 0.
    z : float, optional
        The value of the z-axis. Default is 0.

    Attributes
    ----------
    x : float
        The value of the x-axis.
    y : float
        The value of the y-axis.
    z : float
        The value of the z-axis.
    """

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        super(Location, self).__init__(x, y, z)


class Rotation(object):
    """
    Used to represent the rotation of an actor or obstacle.

    Rotations are applied in the order: Roll (X), Pitch (Y), Yaw (Z).
    A 90-degree "Roll" maps the positive Z-axis to the positive Y-axis.
    A 90-degree "Pitch" maps the positive X-axis to the positive Z-axis.
    A 90-degree "Yaw" maps the positive X-axis to the positive Y-axis.

    Parameters
    ----------
    pitch : float, optional
        Rotation about Y-axis. Default is 0.
    yaw : float, optional
        Rotation about Z-axis. Default is 0.
    roll : float, optional
        Rotation about X-axis. Default is 0.

    Attributes
    ----------
    pitch : float
        Rotation about Y-axis.
    yaw : float
        Rotation about Z-axis.
    roll : float
        Rotation about X-axis.
    """

    def __init__(self, pitch: float = 0, yaw: float = 0, roll: float = 0):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


class Transform(object):
    """
    A class that stores the location and rotation of an obstacle.

    It can be created from a simulator transform, defines helper functions
    needed in Pylot, and makes the simulator transform serializable.

    A transform object is instantiated with either a location and a rotation,
    or using a matrix.

    Parameters
    ----------
    location : Location, optional
        The location of the object represented by the transform.
    rotation : Rotation, optional
        The rotation (in degrees) of the object represented by the transform.

    Attributes
    ----------
    location : Location
        The location of the object represented by the transform.
    rotation : Rotation
        The rotation (in degrees) of the object represented by the transform.
    """

    def __init__(self, location: Location | None = None, rotation: Rotation | None = None):
        self.location = location
        self.rotation = Rotation(0, 0, 0) if not rotation else rotation
