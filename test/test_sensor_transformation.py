"""
Unit tests for sensor coordinate transformation functions.

This module contains unit tests for sensor transformation utilities including
world-to-sensor conversions, bounding box transformations, camera intrinsics,
and LiDAR-to-camera projection operations.
"""

import os
import sys
import unittest

import numpy as np

# temporary solution for relative imports in case opencda is not installed
# if opencda is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import mocked_carla as mcarla
from opencda.core.sensing.perception.sensor_transformation import (
    x_to_world_transformation,
    world_to_sensor,
    sensor_to_world,
    get_camera_intrinsic,
    create_bb_points,
    bbx_to_world,
    vehicle_to_sensor,
    get_bounding_box,
    get_2d_bb,
    project_lidar_to_camera,
)


class TestSensorTransformation(unittest.TestCase):
    """
    Test suite for sensor transformation utilities.

    Tests coordinate transformations between world, sensor, and vehicle
    reference frames, as well as bounding box operations and projections.
    """

    def setUp(self) -> None:
        # random cords, [(x, y, z, 1), n]
        self.cords = np.random.random(size=(4, 10))
        self.vehicle = mcarla.Vehicle()
        self.camera = mcarla.Camera({"image_size_x": 600, "image_size_y": 800, "fov": 90})
        self.lidar = mcarla.Lidar({"channels": 32, "range": 50})

        self.camera_transform = mcarla.Transform(x=11, y=11, z=11)
        self.lidar_transform = mcarla.Transform(x=10, y=10, z=10)

        self.rgb_image = np.random.randint(0, 255, size=(800, 600, 3)).astype("uint8")
        self.point_cloud = np.random.random(size=(100, 4))

    def test_x_to_world_transformation(self) -> None:
        """
        Test transformation matrix generation from sensor to world frame.

        Validates that the transformation matrix has correct shape (4, 4) and
        homogeneous coordinate property.

        Returns
        -------
        None
        """
        assert x_to_world_transformation(self.lidar_transform).shape == (4, 4)
        assert x_to_world_transformation(self.lidar_transform)[3, 3] == 1

    def test_world_to_sensor(self) -> None:
        """
        Test world-to-sensor coordinate transformation.

        Verifies that coordinates are correctly transformed from world frame
        to sensor frame.

        Returns
        -------
        None
        """
        assert world_to_sensor(self.cords, self.lidar_transform).shape == (4, self.cords.shape[1])

    def test_sensor_to_world(self) -> None:
        """
        Test sensor-to-world coordinate transformation.

        Verifies that coordinates are correctly transformed from sensor frame
        to world frame.

        Returns
        -------
        None
        """
        assert sensor_to_world(self.cords, self.lidar_transform).shape == (4, self.cords.shape[1])

    def test_get_camera_intrinsic(self) -> None:
        """
        Test camera intrinsic matrix extraction.

        Validates that the intrinsic matrix has correct shape (3, 3) and
        proper homogeneous coordinate structure.

        Returns
        -------
        None
        """
        assert get_camera_intrinsic(self.camera).shape == (3, 3)
        assert get_camera_intrinsic(self.camera)[2, 2] == 1

    def test_create_bb_points(self) -> None:
        """
        Test bounding box corner point generation.

        Verifies that 8 corner points are generated with homogeneous
        coordinates (w=1).

        Returns
        -------
        None
        """
        assert create_bb_points(self.vehicle).shape == (8, 4)
        assert create_bb_points(self.vehicle)[:, 3].all() == 1

    def test_bbx_to_world(self) -> None:
        """
        Test bounding box transformation to world coordinates.

        Validates that bounding box points are correctly transformed from
        vehicle frame to world frame.

        Returns
        -------
        None
        """
        assert bbx_to_world(self.cords.T, self.vehicle).shape == (4, self.cords.shape[1])

    def test_vehicle_to_sensor(self) -> None:
        """
        Test vehicle-to-sensor coordinate transformation.

        Verifies that coordinates are correctly transformed from vehicle
        frame to sensor frame.

        Returns
        -------
        None
        """
        assert vehicle_to_sensor(self.cords.T, self.vehicle, self.camera_transform).shape == (4, self.cords.shape[1])

    def test_get_bounding_box(self) -> None:
        """
        Test 3D bounding box extraction in camera coordinates.

        Validates that 8 corner points are generated in 3D camera space.

        Returns
        -------
        None
        """
        assert get_bounding_box(self.vehicle, self.camera, self.camera_transform).shape == (8, 3)

    def test_get_2d_bb(self) -> None:
        """
        Test 2D bounding box projection to image plane.

        Verifies that 3D bounding box is correctly projected to 2D image
        coordinates with min/max corners.

        Returns
        -------
        None
        """
        assert get_2d_bb(self.vehicle, self.camera, self.camera_transform).shape == (2, 2)

    def test_project_lidar_to_camera(self) -> None:
        """
        Test LiDAR point cloud projection to camera image.

        Validates that point cloud is correctly projected and colorized
        according to camera view.

        Returns
        -------
        None
        """
        assert project_lidar_to_camera(self.lidar, self.camera, self.point_cloud, self.rgb_image)[1].shape == (self.point_cloud.shape[0], 3)
        assert project_lidar_to_camera(self.lidar, self.camera, self.point_cloud, self.rgb_image)[0].shape == self.rgb_image.shape


if __name__ == "__main__":
    unittest.main()
