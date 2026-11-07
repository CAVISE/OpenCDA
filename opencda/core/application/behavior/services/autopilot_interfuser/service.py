"""InterFuser-backed autopilot behavior service."""

from __future__ import annotations

import logging
import math
import os
import sys
import weakref
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from opencda.core.application.behavior.capability import Capability, CapabilityBindings
from opencda.core.application.behavior.registry import BehaviorServiceRegistry
from opencda.core.application.behavior.transport_message import TransportMessage
from opencda.core.application.behavior.types import Location
from opencda.core.application.behavior.services.movement_controller import MovementControllerRequestMessage
from opencda.core.plan.local_planner_behavior import RoadOption

from .types import AutopilotInterfuserState

if TYPE_CHECKING:
    from opencda.core.common.vehicle_manager import VehicleManager


logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.autopilot_interfuser")

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Resize2FixedSize:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        return pil_img.resize(self.size)


@BehaviorServiceRegistry.register
class AutopilotInterfuser:
    """Run InterFuser inference and feed the nearest predicted waypoint to MovementController."""

    service_type: str = "autopilot_interfuser"
    priority: int = 20

    @property
    def capability_bindings(self) -> CapabilityBindings:
        return {
            Capability.COMMAND_SUBMIT: self.process,
            Capability.STATE_OBSERVE: self.get_state,
        }

    def __init__(
        self,
        priority: int = 20,
        model: str = "interfuser_baseline",
        model_path: str | None = None,
        interfuser_root: str | None = None,
        device: str = "auto",
        debug: bool = False,
        fail_on_model_error: bool = False,
        front_camera_index: int = 0,
        left_camera_index: int = 1,
        right_camera_index: int = 2,
        route_waypoint_index: int = 1,
        predicted_waypoint_index: int = 1,
        waypoint_interval: float = 0.5,
        max_target_speed_kmh: float = 18.0,
        min_target_speed_kmh: float = 0.0,
    ) -> None:
        self.priority = priority
        self.model_name = model
        self.model_path = model_path
        self.interfuser_root = interfuser_root
        self.device_name = device
        self.debug = debug
        self.fail_on_model_error = fail_on_model_error
        self.front_camera_index = front_camera_index
        self.left_camera_index = left_camera_index
        self.right_camera_index = right_camera_index
        self.route_waypoint_index = route_waypoint_index
        self.predicted_waypoint_index = predicted_waypoint_index
        self.waypoint_interval = waypoint_interval
        self.max_target_speed_kmh = max_target_speed_kmh
        self.min_target_speed_kmh = min_target_speed_kmh

        self._owner_ref: weakref.ReferenceType[VehicleManager] | None = None
        self._net: Any | None = None
        self._device: Any | None = None
        self._rgb_front_transform: Any | None = None
        self._rgb_left_transform: Any | None = None
        self._rgb_right_transform: Any | None = None
        self._rgb_center_transform: Any | None = None
        self._model_load_attempted = False

        self._step = 0
        self._last_target_location: Location | None = None
        self._last_target_speed: float | None = None
        self._last_error: str | None = None

    def _get_owner(self) -> VehicleManager:
        owner_ref = self._owner_ref
        if owner_ref is None:
            raise RuntimeError("InterFuser autopilot is not attached to an owner.")

        owner = owner_ref()
        if owner is None:
            raise RuntimeError("InterFuser autopilot owner is no longer available.")

        return owner

    def on_attach(self, owner: VehicleManager) -> None:
        self._owner_ref = weakref.ref(owner)
        self._try_load_model()

    def on_detach(self) -> None:
        self._owner_ref = None
        self._net = None
        self._device = None
        self._model_load_attempted = False
        self._last_target_location = None
        self._last_target_speed = None

    def get_state(self) -> AutopilotInterfuserState:
        owner_id = None
        if self._owner_ref is not None and (owner := self._owner_ref()) is not None:
            owner_id = owner.id

        return AutopilotInterfuserState(
            service_type=self.service_type,
            owner_id=owner_id,
            model_loaded=self._net is not None,
            step=self._step,
            last_target_location=self._last_target_location,
            last_target_speed=self._last_target_speed,
            last_error=self._last_error,
        )

    def _try_load_model(self) -> None:
        if self._net is not None:
            return
        if self._model_load_attempted:
            return
        self._model_load_attempted = True

        if not self.model_path:
            self._last_error = "model_path is not configured; autopilot_interfuser is inactive."
            logger.warning(self._last_error)
            return

        try:
            self._load_model()
            self._last_error = None
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("Failed to load InterFuser model.")
            if self.fail_on_model_error:
                raise

    def _load_model(self) -> None:
        import torch

        package_root = self._resolve_interfuser_package_root()
        self._ensure_interfuser_runtime(package_root)

        from timm.models import create_model

        checkpoint_path = self._resolve_existing_path(self.model_path, package_root)
        device = self._resolve_device(torch)

        net = create_model(self.model_name)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()

        self._net = net
        self._device = device
        self._rgb_front_transform = self._create_carla_rgb_transform(224)
        self._rgb_left_transform = self._create_carla_rgb_transform(128)
        self._rgb_right_transform = self._create_carla_rgb_transform(128)
        self._rgb_center_transform = self._create_carla_rgb_transform(128, need_scale=False)
        logger.info("Loaded InterFuser model '%s' from '%s'.", self.model_name, checkpoint_path)

    def _resolve_device(self, torch_module: Any) -> Any:
        if self.device_name == "auto":
            device_name = "cuda" if torch_module.cuda.is_available() else "cpu"
        else:
            device_name = self.device_name
        if device_name.startswith("cuda") and not torch_module.cuda.is_available():
            logger.warning("CUDA requested for InterFuser but is not available; falling back to CPU.")
            device_name = "cpu"
        return torch_module.device(device_name)

    def _resolve_repo_root(self) -> Path:
        for parent in Path(__file__).resolve().parents:
            if (parent / "opencda").is_dir() and (parent / "requirements.txt").is_file():
                return parent
        return Path.cwd()

    def _resolve_interfuser_package_root(self) -> Path:
        candidates: list[Path] = []
        if self.interfuser_root:
            candidates.append(Path(self.interfuser_root).expanduser())
        if env_root := os.environ.get("INTERFUSER_ROOT"):
            candidates.append(Path(env_root).expanduser())

        opencda_root = self._resolve_repo_root()
        candidates.extend(
            [
                opencda_root / "InterFuser",
                opencda_root / "interfuser",
                opencda_root.parent / "InterFuser",
            ]
        )

        for candidate in candidates:
            package_root = candidate / "interfuser" if (candidate / "interfuser" / "timm").is_dir() else candidate
            if (package_root / "timm" / "models" / "interfuser.py").is_file():
                return package_root.resolve()

        searched = ", ".join(str(path) for path in candidates)
        raise RuntimeError(f"Could not find InterFuser package root. Searched: {searched}")

    def _resolve_existing_path(self, path_value: str, package_root: Path) -> Path:
        path = Path(path_value).expanduser()
        candidates = [path] if path.is_absolute() else []
        candidates.extend(
            [
                Path.cwd() / path,
                self._resolve_repo_root() / path,
                package_root.parent / path,
                package_root / path,
            ]
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"InterFuser checkpoint not found: {path_value}")

    def _ensure_interfuser_runtime(self, package_root: Path) -> None:
        existing_timm = sys.modules.get("timm")
        if existing_timm is not None:
            existing_file = Path(getattr(existing_timm, "__file__", "")).resolve()
            if package_root not in existing_file.parents:
                raise RuntimeError(
                    "A different 'timm' package is already imported. Start OpenCDA with InterFuser loaded first, "
                    f"or run this service in a fresh process. Existing timm: {existing_file}"
                )

        package_root_str = str(package_root)
        if package_root_str not in sys.path:
            sys.path.insert(0, package_root_str)

    def _create_carla_rgb_transform(
        self,
        input_size: int | tuple[int, int] | tuple[int, int, int],
        need_scale: bool = True,
        mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
        std: tuple[float, float, float] = IMAGENET_DEFAULT_STD,
    ) -> Any:
        from torchvision import transforms

        img_size = input_size[-2:] if isinstance(input_size, tuple) else input_size
        input_size_num = input_size[-1] if isinstance(input_size, tuple) else input_size
        tfl: list[Any] = []

        if need_scale:
            if input_size_num == 112:
                tfl.append(Resize2FixedSize((170, 128)))
            elif input_size_num == 128:
                tfl.append(Resize2FixedSize((195, 146)))
            elif input_size_num == 224:
                tfl.append(Resize2FixedSize((341, 256)))
            elif input_size_num == 256:
                tfl.append(Resize2FixedSize((288, 288)))
            else:
                raise ValueError(f"Can't find proper crop size for {input_size_num}.")

        tfl.append(transforms.CenterCrop(img_size))
        tfl.append(transforms.ToTensor())
        tfl.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(tfl)

    def _get_camera_rgb(self, owner: VehicleManager, index: int) -> np.ndarray | None:
        cameras = owner.perception_manager.rgb_camera
        if cameras is None or index < 0 or index >= len(cameras):
            return None
        image = cameras[index].image
        if image is None:
            return None
        return cv2.cvtColor(np.asarray(image[:, :, :3]), cv2.COLOR_BGR2RGB)

    def _get_lidar_features(self, owner: VehicleManager) -> np.ndarray | None:
        lidar = owner.perception_manager.lidar
        if lidar is None or lidar.data is None:
            return None

        lidar_unprocessed = np.asarray(lidar.data[:, :3], dtype=np.float32).copy()
        lidar_unprocessed[:, 1] *= -1
        return self._lidar_to_histogram_features(lidar_unprocessed)

    def _lidar_to_histogram_features(self, lidar: np.ndarray) -> np.ndarray:
        def splat_points(point_cloud: np.ndarray) -> np.ndarray:
            pixels_per_meter = 8
            hist_max_per_pixel = 5
            x_meters_max = 14
            y_meters_max = 28
            xbins = np.linspace(-2 * x_meters_max, 2 * x_meters_max + 1, 2 * x_meters_max * pixels_per_meter + 1)
            ybins = np.linspace(-y_meters_max, 0, y_meters_max * pixels_per_meter + 1)
            hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
            hist[hist > hist_max_per_pixel] = hist_max_per_pixel
            return hist / hist_max_per_pixel

        below = lidar[lidar[..., 2] <= -2.0]
        above = lidar[lidar[..., 2] > -2.0]
        below_features = splat_points(below)
        above_features = splat_points(above)
        total_features = below_features + above_features
        features = np.stack([below_features, above_features, total_features], axis=-1)
        return np.transpose(features, (2, 0, 1)).astype(np.float32)

    def _build_model_inputs(self, owner: VehicleManager) -> dict[str, Any] | None:
        import torch

        if self._device is None:
            return None

        rgb = self._get_camera_rgb(owner, self.front_camera_index)
        rgb_left = self._get_camera_rgb(owner, self.left_camera_index)
        rgb_right = self._get_camera_rgb(owner, self.right_camera_index)
        lidar = self._get_lidar_features(owner)

        if rgb is None or rgb_left is None or rgb_right is None or lidar is None:
            self._last_error = "Waiting for InterFuser camera/LiDAR frames."
            return None

        transform = self._get_ego_transform(owner)
        command, target_point = self._get_route_context(owner, transform)
        velocity = self._get_speed_mps(owner)

        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd = max(1, min(6, command)) - 1
        cmd_one_hot[cmd] = 1
        cmd_one_hot.append(velocity)

        return {
            "rgb": self._image_to_tensor(rgb, self._rgb_front_transform),
            "rgb_left": self._image_to_tensor(rgb_left, self._rgb_left_transform),
            "rgb_right": self._image_to_tensor(rgb_right, self._rgb_right_transform),
            "rgb_center": self._image_to_tensor(rgb, self._rgb_center_transform),
            "measurements": torch.from_numpy(np.asarray(cmd_one_hot, dtype=np.float32)).unsqueeze(0).to(self._device),
            "target_point": torch.from_numpy(target_point.astype(np.float32)).view(1, -1).to(self._device),
            "lidar": torch.from_numpy(lidar).float().unsqueeze(0).to(self._device),
        }

    def _image_to_tensor(self, rgb_image: np.ndarray, transform: Any) -> Any:
        if transform is None:
            raise RuntimeError("InterFuser image transforms are not initialized.")
        return transform(Image.fromarray(rgb_image)).unsqueeze(0).to(self._device).float()

    def _get_ego_transform(self, owner: VehicleManager) -> Any:
        ego_pos = owner.localizer.get_ego_pos()
        return ego_pos if ego_pos is not None else owner.vehicle.get_transform()

    def _get_speed_mps(self, owner: VehicleManager) -> float:
        velocity = owner.vehicle.get_velocity()
        return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    def _get_route_context(self, owner: VehicleManager, ego_transform: Any) -> tuple[int, np.ndarray]:
        waypoint_buffer = owner.agent.get_local_planner().get_waypoint_buffer()
        if not waypoint_buffer:
            return RoadOption.LANEFOLLOW.value, np.asarray([0.0, -10.0], dtype=np.float32)

        command = getattr(waypoint_buffer[0][1], "value", RoadOption.LANEFOLLOW.value)
        route_index = min(max(self.route_waypoint_index, 0), len(waypoint_buffer) - 1)
        waypoint = waypoint_buffer[route_index][0]
        target_location = waypoint.transform.location if hasattr(waypoint, "transform") else waypoint.location
        return int(command), self._world_delta_to_interfuser_local(ego_transform, target_location)

    def _world_delta_to_interfuser_local(self, ego_transform: Any, target_location: Any) -> np.ndarray:
        delta = np.asarray(
            [
                target_location.x - ego_transform.location.x,
                target_location.y - ego_transform.location.y,
            ],
            dtype=np.float32,
        )
        rotation = self._interfuser_rotation(ego_transform.rotation.yaw)
        return rotation.T.dot(delta).astype(np.float32)

    def _interfuser_local_to_world(self, ego_transform: Any, local_point: np.ndarray) -> Location:
        rotation = self._interfuser_rotation(ego_transform.rotation.yaw)
        world_delta = rotation.dot(local_point.astype(np.float32))
        return Location(
            x=float(ego_transform.location.x + world_delta[0]),
            y=float(ego_transform.location.y + world_delta[1]),
            z=float(ego_transform.location.z),
        )

    def _interfuser_rotation(self, yaw_degrees: float) -> np.ndarray:
        theta = math.radians(yaw_degrees) + math.pi / 2.0
        return np.asarray(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ],
            dtype=np.float32,
        )

    def _run_inference(self, model_inputs: dict[str, Any]) -> np.ndarray:
        import torch

        if self._net is None:
            raise RuntimeError("InterFuser model is not loaded.")

        with torch.no_grad():
            output = self._net(model_inputs)

        waypoints = output[1] if isinstance(output, tuple) else output
        return waypoints.detach().cpu().numpy()[0].astype(np.float32)

    def _build_movement_command(self, owner: VehicleManager, pred_waypoints: np.ndarray) -> TransportMessage[MovementControllerRequestMessage]:
        ego_transform = self._get_ego_transform(owner)
        waypoint_index = min(max(self.predicted_waypoint_index, 0), pred_waypoints.shape[0] - 1)
        local_target = pred_waypoints[waypoint_index]
        target_location = self._interfuser_local_to_world(ego_transform, local_target)
        target_speed = self._estimate_target_speed_kmh(local_target, waypoint_index)

        self._last_target_location = target_location
        self._last_target_speed = target_speed

        if self.debug:
            self._draw_target(owner, target_location)

        return TransportMessage(
            src_owner_id=owner.id,
            src_service_type=self.service_type,
            dst_owner_id=owner.id,
            dst_service_type="movement_controller",
            payload=MovementControllerRequestMessage(target_location=target_location, target_speed=target_speed),
        )

    def _estimate_target_speed_kmh(self, local_target: np.ndarray, waypoint_index: int) -> float:
        horizon = max((waypoint_index + 1) * self.waypoint_interval, self.waypoint_interval)
        speed_kmh = float(np.linalg.norm(local_target) / horizon * 3.6)
        return float(np.clip(speed_kmh, self.min_target_speed_kmh, self.max_target_speed_kmh))

    def _draw_target(self, owner: VehicleManager, target_location: Location) -> None:
        try:
            import carla

            owner.vehicle.get_world().debug.draw_point(
                carla.Location(target_location.x, target_location.y, target_location.z + 0.5),
                size=0.12,
                color=carla.Color(0, 255, 255),
                life_time=0.1,
            )
        except Exception:
            logger.debug("Failed to draw InterFuser target.", exc_info=True)

    def process(self, messages: Sequence[TransportMessage[Any]]) -> tuple[TransportMessage[MovementControllerRequestMessage], ...]:
        del messages
        self._step += 1

        if self._net is None:
            self._try_load_model()
        if self._net is None:
            return ()

        owner = self._get_owner()
        model_inputs = self._build_model_inputs(owner)
        if model_inputs is None:
            return ()

        try:
            pred_waypoints = self._run_inference(model_inputs)
            self._last_error = None
            return (self._build_movement_command(owner, pred_waypoints),)
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("InterFuser inference failed.")
            if self.fail_on_model_error:
                raise
            return ()
