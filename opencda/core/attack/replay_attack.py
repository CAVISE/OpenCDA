"""
Replay-атака для V2X-симуляции в OpenCDA.

Расположение: opencda/core/attack/replay_attack.py

Сценарий: delayed self-replay.
- Атакующий пишет свою траекторию в буфер фиксированного размера.
- Начиная с replay_start, на каждый тик подмешивает в перцепцию жертв
  фантомный "клон" самого себя, идущий с задержкой в buffer_size тиков
  и сдвинутый на lateral_offset метров вбок (соседняя полоса).
- Когда буфер исчерпан и replay-указатель доходит до конца записи,
  клон замораживается на последней позиции.

Архитектура зеркальна GhostSybilAttack:
- from_v2x_config(...) — фабричный метод, вызывается диспетчером
- VehicleManager в каждом тике вызывает attack.tick() и attack.inject()

Буфер делится между всеми экземплярами ReplayAttack через class-level
словарь _shared_buffer: запись принадлежит атакующему (его vid),
проигрывают её все получатели в радиусе.
"""

import math

import carla


class ReplayCloneVehicle:
    """Имитация carla-актора. В отличие от GhostObstacleVehicle:
    - скорость ненулевая (берётся из записанного буфера)
    - id = -2 (чтобы отличать от ghost'ов с id = -1)
    - type_id = vehicle.replay.* (отдельный префикс)
    """

    def __init__(self, transform, velocity, extent_x=2.5, extent_y=1.1, extent_z=0.8, label="clone"):
        self.transform = transform
        self.location = transform.location
        self.velocity = velocity
        self.bounding_box = type("BBox", (), {})()
        self.bounding_box.extent = carla.Vector3D(
            x=extent_x,
            y=extent_y,
            z=extent_z,
        )
        self.carla_id = -2
        self.id = -2
        self.type_id = f"vehicle.replay.{label}"

    def get_location(self):
        return self.location

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        return self.velocity


class ReplayAttack:
    """Delayed self-replay атакующего CAV."""

    REPLAY_TYPE_PREFIX = "vehicle.replay."

    # буфер делится между всеми экземплярами:
    # {attacker_vid: list[frame_dict]}
    _shared_buffer = {}

    def __init__(self, vid, attack_cfg, cav_world, carla_map):
        self.vid = vid
        self.cav_world = cav_world
        self.carla_map = carla_map

        self.enabled = (
            attack_cfg.get("enabled", False)
            and attack_cfg.get("type", "") == "replay"
        )
        self.attacker_vid = attack_cfg.get("attacker_vid", "cav-100")
        self.record_start = int(attack_cfg.get("record_start", 0))
        self.buffer_size = int(attack_cfg.get("buffer_size", 100))
        self.replay_start = int(attack_cfg.get("replay_start", 200))
        self.lateral_offset = float(attack_cfg.get("lateral_offset", 3.5))
        self.visible_to_attacker = bool(attack_cfg.get("visible_to_attacker", False))

        ReplayAttack._shared_buffer.setdefault(self.attacker_vid, [])

        self.local_tick = 0

        print(
            f"[REPLAY INIT] vid={self.vid}, enabled={self.enabled}, "
            f"attacker_vid={self.attacker_vid}, "
            f"record_start={self.record_start}, buffer_size={self.buffer_size}, "
            f"replay_start={self.replay_start}, lateral_offset={self.lateral_offset}"
        )

    @classmethod
    def from_v2x_config(cls, vid, v2x_config, cav_world, carla_map):
        attack_cfg = (v2x_config or {}).get("attack", {}) or {}
        return cls(vid, attack_cfg, cav_world, carla_map)

    @classmethod
    def reset_buffers(cls):
        """Очистить буферы между прогонами симуляции в одном процессе."""
        cls._shared_buffer.clear()

    # -------------------------------------------------------------- public API

    def tick(self):
        self.local_tick += 1
        # запись делает только тот экземпляр, что принадлежит атакующему
        if self.vid == self.attacker_vid:
            self._record_frame()

    def inject(self, objects, ego_pos):
        if not self._should_receive(ego_pos):
            return objects

        if "vehicles" not in objects:
            objects["vehicles"] = []

        # вычистим клонов с прошлого тика (страховка)
        objects["vehicles"] = [
            obj for obj in objects["vehicles"]
            if not str(getattr(obj, "type_id", "")).startswith(self.REPLAY_TYPE_PREFIX)
        ]

        clone = self._build_clone()
        if clone is None:
            print(f"[REPLAY] receiver={self.vid} reason=buffer_empty")
            return objects

        objects["vehicles"].append(clone)
        loc = clone.get_location()
        v = clone.get_velocity()
        speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

        if self.local_tick == self.replay_start:
            print(
                f"[REPLAY TRIGGERED] receiver={self.vid}, "
                f"tick={self.local_tick}, buffer_size={len(self._buffer())}"
            )

        print(
            f"[REPLAY POS] receiver={self.vid}, "
            f"x={loc.x:.2f}, y={loc.y:.2f}, "
            f"yaw={clone.get_transform().rotation.yaw:.2f}, speed={speed:.2f}"
        )
        return objects

    # --------------------------------------------------------------- recording

    def _buffer(self):
        return ReplayAttack._shared_buffer.setdefault(self.attacker_vid, [])

    def _record_frame(self):
        if self.local_tick < self.record_start:
            return
        buf = self._buffer()
        if len(buf) >= self.buffer_size:
            return  # буфер заполнен, останавливаем запись

        attacker_vm = self.cav_world.get_vehicle_managers().get(self.attacker_vid)
        if attacker_vm is None:
            print(
                f"[REPLAY RECORD SKIP] tick={self.local_tick} "
                f"reason=no_attacker_vm attacker_vid={self.attacker_vid} "
                f"known_vids={list(self.cav_world.get_vehicle_managers().keys())}"
            )
            return
        ego_pos = attacker_vm.localizer.get_ego_pos()
        ego_spd = attacker_vm.localizer.get_ego_spd()
        if ego_pos is None:
            print(
                f"[REPLAY RECORD SKIP] tick={self.local_tick} "
                f"reason=localizer_returned_none vid={self.vid}"
            )
            return

        frame = {
            "x": ego_pos.location.x,
            "y": ego_pos.location.y,
            "z": ego_pos.location.z,
            "pitch": ego_pos.rotation.pitch,
            "yaw": ego_pos.rotation.yaw,
            "roll": ego_pos.rotation.roll,
            "speed": ego_spd if ego_spd is not None else 0.0,
        }
        buf.append(frame)

        if len(buf) == 1:
            print(f"[REPLAY RECORD START] tick={self.local_tick} attacker={self.attacker_vid}")
        if len(buf) == self.buffer_size:
            print(
                f"[REPLAY RECORD DONE] tick={self.local_tick} "
                f"attacker={self.attacker_vid} frames={len(buf)}"
            )

    # ----------------------------------------------------------------- replay

    def _should_receive(self, ego_pos):
        if not self.enabled:
            return False
        if self.local_tick < self.replay_start:
            return False
        if self.vid == self.attacker_vid and not self.visible_to_attacker:
            return False

        attacker_vm = self.cav_world.get_vehicle_managers().get(self.attacker_vid)
        if attacker_vm is None:
            print(f"[REPLAY SHOULD_RECEIVE:NO] vid={self.vid} reason=no_attacker_vm")
            return False

        attacker_pos = attacker_vm.v2x_manager.get_ego_pos()
        if attacker_pos is None or ego_pos is None:
            print(
                f"[REPLAY SHOULD_RECEIVE:NO] vid={self.vid} reason=no_positions "
                f"attacker_pos={attacker_pos is not None} ego_pos={ego_pos is not None}"
            )
            return False

        distance = attacker_pos.location.distance(ego_pos.location)
        comm_range = attacker_vm.v2x_manager.communication_range
        if distance > comm_range:
            print(
                f"[REPLAY SHOULD_RECEIVE:NO] vid={self.vid} "
                f"reason=out_of_range distance={distance:.2f} range={comm_range}"
            )
            return False

        print(
            f"[REPLAY SHOULD_RECEIVE:YES] vid={self.vid} "
            f"tick={self.local_tick} buffer_len={len(self._buffer())}"
        )
        return True

    def _build_clone(self):
        buf = self._buffer()
        if not buf:
            return None

        # индекс кадра: сколько тиков прошло с replay_start
        idx = self.local_tick - self.replay_start
        if idx < 0:
            return None
        if idx >= len(buf):
            idx = len(buf) - 1  # frozen на последнем кадре
            frozen = True
        else:
            frozen = False

        frame = buf[idx]
        yaw_rad = math.radians(frame["yaw"])
        # перпендикуляр к направлению движения (положительный offset = влево по ходу)
        offset_x = -math.sin(yaw_rad) * self.lateral_offset
        offset_y = math.cos(yaw_rad) * self.lateral_offset

        clone_tf = carla.Transform(
            carla.Location(
                x=frame["x"] + offset_x,
                y=frame["y"] + offset_y,
                z=frame["z"],
            ),
            carla.Rotation(
                pitch=frame["pitch"],
                yaw=frame["yaw"],
                roll=frame["roll"],
            ),
        )

        if frozen:
            velocity = carla.Vector3D(0.0, 0.0, 0.0)
        else:
            speed = frame["speed"]
            velocity = carla.Vector3D(
                x=math.cos(yaw_rad) * speed,
                y=math.sin(yaw_rad) * speed,
                z=0.0,
            )

        return ReplayCloneVehicle(transform=clone_tf, velocity=velocity, label="clone")
