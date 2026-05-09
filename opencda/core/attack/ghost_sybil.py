"""
Ghost-Sybil атака для V2X-симуляции в OpenCDA.

Расположение в проекте: opencda/core/attack/ghost_sybil.py

Идея:
- VehicleManager не знает деталей атаки.
- Он только создаёт объект через диспетчер (opencda.core.attack.build_attack)
  и в каждом тике вызывает:
    self.attack.tick()
    objects = self.attack.inject(objects, ego_pos)
- Если атака выключена в конфиге, диспетчер возвращает NoAttack(),
  у которого inject() — no-op.

Поведение:
- Перед получателем (любой CAV в радиусе communication_range от атакующего,
  кроме самого атакующего, если visible_to_attacker=False) каждый тик
  начиная с start_tick создаются `count` фантомных ТС, выложенных в линию
  по полосе с шагом ghost_gap, начиная с расстояния ghost_distance.

NB: ghost'ы строятся от waypoint'а ПОЛУЧАТЕЛЯ, а не атакующего, поэтому
это перцепционная инъекция, а не настоящая V2X-Sybil. Для V2X-Sybil менять
надо здесь, в _build_ghosts() — брать позицию от attacker_vm и кэшировать.
"""

import carla


class GhostObstacleVehicle:
    """Имитация carla-актора, чтобы downstream-код принял её за обычное ТС."""

    def __init__(self, transform, extent_x=2.8, extent_y=1.2, extent_z=1.0, label="ghost"):
        self.transform = transform
        self.location = transform.location
        self.velocity = carla.Vector3D(0.0, 0.0, 0.0)
        self.bounding_box = type("BBox", (), {})()
        self.bounding_box.extent = carla.Vector3D(
            x=extent_x,
            y=extent_y,
            z=extent_z,
        )
        self.carla_id = -1
        self.id = -1
        self.type_id = f"vehicle.ghost.{label}"

    def get_location(self):
        return self.location

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        return self.velocity


class GhostSybilAttack:
    """Подмешивает в objects['vehicles'] фейковые ТС перед получателем."""

    GHOST_TYPE_PREFIX = "vehicle.ghost."

    def __init__(self, vid, attack_cfg, cav_world, carla_map):
        self.vid = vid
        self.cav_world = cav_world
        self.carla_map = carla_map

        self.enabled = (
            attack_cfg.get("enabled", False)
            and attack_cfg.get("type", "") == "ghost_sybil"
        )
        self.start_tick = int(attack_cfg.get("start_tick", 100))
        self.attacker_vid = attack_cfg.get("attacker_vid", "cav-100")
        self.visible_to_attacker = bool(attack_cfg.get("visible_to_attacker", False))
        self.count = int(attack_cfg.get("count", 1))
        self.first_distance = float(attack_cfg.get("ghost_distance", 8.0))
        self.gap = float(attack_cfg.get("ghost_gap", 6.0))
        self.freeze_positions = bool(attack_cfg.get("freeze_positions", True))

        self.local_tick = 0
        self._ghost_cache = {}

        print(
            f"[ATTACK INIT] vid={self.vid}, enabled={self.enabled}, "
            f"start_tick={self.start_tick}, attacker_vid={self.attacker_vid}"
        )

    @classmethod
    def from_v2x_config(cls, vid, v2x_config, cav_world, carla_map):
        """Используется диспетчером atak. NoAttack возвращается там."""
        attack_cfg = (v2x_config or {}).get("attack", {}) or {}
        return cls(vid, attack_cfg, cav_world, carla_map)

    # -------------------------------------------------------------- public API

    def tick(self):
        self.local_tick += 1

    def inject(self, objects, ego_pos):
        if not self._should_receive(ego_pos):
            return objects

        if "vehicles" not in objects:
            objects["vehicles"] = []

        # очистить ghost'ов с прошлого тика, если они каким-то образом затесались
        objects["vehicles"] = [
            obj for obj in objects["vehicles"]
            if not str(getattr(obj, "type_id", "")).startswith(self.GHOST_TYPE_PREFIX)
        ]

        # атакующему ничего не показываем
        if self.vid == self.attacker_vid and not self.visible_to_attacker:
            print(f"[GHOST SKIP] receiver={self.vid} reason=attacker_hidden")
            return objects

        ghosts = self._build_ghosts(ego_pos)
        if not ghosts:
            print(f"[GHOST ATTACK FAILED] receiver={self.vid} no ghosts generated")
            return objects

        self._ghost_cache[self.vid] = ghosts

        if self.local_tick == self.start_tick:
            print(
                f"[GHOST ATTACK TRIGGERED] receiver={self.vid}, "
                f"tick={self.local_tick}, ghosts={len(ghosts)}"
            )

        for g in ghosts:
            print(
                f"[GHOST POS] receiver={self.vid}, "
                f"x={g.get_location().x:.2f}, y={g.get_location().y:.2f}, "
                f"yaw={g.get_transform().rotation.yaw:.2f}"
            )

        objects["vehicles"].extend(ghosts)
        print(
            f"[GHOST ATTACK APPLIED] receiver={self.vid}, "
            f"tick={self.local_tick}, vehicles_total={len(objects['vehicles'])}"
        )
        return objects

    # --------------------------------------------------------------- internals

    def _get_attacker_vm(self):
        return self.cav_world.get_vehicle_managers().get(self.attacker_vid)

    def _should_receive(self, ego_pos):
        print(
            f"[SHOULD_RECEIVE] vid={self.vid}, "
            f"attack_enabled={self.enabled}, "
            f"tick={self.local_tick}, start_tick={self.start_tick}, "
            f"attacker_vid={self.attacker_vid}"
        )

        if not self.enabled:
            print(f"[SHOULD_RECEIVE:NO] vid={self.vid} reason=attack_disabled")
            return False

        if self.local_tick < self.start_tick:
            print(f"[SHOULD_RECEIVE:NO] vid={self.vid} reason=before_start_tick")
            return False

        if self.vid == self.attacker_vid and not self.visible_to_attacker:
            print(f"[SHOULD_RECEIVE:NO] vid={self.vid} reason=attacker_hidden")
            return False

        attacker_vm = self._get_attacker_vm()
        if attacker_vm is None:
            print(f"[SHOULD_RECEIVE:NO] vid={self.vid} reason=no_attacker_vm")
            return False

        attacker_pos = attacker_vm.v2x_manager.get_ego_pos()
        if attacker_pos is None or ego_pos is None:
            print(f"[SHOULD_RECEIVE:NO] vid={self.vid} reason=no_positions")
            return False

        distance = attacker_pos.location.distance(ego_pos.location)
        comm_range = attacker_vm.v2x_manager.communication_range
        print(
            f"[SHOULD_RECEIVE:DIST] vid={self.vid}, "
            f"distance={distance:.2f}, range={comm_range}"
        )

        if distance > comm_range:
            print(f"[SHOULD_RECEIVE:NO] vid={self.vid} reason=out_of_range")
            return False

        print(f"[SHOULD_RECEIVE:YES] vid={self.vid}")
        return True

    def _build_ghosts(self, ego_pos):
        ego_wp = self.carla_map.get_waypoint(
            ego_pos.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        ghosts = []
        for i in range(self.count):
            distance_ahead = self.first_distance + i * self.gap
            next_wps = ego_wp.next(distance_ahead)
            if not next_wps:
                continue

            ghost_tf = next_wps[0].transform
            ghost = GhostObstacleVehicle(
                transform=carla.Transform(
                    carla.Location(
                        x=ghost_tf.location.x,
                        y=ghost_tf.location.y,
                        z=ghost_tf.location.z + 0.2,
                    ),
                    carla.Rotation(
                        pitch=ghost_tf.rotation.pitch,
                        yaw=ghost_tf.rotation.yaw,
                        roll=ghost_tf.rotation.roll,
                    ),
                ),
                label=f"sybil_{i + 1}",
            )
            ghosts.append(ghost)

        return ghosts
