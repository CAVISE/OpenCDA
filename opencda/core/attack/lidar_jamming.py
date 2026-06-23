import math


class LidarJammingAttack:
    """
    LiDAR jamming emulation at perception-output level.

    Effect:
    - removes selected perceived vehicles from objects["vehicles"]
    - can act only on receivers in V2X range of attacker
    - attacker itself can be excluded from jamming
    """

    def __init__(self, vehicle_manager, v2x_config):
        self.vm = vehicle_manager

        attack_cfg = v2x_config.get("attack", {})
        self.enabled = (
            attack_cfg.get("enabled", False)
            and attack_cfg.get("type", "") == "lidar_jamming"
        )

        self.start_tick = int(attack_cfg.get("start_tick", 100))
        self.attacker_vid = attack_cfg.get("attacker_vid", "cav-100")
        self.visible_to_attacker = bool(
            attack_cfg.get("visible_to_attacker", False)
        )

        # jamming geometry
        self.mode = attack_cfg.get("mode", "front_cone")   # front_cone | all | range_only
        self.range_m = float(attack_cfg.get("range_m", 25.0))
        self.fov_deg = float(attack_cfg.get("fov_deg", 60.0))

        # drop policy
        self.drop_all = bool(attack_cfg.get("drop_all", True))
        self.max_drop = int(attack_cfg.get("max_drop", 999))
        self.drop_traffic_lights = bool(attack_cfg.get("drop_traffic_lights", False))

        self.local_tick = 0

        print(
            f"[LIDAR JAM INIT] vid={self.vm.vid}, enabled={self.enabled}, "
            f"attacker_vid={self.attacker_vid}, start_tick={self.start_tick}, "
            f"mode={self.mode}, range={self.range_m}, fov={self.fov_deg}"
        )

    def is_active_now(self):
        return self.enabled and self.local_tick >= self.start_tick

    def _get_attacker_vm(self):
        vm_dict = self.vm.cav_world.get_vehicle_managers()
        return vm_dict.get(self.attacker_vid)

    def _should_jam(self):
        print(
            f"[LIDAR JAM SHOULD] vid={self.vm.vid}, "
            f"enabled={self.enabled}, tick={self.local_tick}, "
            f"start_tick={self.start_tick}, attacker_vid={self.attacker_vid}"
        )

        if not self.enabled:
            print(f"[LIDAR JAM:NO] vid={self.vm.vid} reason=attack_disabled")
            return False

        if self.local_tick < self.start_tick:
            print(f"[LIDAR JAM:NO] vid={self.vm.vid} reason=before_start_tick")
            return False

        if self.vm.vid == self.attacker_vid and not self.visible_to_attacker:
            print(f"[LIDAR JAM:NO] vid={self.vm.vid} reason=attacker_hidden")
            return False

        attacker_vm = self._get_attacker_vm()
        if attacker_vm is None:
            print(f"[LIDAR JAM:NO] vid={self.vm.vid} reason=no_attacker_vm")
            return False

        attacker_pos = attacker_vm.v2x_manager.get_ego_pos()
        my_pos = self.vm.localizer.get_ego_pos()

        if attacker_pos is None or my_pos is None:
            print(f"[LIDAR JAM:NO] vid={self.vm.vid} reason=no_positions")
            return False

        distance = attacker_pos.location.distance(my_pos.location)
        print(
            f"[LIDAR JAM:DIST] vid={self.vm.vid}, "
            f"distance={distance:.2f}, range={attacker_vm.v2x_manager.communication_range}"
        )

        if distance > attacker_vm.v2x_manager.communication_range:
            print(f"[LIDAR JAM:NO] vid={self.vm.vid} reason=out_of_range")
            return False

        print(f"[LIDAR JAM:YES] vid={self.vm.vid}")
        return True

    def _vehicle_in_jam_region(self, ego_pos, obj):
        """
        Decide whether a perceived object is affected by jamming.
        """
        if not hasattr(obj, "get_location"):
            return False
        def _vehicle_in_jam_region(self, ego_pos, obj):
            # Для dict-объектов из PerceptionManager
            if isinstance(obj, dict):
                loc = obj.get("location") or obj.get("carla_location")
                if loc is None:
                    return False
                dx = loc.x - ego_pos.location.x
                dy = loc.y - ego_pos.location.y
            elif hasattr(obj, "get_location"):
                obj_loc = obj.get_location()
                dx = obj_loc.x - ego_pos.location.x
                dy = obj_loc.y - ego_pos.location.y
            else:
                return False
        obj_loc = obj.get_location()
        dx = obj_loc.x - ego_pos.location.x
        dy = obj_loc.y - ego_pos.location.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist > self.range_m:
            return False

        if self.mode == "all":
            return True

        if self.mode == "range_only":
            return True

        # front_cone
        ego_yaw_rad = math.radians(ego_pos.rotation.yaw)
        forward_x = math.cos(ego_yaw_rad)
        forward_y = math.sin(ego_yaw_rad)

        if dist < 1e-6:
            return True

        dir_x = dx / dist
        dir_y = dy / dist
        dot = max(-1.0, min(1.0, forward_x * dir_x + forward_y * dir_y))
        angle_deg = math.degrees(math.acos(dot))

        return angle_deg <= (self.fov_deg / 2.0)

    def inject(self, objects, ego_pos):
        self.local_tick += 1

        if not self._should_jam():
            return objects

        if "vehicles" not in objects:
            objects["vehicles"] = []

        original = objects["vehicles"]
        kept = []
        dropped = []

        for obj in original:
            if self._vehicle_in_jam_region(ego_pos, obj):
                if self.drop_all or len(dropped) < self.max_drop:
                    dropped.append(obj)
                    continue
            kept.append(obj)

        objects["vehicles"] = kept

        if self.drop_traffic_lights and "traffic_lights" in objects:
            objects["traffic_lights"] = []

        print(
            f"[LIDAR JAM APPLIED] receiver={self.vm.vid}, tick={self.local_tick}, "
            f"dropped={len(dropped)}, kept={len(kept)}"
        )

        for obj in dropped[:5]:
            loc = obj.get_location()
            print(
                f"[LIDAR JAM DROP] receiver={self.vm.vid}, "
                f"x={loc.x:.2f}, y={loc.y:.2f}"
            )

        return objects