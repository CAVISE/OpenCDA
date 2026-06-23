# LiDAR Jamming Attack

## Overview

The LiDAR jamming attack emulates sensor interference at the perception-output level. Instead of modifying raw LiDAR point clouds, it removes selected perceived objects from the receiver's `objects["vehicles"]` list.

The goal is to make a victim vehicle fail to perceive vehicles located in a configurable jamming region. The attack can operate on all nearby vehicles, vehicles within a distance range, or vehicles inside a forward cone defined by range and field of view.

## How the Attack Works

On each call to `inject()`, the attack increments its local tick counter and checks whether jamming should be applied. Jamming is applied only if the attack is enabled, the current tick is greater than or equal to `start_tick`, the attacker exists, and the receiver is within the attacker's V2X communication range.

If the receiver should be jammed, the attack iterates over `objects["vehicles"]`. For each perceived vehicle, it computes the relative distance from the receiver. Depending on the configured mode, the object is considered inside the jammed region if it is within range, or if it is both within range and inside the receiver's forward field of view.

Objects inside the jamming region are removed according to the configured drop policy. The attack can remove all affected vehicles or limit the number of removed vehicles using `max_drop`. Optionally, it can also clear `objects["traffic_lights"]`.

## Jamming Modes

- `front_cone`: removes objects inside a forward cone in front of the receiver. The cone is controlled by `range_m` and `fov_deg`.
- `range_only`: removes objects within `range_m`, regardless of angle.
- `all`: currently behaves like range-based removal after the distance check, so objects must still be within `range_m`.

## Effect on the Victim

The victim may behave as if vehicles in the jammed region do not exist. This can lead to unsafe acceleration, missed collision avoidance, incorrect lane-change decisions, or failure to react to stopped or slow traffic.

## Additional YAML Configuration Parameters for attack

```yaml
attack:
  enabled: true
  type: lidar_jamming
  start_tick: 100
  attacker_vid: cav-100
  visible_to_attacker: false
  mode: front_cone
  range_m: 25.0
  fov_deg: 60.0
  drop_all: true
  max_drop: 999
  drop_traffic_lights: false
```

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `enabled` | boolean | `false` | Enables or disables the attack. |
| `type` | string | `""` | Must be set to `lidar_jamming` for this attack to run. |
| `start_tick` | integer | `100` | Simulation tick from which jamming starts. |
| `attacker_vid` | string | `cav-100` | Vehicle ID of the attacker. |
| `visible_to_attacker` | boolean | `false` | If `false`, the attacker is not affected by its own jamming attack. |
| `mode` | string | `front_cone` | Jamming geometry mode: `front_cone`, `range_only`, or `all`. |
| `range_m` | float | `25.0` | Maximum distance in meters from the receiver within which perceived vehicles can be removed. |
| `fov_deg` | float | `60.0` | Field of view angle in degrees for `front_cone` mode. The attack uses half of this angle on each side of the receiver's heading. |
| `drop_all` | boolean | `true` | If `true`, all vehicles inside the jamming region are removed. |
| `max_drop` | integer | `999` | Maximum number of vehicles to remove when `drop_all` is `false`. |
| `drop_traffic_lights` | boolean | `false` | If `true`, the attack also clears perceived traffic lights. |
```
