# Ghost-Sybil Attack

## Overview

The Ghost-Sybil attack injects fake vehicles into the local perception output of selected connected autonomous vehicles (CAVs). The injected vehicles are not real CARLA actors. Instead, they are lightweight actor-like objects that imitate the interface expected by the downstream perception and planning pipeline.

This attack is designed to make a victim vehicle believe that one or more phantom vehicles are present ahead of it. The phantom vehicles are placed along the victim's lane, starting at a configurable distance in front of the victim and separated by a configurable gap.

## How the Attack Works

Each simulation tick, the attack object checks whether the current receiver should be affected. The receiver is affected only if the attack is enabled, the local tick has reached `start_tick`, the attacker vehicle exists, and the receiver is within the attacker's V2X communication range.

If these conditions are satisfied, the attack removes any ghost vehicles injected during a previous tick and creates a new set of fake vehicles. These vehicles are inserted into `objects["vehicles"]`, so the rest of the perception and planning stack treats them as ordinary detected vehicles.

The ghost vehicles are generated relative to the receiver's current road waypoint, not relative to the attacker's physical position. Therefore, this implementation behaves as a perception-level injection attack rather than a true V2X Sybil attack that broadcasts multiple fake sender identities from the attacker's trajectory.

The attacker vehicle can be hidden from its own attack output unless `visible_to_attacker` is enabled.

## Effect on the Victim

The victim receives additional fake vehicles in its perceived vehicle list. Depending on the planner and controller behavior, this may cause the victim to slow down, stop, change its trajectory, or incorrectly reason about traffic density and lane occupancy.

## Additional YAML Configuration Parameters for attack

```yaml
attack:
  enabled: true
  type: ghost_sybil
  start_tick: 100
  attacker_vid: cav-100
  visible_to_attacker: false
  count: 1
  ghost_distance: 8.0
  ghost_gap: 6.0
  freeze_positions: true
```

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `enabled` | boolean | `false` | Enables or disables the attack. |
| `type` | string | `""` | Must be set to `ghost_sybil` for this attack to run. |
| `start_tick` | integer | `100` | Simulation tick from which the attack starts affecting receivers. |
| `attacker_vid` | string | `cav-100` | Vehicle ID of the attacker. |
| `visible_to_attacker` | boolean | `false` | If `false`, the attacker does not receive its own injected ghost vehicles. |
| `count` | integer | `1` | Number of ghost vehicles injected in front of each affected receiver. |
| `ghost_distance` | float | `8.0` | Distance in meters from the receiver to the first ghost vehicle. |
| `ghost_gap` | float | `6.0` | Distance in meters between consecutive ghost vehicles. |
| `freeze_positions` | boolean | `true` | Parsed by the implementation and stored in the attack object. In the current code, ghost vehicles are rebuilt from the receiver waypoint each tick, so this flag is not actively used to freeze cached positions. |
```
