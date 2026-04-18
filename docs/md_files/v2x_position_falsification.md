## V2X Position Falsification Attack

### Overview

V2X Position Falsification is an attack on the cooperative communication layer (V2X).
Unlike GNSS Spoofing, which corrupts a vehicle's **own** position perception at the sensor level,
this attack falsifies the position and speed data broadcast via V2X (CAM messages) to deceive **other** vehicles.
The attacker vehicle knows its real position but intentionally lies when broadcasting Cooperative Awareness Messages.

The attack is configured **per-vehicle** in the scenario YAML file, meaning any subset of CAVs can be designated as attackers while the rest remain honest participants.

### Attack Methods

All attack classes implement two methods: `falsify_position(true_pos)` and `falsify_speed(true_speed)`.

#### Constant Offset (`V2XConstantOffsetAttacker`)

Shifts the reported position by a fixed offset `(dx, dy, dz)` in meters.
Speed is reported truthfully.

Typical use case: the attacker claims to be in an adjacent lane (e.g. `dx=3.5` shifts roughly one lane width laterally).

#### Ghost Vehicle (`V2XGhostVehicleAttacker`)

Reports a completely fabricated position and speed, ignoring the vehicle's true state entirely.
The broadcast always claims the vehicle is at `(ghost_x, ghost_y, ghost_z)` with `ghost_speed`.

Typical use case: simulating a non-existent obstacle or vehicle to force reactions from other CAVs.

#### Progressive Drift (`V2XProgressiveDriftAttacker`)

Gradually drifts the reported position away from the real one.
On each simulation tick the offset grows by `(dx, dy, dz)` meters.
Speed is reported truthfully.

Typical use case: modelling a slowly diverging position claim that is harder to detect than a sudden jump.

### Scenario Configuration

Attack and detection are configured in the `v2x` section of each vehicle in the scenario YAML.

**Attacker configuration:**

```yaml
v2x:
  position_falsification:
    enabled: true
    type: constant_offset   # constant_offset | ghost_vehicle | progressive_drift
    start_tick: 100          # attack activates after N simulation ticks
    dx: 3.5                  # offset in meters (constant_offset / progressive_drift)
    dy: 0.0
    dz: 0.0
```

### Source Files

| File | Description |
|------|-------------|
| `opencda/core/attack/v2x_position_falsification.py` | Attack and detection classes |
| `opencda/core/common/v2x_manager.py` | Integration into the V2X communication pipeline |
| `opencda/scenario_testing/config_yaml/v2x_position_falsification_attack.yaml` | Example scenario |
