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

### Detection (`V2XAttackDetector`)

Detection is performed on the **receiver** side: every CAV with `attack_detection.enabled: true` instantiates its own `V2XAttackDetector`, which inspects the position/speed broadcast by every neighbor that enters the V2X communication range.

The detector compares the position reported via V2X with the neighbor's true CARLA position (`vehicle.get_location()`). In a real deployment the trusted reference would come from independent perception (LiDAR/camera/radar/RSU); using CARLA's ground truth is a simulation-only stand-in for that channel.

Three detection strategies are available, selected via `attack_detection.method`:

#### `threshold` — single-tick distance check

Computes the planar distance between the reported and the true XY position. Raises an alert when it exceeds `position_threshold` (meters). Stateless. Catches obvious displacement attacks (`constant_offset` with non-trivial `dx/dy`, `ghost_vehicle`).

#### `drift` — sliding-window mean error

Stores per-neighbor history of `||reported - true||` in a `deque(maxlen=window_size)`. Raises an alert when the window mean exceeds `drift_mean_threshold` **and** the error grew across the window (`history[-1] > history[0]`). Designed for `progressive_drift`, where a single-tick threshold would only trigger after the offset has accumulated for many ticks.

#### `velocity_consistency` — passive sanity check

Compares the displacement between two consecutive V2X broadcasts with what the reported speed would predict:

```
expected = avg(reported_speed) * dt
actual   = ||reported_pos[t] - reported_pos[t-1]||
alert if |actual - expected| > speed_threshold
```

Does not use ground truth — only V2X data — and therefore acts as an independent cross-check. Catches attacks that falsify position without keeping reported speed consistent (`constant_offset`, `progressive_drift`). It does **not** catch `ghost_vehicle` if both position and speed are fabricated coherently (e.g. constant ghost coordinates with `ghost_speed: 0`).

When an anomaly is detected, the detector emits a `logger.warning` line of the form:

```
[V2X-DETECTOR][CAV cav-2] Anomaly in V2X data from CAV cav-1: method=threshold, error=3.50m, threshold=1.5m, tick=42
```

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

**Receiver / detector configuration:**

```yaml
v2x:
  communication_range: 250         # neighbor must be within this range for detection to run
  attack_detection:
    enabled: true
    method: threshold              # threshold | drift | velocity_consistency
    position_threshold: 1.5        # meters; used by `threshold`
    window_size: 10                # used by `drift`
    drift_mean_threshold: 0.5      # meters; used by `drift`
    speed_threshold: 1.0           # meters; used by `velocity_consistency`
    dt: 0.05                       # seconds; used by `velocity_consistency`
```

### Source Files

| File | Description |
|------|-------------|
| `opencda/core/attack/v2x_position_falsification.py` | Attack and detection classes |
| `opencda/core/common/v2x_manager.py` | Integration into the V2X communication pipeline |
| `opencda/scenario_testing/config_yaml/v2x_position_falsification_attack.yaml` | Example scenario |
