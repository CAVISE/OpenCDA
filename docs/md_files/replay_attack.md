# Replay Attack

## Overview

The replay attack implements a delayed self-replay scenario. The attacker records its own trajectory into a shared buffer and later injects a fake clone of itself into the perception output of nearby victim vehicles.

The injected clone is not a real CARLA actor. It is an actor-like object with a transform, velocity, bounding box, and `type_id` prefix that allows the downstream pipeline to process it as a perceived vehicle.

## How the Attack Works

The attack has two phases: recording and replay.

During the recording phase, only the attack instance belonging to the attacker vehicle records frames. Each recorded frame contains the attacker's position, orientation, and speed. Recording starts at `record_start` and stops once `buffer_size` frames have been stored.

During the replay phase, receivers within the attacker's V2X communication range receive a fake clone. Replay starts at `replay_start`. On each tick, the attack selects a frame from the recorded buffer based on how many ticks have passed since replay started.

The clone is shifted laterally by `lateral_offset` meters relative to the recorded heading. A positive offset places the clone to the left of the recorded direction of travel. While replayed frames are available, the clone receives a velocity vector derived from the recorded speed and yaw. Once the replay index reaches the end of the buffer, the clone freezes at the last recorded position with zero velocity.

The replay buffer is shared between all attack instances through a class-level dictionary keyed by `attacker_vid`, so the attacker records once and all receivers can replay the same trajectory.

## Effect on the Victim

The victim perceives a delayed clone of the attacker, offset into a nearby lane or lateral position. This can create false occupancy, confuse prediction and planning modules, and make the victim react to a vehicle that is no longer physically present at that location.

## Additional YAML Configuration Parameters for attack

```yaml
attack:
  enabled: true
  type: replay
  attacker_vid: cav-100
  record_start: 0
  buffer_size: 100
  replay_start: 200
  lateral_offset: 3.5
  visible_to_attacker: false
```

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `enabled` | boolean | `false` | Enables or disables the attack. |
| `type` | string | `""` | Must be set to `replay` for this attack to run. |
| `attacker_vid` | string | `cav-100` | Vehicle ID of the attacker whose trajectory is recorded and replayed. |
| `record_start` | integer | `0` | Tick from which the attacker starts recording its trajectory. |
| `buffer_size` | integer | `100` | Maximum number of recorded trajectory frames. |
| `replay_start` | integer | `200` | Tick from which victims start receiving the replayed clone. |
| `lateral_offset` | float | `3.5` | Lateral shift in meters applied to the replayed clone. Positive values move the clone left relative to the recorded heading. |
| `visible_to_attacker` | boolean | `false` | If `false`, the attacker does not receive its own replay clone. |
```
