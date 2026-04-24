## GNSS Spoofing Attack

Module for simulating GNSS coordinate spoofing attacks and their detection within the OpenCDA localization pipeline.

**Source code:** `opencda/core/attack/gnss_spoofing.py`

---

### Attack Models

#### GNSSProgressiveSpoofer

Applies a linearly increasing offset on each simulation tick. The spoofed coordinates progressively diverge from the true position over time.

| Parameter | Description |
|-----------|-------------|
| `dx`, `dy`, `dz` | Offset increment per tick (latitude, longitude, altitude) |

#### GNSSPeriodicSpoofer

Generates sporadic spoofing bursts at randomized intervals. Between bursts, coordinates are passed through unmodified.

| Parameter | Description |
|-----------|-------------|
| `dx`, `dy`, `dz` | Spoofing magnitude (mean of Gaussian distribution) |
| `period` | Mean number of ticks between bursts |
| `count` | Mean burst duration (ticks) |

Both models share a unified interface: `update(lat, lon, alt) -> (lat, lon, alt)`.

---

### Detection

#### GNSSSpoofingDetector

Detects spoofing by comparing the displacement between successive positions against the expected distance based on vehicle velocity.

Triggering condition: `displacement > velocity * dt + threshold`

| Parameter | Description |
|-----------|-------------|
| `dt` | Simulation time step (seconds) |
| `th` | Distance threshold (meters, default 1.0) |

---

### Integration with LocalizationManager

The attack and detector are activated via flags in the YAML configuration under the `localization` section:

```yaml
localization:
  activate: true
  attack: true   # enable GNSS spoofing
  detect: true   # enable spoofing detection
```

When enabled, `LocalizationManager.localize()` applies the spoofer to GNSS coordinates before the Kalman filter, and the detector runs after filtering.

---

### Scenario Configuration

A dedicated scenario configuration with the attack enabled:

`config_yaml/realistic_town06_cosim_gnss_attack.yaml`

This file is based on `realistic_town06_cosim.yaml` with attack/detection enabled and localization activated. The base scenario file remains unchanged.
