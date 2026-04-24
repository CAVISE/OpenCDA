"""
GNSS Spoofing attack
"""

import math
from random import gauss


class GNSSProgressiveSpoofer:
    def __init__(self, dx: float, dy: float, dz: float) -> None:
        self.tick = 0
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0

    def get_geolocation(self) -> tuple[float, float, float]:
        return self.lat, self.lon, self.alt

    def update(self, lat: float, lon: float, alt: float) -> tuple[float, float, float]:
        self.lat = lat + self.tick * self.dx
        self.lon = lon + self.tick * self.dy
        self.alt = alt + self.tick * self.dz
        self.tick += 1
        return self.get_geolocation()


class GNSSPeriodicSpoofer:
    def __init__(self, dx: float, dy: float, dz: float, period: float, count: float = 5) -> None:
        self.period = period
        self.count = count
        self.tick = 0
        self.next_tick = abs(round(gauss(self.period, self.period / 10)))
        self.next_count = abs(round(gauss(self.count)))
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0

    def get_geolocation(self) -> tuple[float, float, float]:
        return self.lat, self.lon, self.alt

    def update(self, lat: float, lon: float, alt: float) -> tuple[float, float, float]:
        self.lat = lat
        self.lon = lon
        self.alt = alt
        if self.tick < self.next_tick:
            self.tick += 1
        elif self.tick < self.next_tick + self.next_count:
            self.lat += gauss(self.dx, self.dx / 10)
            self.lon += gauss(self.dy, self.dy / 10)
            # self.alt += gauss(self.dz, self.dz / 10)
            self.tick += 1
        else:
            self.tick = 0
            self.next_tick = abs(round(gauss(self.period, self.period / 10)))
            self.next_count = abs(round(gauss(self.count)))
        return self.get_geolocation()


class GNSSSpoofingDetector:
    def __init__(self, dt: float, th: float = 1.0) -> None:
        self.dt = dt
        self.th = th
        self.x = 0.0
        self.y = 0.0

    def first_step(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def detect(self, x: float, y: float, v: float) -> bool:
        s = math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)
        self.x = x
        self.y = y
        return s > v * self.dt + self.th
