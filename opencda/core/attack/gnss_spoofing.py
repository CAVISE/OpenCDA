"""
GNSS Spoofing attack
"""

import math
from random import gauss

class GNSSProgressiveSpoofer:
    def __init__(self, dx, dy, dz):
        self.tick = 0
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.lat = 0
        self.lon = 0
        self.alt = 0

    def get_geolocation(self):
        return self.lat, self.lon, self.alt

    def update(self, lat, lon, alt):
        self.lat = lat + self.tick * self.dx
        self.lon = lon + self.tick * self.dy
        self.alt = alt + self.tick * self.dz
        self.tick += 1
        return self.get_geolocation()


class GNSSPeriodicSpoofer:
    def __init__(self, dx, dy, dz, period, count=5):
        self.period = period
        self.count = count
        self.tick = 0
        self.next_tick = abs(round(gauss(self.period, self.period/10)))
        self.next_count = abs(round(gauss(self.count)))
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.lat = 0
        self.lon = 0
        self.alt = 0

    def get_geolocation(self):
        return self.lat, self.lon, self.alt

    def update(self, lat, lon, alt):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        if self.tick < self.next_tick:
            self.tick += 1
        elif self.tick < self.next_tick + self.next_count:
            self.lat += gauss(self.dx, self.dx/10)
            self.lon += gauss(self.dy, self.dy/10)
            #self.alt += gauss(self.dz, self.dz/10)
            self.tick += 1
        else:
            self.tick = 0
            self.next_tick = abs(round(gauss(self.period, self.period/10)))
            self.next_count = abs(round(gauss(self.count)))
        return self.get_geolocation()


class GNSSSpoofingDetector:
    def __init__(self, dt, th=1.0):
        self.dt = dt
        self.th = th
        self.x = 0
        self.y = 0


    def first_step(self, x, y):
        self.x = x
        self.y = y

    def detect(self, x, y, v):
        s = math.sqrt((self.x - x)**2 + (self.y - y)**2)
        self.x = x
        self.y = y
        return s > v * self.dt + self.th
