import numpy as np


class ERA5LandGrid(object):
    """A class defining ERA5 land global 0.1 deg resolution regular grid"""

    def __init__(self):
        self.lon_min = -180
        self.lat_min = -90
        self.nx = 3600
        self.ny = 1801
        self.dx = 0.1
        self.dy = 0.1

    def modulo_positive(self, arr):
        return np.where(arr >= 0, arr % self.ny, (arr + self.ny) % self.ny)

    def find_point_xy(self, lat_arr, lon_arr):
        """Find the grid indices for a given latitude and longitude array"""
        # Calculate the x and y grid indices for the arrays
        x = np.round((lon_arr - self.lon_min) / self.dx).astype(int)
        y = np.round((lat_arr - self.lat_min) / self.dy).astype(int)

        # Check if the grid wraps around for global grids
        xx = np.where((self.nx * self.dx >= 359), x % self.nx, x)
        yy = np.where((self.ny * self.dy >= 179), self.modulo_positive(y), y)

        # Create masks to check for out-of-bound indices
        valid_mask = (yy >= 0) & (xx >= 0) & (yy < self.ny) & (xx < self.nx)

        # Return indices where valid, otherwise return None
        return np.where(valid_mask, xx, np.nan), np.where(valid_mask, yy, np.nan)
