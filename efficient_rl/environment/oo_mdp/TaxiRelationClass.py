import numpy as np


class TaxiRelations:

    @staticmethod
    def touch_south(o1, o2):
        # north wall indices that hold the same x position as object 1
        north_walls_idx = np.logical_and(o2['position'] == 'above', o2['x'] == o1['x'])
        # check for collision on north wall
        collide_n = np.any(o2['y'][north_walls_idx] == o1['y'])
        # south wall indices that hold the same x position as object 1
        south_walls_idx = np.logical_and(o2['position'] == 'below', o2['x'] == o1['x'])
        # check for collision on south wall
        collide_s = np.any(o2['y'][south_walls_idx] == o1['y'] - 1)
        return collide_s or collide_n

    @staticmethod
    def touch_north(o1, o2):
        # north wall indices that hold the same x position as object 1
        north_walls_idx = np.logical_and(o2['position'] == 'above', o2['x'] == o1['x'])
        # check for collision on north wall
        collide_n = np.any(o2['y'][north_walls_idx] == o1['y'] + 1)
        # south wall indices that hold the same x position as object 1
        south_walls_idx = np.logical_and(o2['position'] == 'below', o2['x'] == o1['x'])
        collide_s = np.any(o2['y'][south_walls_idx] == o1['y'])
        return collide_s or collide_n

    @staticmethod
    def touch_east(o1, o2):
        # east wall indices that hold the same y position as object 1
        east_wall_idx = np.logical_and(o2['position'] == 'right', o2['y'] == o1['y'])
        # check for collision on east wall
        collide_e = np.any(o2['x'][east_wall_idx] == o1['x'])
        # west wall indices that hold the same y position as object 1
        west_wall_idx = np.logical_and(o2['position'] == 'left', o2['y'] == o1['y'])
        collide_w = np.any(o2['x'][west_wall_idx] == o1['x'] + 1)
        return collide_e or collide_w

    @staticmethod
    def touch_west(o1, o2):
        # east wall indices that hold the same y position as object 1
        east_wall_idx = np.logical_and(o2['position'] == 'right', o2['y'] == o1['y'])
        # check for collision on east wall
        collide_e = np.any(o2['x'][east_wall_idx] == o1['x'] - 1)
        # west wall indices that hold the same y position as object 1
        west_wall_idx = np.logical_and(o2['position'] == 'left', o2['y'] == o1['y'])
        collide_w = np.any(o2['x'][west_wall_idx] == o1['x'])
        return collide_e or collide_w

    @staticmethod
    def on(o1, o2):
        return o1['x'] == o2['x'] and o1['y'] == o2['y']
