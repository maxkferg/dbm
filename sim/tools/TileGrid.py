import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.OBJModel import OBJModel
import tools.Math2D as m2d


PA = 0
PB = 1
X = 0
Y = 1


def compute_centre(AABB):
    return [
        (AABB[PB][X] - AABB[PA][X]) / 2 + AABB[PA][X],
        (AABB[PB][Y] - AABB[PA][Y]) / 2 + AABB[PA][Y]
    ]


# The TileGrid is an abstraction of the floor of the building so we can model it as a gridded surface that contains a
# binary field indicating whether or not the tile has been "seen" by the robot (i.e. that the tile has intersected a
# portion of the LIDAR rays triangles

class TileGrid:
    def __init__(self, floor_file):
        self.obj = OBJModel(floor_file)
        if not self.obj.parse():
            self.obj = None
            return

        self.bound = self.obj.model_AABB()
        self.grid = []                                      # The AABB grid (integer)
        self.visited = []                                   # Visited flag
        self.min_dims = []                                  # The minimum dimensions of the grid in each axis
        self.quadtree = None                                # The TileGrid quadtree (intersection queries)

    def is_valid(self):
        return self.obj is not None

    def build_grid(self):
        centre = compute_centre(self.bound)
        prim_count = self.obj.get_prim_count()

        min_dims = m2d.sub(self.bound[PB], self.bound[PA])
        print(min_dims)

        # Find minimum dimensions
        for floor in range(prim_count):
            prim = self.obj.get_prim(floor)
            if len(prim) != 3:
                continue

            A = self.obj.get_position(prim[0])[:-1]         # Left Bottom corner, truncate z coord
            B = self.obj.get_position(prim[2])[:-1]         # Top Right corner, truncate z coord

            tile_delta = m2d.sub(B, A)

            if tile_delta[X] == 0:
                print(floor, tile_delta)

            if tile_delta[X] < min_dims[X]:
                min_dims[X] = tile_delta[X]
            if tile_delta[Y] < min_dims[Y]:
                min_dims[Y] = tile_delta[Y]

        print(min_dims)
        self.min_dims = min_dims


if __name__ == '__main__':
    filename = '/Users/otgaard/Development/dbm/sim/assets/output_floors.obj'
    grid = TileGrid(filename)
    grid.build_grid()
