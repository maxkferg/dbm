import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.OBJModel import OBJModel
import tools.Math2D as m2d

# The TileGrid is an abstraction of the floor of the building so we can model it as a gridded surface that contains a
# binary field indicating whether or not the tile has been "seen" by the robot (i.e. that the tile has intersected a
# portion of the LIDAR rays triangles

class TileGrid:
    def __init__(self, floor_file):
        self.obj = OBJModel(floor_file)
        if not self.obj.parse():
            self.obj = None
            return

        self.world_bound = self.obj.model_AABB()
        self.poly_arr = []
        self.visited = []

    def is_valid(self):
        return self.obj is not None

    def build_grid(self):
        pass

if __name__ == '__main__':
    filename = '/Users/otgaard/Development/dbm/sim/assets/output_floors.obj'
    TileGrid(filename)


