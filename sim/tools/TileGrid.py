import os
import sys
import functools

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.OBJModel import OBJModel
import tools.Math2D as m2d
from math import gcd


PA = 0
PB = 1
X = 0
Y = 1


def compute_centre(AABB):
    return [
        (AABB[PB][X] - AABB[PA][X]) / 2 + AABB[PA][X],
        (AABB[PB][Y] - AABB[PA][Y]) / 2 + AABB[PA][Y]
    ]


def AABB_to_vertices(AABB):
    verts = [AABB[0], [AABB[1][0], AABB[0][1]], AABB[1], [AABB[0][0], AABB[1][1]]]
    centre = [(AABB[1][0] - AABB[0][0]) / 2, (AABB[1][1] - AABB[0][1]) / 2]
    return verts, centre



# The TileGrid is an abstraction of the floor of the building so we can model it as a gridded surface that contains a
# binary field indicating whether or not the tile has been "seen" by the robot (i.e. that the tile has intersected a
# portion of the LIDAR triangles

class TileGrid:
    def __init__(self, floor_file):
        self.obj = OBJModel(floor_file)
        if not self.obj.parse():
            print("Could not load OBJ file", floor_file)
            self.obj = None
            return

        self.bound = self.obj.model_AABB()
        self.grid = []                                      # The AABB grid (integer)
        self.visited = []                                   # Visited flag
        self.min_dims = []                                  # The minimum dimensions of the grid in each axis
        self.quadtree = None                                # The TileGrid quadtree (intersection queries)
        self.poly_arr = []                                  # Polygon array

    def is_valid(self):
        return self.obj is not None

    def poly_count(self):
        return len(self.poly_arr)

    def get_poly(self, idx):
        return self.poly_arr[idx]

    def build_grid(self):
        scale = self.obj.scale
        dims = self.obj.dims

        centre = m2d.mul(compute_centre(self.bound), dims[0])

        print("Map Dims:", scale, dims, centre)

        prim_count = self.obj.get_prim_count()

        min_dims = m2d.mul(m2d.sub(self.bound[PB][:-1], self.bound[PA][:-1]), dims[0])
        print(min_dims)

        tile_dims = []

        # Find minimum dimensions
        for floor in range(prim_count):
            prim = self.obj.get_prim(floor)
            if len(prim) != 3:
                continue

            A = self.obj.get_position(prim[0])[:-1]         # Left Bottom corner, truncate z coord
            B = self.obj.get_position(prim[1])[:-1]         # Right Bottom corner
            C = self.obj.get_position(prim[2])[:-1]         # Left Top corner

            lb = m2d.sub(m2d.mul([A[X], A[Y]], dims[0]), centre)
            rt = m2d.sub(m2d.mul([B[X], C[Y]], dims[0]), centre)

            self.poly_arr.append([lb, rt])

            tile_delta = m2d.sub(rt, lb)
            print(floor, A, B, tile_delta)

            if tile_delta[X] < min_dims[X]:
                min_dims[X] = tile_delta[X]
            if tile_delta[Y] < min_dims[Y]:
                min_dims[Y] = tile_delta[Y]

            tile_dims.append(tile_delta)

        print(min_dims)
        self.min_dims = min_dims

        # Compute the greatest common divisor for each axis
        x_dims = list(map(lambda x: x[0], tile_dims))
        y_dims = list(map(lambda x: x[1], tile_dims))
        xmin = functools.reduce(lambda x, y: gcd(int(x), int(y)), x_dims)
        ymin = functools.reduce(lambda x, y: gcd(int(x), int(y)), y_dims)

        print("X Axis GCD:", xmin, x_dims)                  # Seems to be 1 in most cases... will have to be by pixel
        print("Y Axis GCD:", ymin, y_dims)
        print("Polygons:", self.poly_arr)


if __name__ == '__main__':
    print("Hello")

    filename = '/Users/otgaard/Development/dbm/sim/output/test2_floors.obj'
    grid = TileGrid(filename)
    grid.build_grid()
