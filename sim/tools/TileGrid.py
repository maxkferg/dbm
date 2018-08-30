import os
import sys
import functools
from PIL import ImageTk, Image
import tools.TriangleRasteriser as tr

# Move intersection, rasterisation, and image handling here so that the TileGrid can be used in parallel in
# SeekerSimEnv.  This means that the visualiser only displays the results whereas the SeekerSimEnv uses this
# class to maintain an internal representation of the actual visiting algorithm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.OBJModel import OBJModel
import tools.Math2D as m2d
from math import gcd

# Keys for AABB type
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
    verts = [AABB[PA], [AABB[PB][X], AABB[PA][Y]], AABB[PB], [AABB[PA][X], AABB[PB][Y]]]
    centre = compute_centre(AABB)
    return verts, centre


def verts(AABB):
    return [[AABB[0][0], AABB[1][1]], AABB[1] , [AABB[1][0], AABB[0][1]], AABB[0]]


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
        self.centre = compute_centre(self.bound)
        self.grid = []                                      # The AABB grid (integer)
        self.visited = []                                   # Visited flag
        self.min_dims = []                                  # The minimum dimensions of the grid in each axis
        self.quadtree = None                                # The TileGrid quadtree (intersection queries)
        self.poly_arr = []                                  # Polygon array (AABB)
        self.offset = [0, 0]                                # 2D offset
        self.screen_scale = [1, 1]                          # The screen scaling parameter
        self.images = []                                    # The source image, used to store visited pixels

    def is_valid(self):
        return self.obj is not None

    def get_map_dims(self):
        return self.obj.dims

    def poly_count(self):
        return len(self.poly_arr)

    def get_poly(self, idx):
        poly = self.poly_arr[idx]
        return [
            m2d.cp_mul(poly[0], self.screen_scale),
            m2d.cp_mul(poly[1], self.screen_scale)]

    def get_image(self, idx):
        return self.images[idx]

    def set_screen_scale(self, scale):
        if m2d.is_vec2(scale):
            self.screen_scale = scale

    def build_grid(self):
        scale = self.obj.scale
        dims = self.obj.dims

        centre = m2d.mul(compute_centre(self.bound), dims[0])

        print("Map Dims:", scale, dims, centre)

        prim_count = self.obj.get_prim_count()

        min_dims = m2d.mul(m2d.sub(self.bound[PB][:-1], self.bound[PA][:-1]), dims[0])
        print(min_dims)

        tile_dims = []

        ss_offset = [dims[0]/2, dims[1]/2]                  # Screen space offset
        ss_scale = [1, -1]                                  # Flip y axis on screen
        # Find minimum dimensions
        for floor in range(prim_count):
            if floor % 2 == 1:
                continue                                    # Skip odd numbered primitives (the other tri in the quad)

            prim = self.obj.get_prim(floor)
            if len(prim) != 3:
                continue

            A = self.obj.get_position(prim[0])[:-1]         # Left Bottom corner, truncate z coord
            B = self.obj.get_position(prim[1])[:-1]         # Right Bottom corner
            C = self.obj.get_position(prim[2])[:-1]         # Left Top corner

            lb = m2d.sub(m2d.cp_mul([A[X], A[Y]], dims), centre)
            rt = m2d.sub(m2d.cp_mul([B[X], C[Y]], dims), centre)

            # Move the polygon into screen-space for direct display by the Display Window
            slb = m2d.add(m2d.cp_mul(lb, ss_scale), ss_offset)
            srt = m2d.add(m2d.cp_mul(rt, ss_scale), ss_offset)

            self.poly_arr.append([slb, srt])

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

    def build_map(self):
        """The build_map routine takes the grid defined in build_grid, and creates the associated set of images"""
        self.images.clear()
        self.set_screen_scale([1, 1])

        scaled = []

        for floor in range(len(self.poly_arr)):
            LB, TR = self.get_poly(floor)
            # Create an image the same size as the rectangle and map pixels 1 to 1
            w = int(TR[0] - LB[0])
            h = int(LB[1] - TR[1])  # Y is flipped
            if w < 1 or h < 0:
                continue

            img = Image.new("RGB", (w, h))
            pixels = [None] * (w * h)

            half = w * h / 2
            A = m2d.rand_colour3()
            B = m2d.rand_colour3()
            for i in range(w * h):
                pixels[i] = A if i < half else B
            img.putdata(pixels)
            self.images.append(img)
            photo = ImageTk.PhotoImage(image=img)
            scaled.append(photo)

        return scaled

    def scale_map(self, scale):
        """Returns a list of images scaled to the input scale"""
        self.set_screen_scale(scale)

        scaled = []

        for floor in range(self.poly_count()):
            img = self.images[floor]
            new_size = (int(scale[0] * img.width), int(scale[1] * img.height))
            print(new_size)
            img = img.resize(new_size, Image.NEAREST)
            photo = ImageTk.PhotoImage(image=img)
            scaled.append(photo)

        return scaled


if __name__ == '__main__':
    filename = '/Users/otgaard/Development/dbm/sim/output/test2_floors.obj'
    grid = TileGrid(filename)
    grid.build_grid()
