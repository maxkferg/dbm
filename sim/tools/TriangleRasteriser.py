import math
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from tools.OBJModel import OBJModel
import tools.Math2D as m2d

# The LIDAR is represented by a set of triangles representing individual rays, starting at the origin in robot space
# and storing the visible field of the robot.  The TriangleRasteriser is responsible for computing the intersection
# between these triangles and the visited pixels in the image.  As the car moves around, the visible field is moved
# and intersected with a bitfield representing whether the unit square has been visited or not.  The square is
# identified by rasterisation because this is a lot faster than populating and querying a quadtree.  Rather than search
# a prepared space of pixels, the algorithm will simply visit the pixels directly by visiting the pixels in the
# rasterisation pattern.  The algorithm uses a simple barycentric algorithm, to be upgraded to a better performing
# algorithm if required


def minmax(v0, v1, v2):
    maxX = max(v0.x, v1.x, v2.x)
    minX = min(v0.x, v1.x, v2.x)
    maxY = max(v0.y, v1.y, v2.y)
    minY = min(v0.y, v1.y, v2.y)
    return [[minX, maxX], [minY, maxY]]


def kross(u, v):
    return u[0]*v[1] - u[1]*u[0]


# Will call a callback with a integer tuple representing a visited pixel, this must be converted into screen
def rasterise(vertices, callback):
    if len(vertices) != 3:
        print("Error, incorrect number of vertices")
        return False

    v0 = vertices[0]
    v1 = vertices[1]
    v2 = vertices[2]

    d0 = m2d.sub(v1, v0)
    d1 = m2d.sub(v2, v0)

    mnmx = minmax(v0, v1, v2)

    k = kross(d0, d1)

    for x in range(mnmx[0][0], mnmx[0][1]+1):
        for y in range(mnmx[1][0], mnmx[1][1]+1):
            q = m2d.sub([x, y], v0)
            s = float(kross(q, d1) / k)
            t = float(kross(d0, q) / k)
            if s >= 0 and t >= 0 and (s + t <= 1):
                callback([x, y])
