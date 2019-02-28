import math
import sys
import os
import time

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
    maxX = max(v0[0], v1[0], v2[0])
    minX = min(v0[0], v1[0], v2[0])
    maxY = max(v0[1], v1[1], v2[1])
    minY = min(v0[1], v1[1], v2[1])
    return [[minX, maxX], [minY, maxY]]


def kross(u, v):
    return u[0]*v[1] - u[1]*v[0]


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
                callback((x, y))


if __name__ == "__main__":
    # Test the triangle rasterisation routine
    from tkinter import Tk, Canvas
    from PIL import ImageTk, Image

    root = Tk()
    root.title("Triangle Rasterisation Test")
    root.geometry("512x512")
    canvas = Canvas(root, width=512, height=512)
    canvas.pack(fill="both", expand=True)

    w = 512
    h = 512
    size = w*h

    img = Image.new("RGB", (w, h))
    pixels = [(255, 255, 255)] * size
    img.putdata(pixels)
    photo = ImageTk.PhotoImage(image=img)
    vertices = [(100, 100), (200, 200), (300, 100)]
    centre = m2d.find_centre(vertices)

    # Check rotation & various close-to-degenerate tests
    increment = math.pi/30.
    angle = 0.

    def callback(coord):
        global pixels
        pixels[coord[1]*w + coord[0]] = (0, 0, 0)

    def update_fnc():
        global pixels
        global angle
        global img
        global photo
        global canvas

        print(angle)

        canvas.delete("all")
        pixels = [(255, 255, 255)] * size
        new_verts = list(map(lambda x: (int(x[0]), int(x[1])), m2d.rotate(vertices, angle, centre)))
        print(new_verts)
        rasterise(new_verts, callback)
        img.putdata(pixels)
        photo = ImageTk.PhotoImage(image=img)
        canvas.create_image(w/2, h/2, image=photo)
        angle += increment
        canvas.after(200, update_fnc)


    update_fnc()
    root.mainloop()
