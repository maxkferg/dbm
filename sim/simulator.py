import sys
from PIL import Image
import numpy as np
import scipy as sp

im = Image.open("assets/Level 2 floor plan walls.png")
px = im.load()
print("Image Size:", im.size)

# Notes:
# 1) All lines are vertical or horizontal
# 2) Any corner in the image must be split into two lines
# 3) A line can be queried by a fixed column or row and a start and an end pixel (pixels in between are assumed to have
#    been processed.
# 4) A line is defined by:
#       horizontal: [ [ x_min, x_max ], y               ]
#       vertical:   [ x,                [ y_min, y_max] ]
# 5) Lines are held in different containers, first search the horizontal by y or vertical by x, if it contains lines in
#    fixed column or row, search the [ min, max ] values to determine if it can be appended or must be split
# 6) A line [ x, [ y, y ] ] is valid (a single pixel)
# 7) There is no need to check if two lines merge due to the image being processed top to bottom, left to right, in that
#    order, meaning that we first try to join verticals and then horizontals.


def is_horizontal(line): return isinstance(line, list) and isinstance(line[0], list)


def is_vertical(line): return isinstance(line, list) and isinstance(line[1], list)


horizontals = []
verticals = []

for row in range(im.size[1]):
    if row % 10 == 0 and row != 0:
        sys.stdout.write('.')
        sys.stdout.flush()
    if row % 500 == 0 and row != 0:
        print('#')
    for col in range(im.size[0]):
        if px[col, row][0] != 255:
            verts = list(filter(lambda l: l[0] == col and l[1][1] == row-1, verticals))
            horzs = list(filter(lambda l: l[1] == row and l[0][1] == col-1, horizontals))

            if len(verts) == 0 and len(horzs) == 0:
                verticals.append([col, [row, row]])          # Add a pixel
                horizontals.append([[col, col], row])
            else:
                if len(verts) > 0:
                    verts[0][1][1] += 1
                else:
                    verticals.append([col, [row, row]])

                if len(horzs) > 0:
                    horzs[0][0][1] += 1
                else:
                    horizontals.append([[col, col], row])

# Remove all single-pixel lines
clean_horizontals = [h for h in horizontals if h[0][0] != h[0][1]]
clean_verticals = [v for v in verticals if v[1][0] != v[1][1]]

print("\n")

print("HORIZONTALS:" + str(len(clean_horizontals)))
print(clean_horizontals)
print("\n")
print("VERTICALS:" + str(len(clean_verticals)))
print(clean_verticals)
print("\n")

# Compute stats about the mesh

min_x = im.size[0]
max_x = 0
min_y = im.size[1]
max_y = 0

for h in clean_horizontals:
    if h[0][0] < min_x: min_x = h[0][0]
    elif h[0][1] > max_x: max_x = h[0][1]

    if h[1] < min_y: min_y = h[1]
    elif h[1] > max_y: max_y = h[1]

print("TOP_LEFT:     [", min_x, min_y, "]")
print("BOTTOM_RIGHT: [", max_x, max_y, "]")


def find_adjoining(line):
    if is_horizontal(line):                             # [ [x_min, x_max], y ]
        print("Horizontal, search the verticals")       # [ x, [y_min, y_max] ]
        return list(filter(lambda l: (line[1] == l[1][0] or line[1] == l[1][1]) and
                                     (line[0][0] == l[0] or line[0][1] == l[0]), verticals))
    elif is_vertical(line):                             # [ x, [y_min, y_max] ]
        print("Vertical, search the horizontals")       # [ [x_min, x_max], y ]
        return list(filter(lambda l: (line[0] == l[0][0] or line[0] == l[0][1]) and
                                     (line[1][0] == l[1] or line[1][1] == l[1]), horizontals))
    else:
        print("Error, not a valid vertical or horizontal line")
        return []


# Compute the normals of the walls
#
# Notes:
# 1) Seed the first two lines with the correct normal
# 2) For each line, find all connectors and rotate normal accordingly
# 3) If no connector is found, search for the nearest line above or to the left of the line and negate


horz_normals = [[0, -1]]
vert_normals = [[+1, 0]]


for h in clean_horizontals:
    print(h, find_adjoining(h))

for v in clean_verticals:
    print(v, find_adjoining(v))

# Test the algorithm by outputting a diagram of the same size with the walls highlighted in red and normals in blue

img = Image.new('RGB', im.size, color='black')
px = img.load()

for h in clean_horizontals:
    for col in range(h[0][0], h[0][1]+1):
        px[col, h[1]] = (255, 0, 0)

for v in clean_verticals:
    for row in range(v[1][0], v[1][1]+1):
        px[v[0], row] = (255, 0, 0)

img.save("/Users/otgaard/Development/dbm/sim/assets/test.png", "PNG")
