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
    if row % 10 == 0:
        sys.stdout.write('.')
        sys.stdout.flush()
    if row % 500 == 0:
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

print("HORIZONTALS:")
print(clean_horizontals)
print("\n\n\n")
print("VERTICALS:")
print(clean_verticals)

# Test the algorithm by outputting a diagram of the same size with the walls highlighted in red

img = Image.new('RGB', im.size, color='white')
px = img.load()

for h in clean_horizontals:
    for col in range(h[0][0], h[0][1]):
        px[col, h[1]] = (255, 0, 0)

for v in clean_verticals:
    for row in range(v[1][0], v[1][1]):
        px[v[0], row] = (255, 0, 0)

img.save("/Users/otgaard/Development/dbm/sim/assets/test.png", "PNG")

