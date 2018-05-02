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

horizontals = clean_horizontals
verticals = clean_verticals

print("\n")

print("HORIZONTALS:" + str(len(horizontals)))
print(horizontals)
print("\n")
print("VERTICALS:" + str(len(verticals)))
print(verticals)
print("\n")

# Compute stats about the mesh

min_x = im.size[0]
max_x = 0
min_y = im.size[1]
max_y = 0

for h in horizontals:
    if h[0][0] < min_x: min_x = h[0][0]
    elif h[0][1] > max_x: max_x = h[0][1]

    if h[1] < min_y: min_y = h[1]
    elif h[1] > max_y: max_y = h[1]

print("TOP_LEFT:     [", min_x, min_y, "]")
print("BOTTOM_RIGHT: [", max_x, max_y, "]")


def start_pos(line):
    return [line[0][0], line[1]] if is_horizontal(line) else [line[0], line[1][0]]


def end_pos(line):
    return [line[0][1], line[1]] if is_horizontal(line) else [line[0], line[1][1]]


def find_adjoining(line):
    if is_horizontal(line):                             # [ [x_min, x_max], y ], [ x, [y_min, y_max] ]
        return list(filter(lambda l: (line[1] == l[1][0] or line[1] == l[1][1]) and
                                     (line[0][0] == l[0] or line[0][1] == l[0]), verticals))
    elif is_vertical(line):                             # [ x, [y_min, y_max] ], [ [x_min, x_max], y ]
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
# 3) If no connector is found, assume must be interior wall, if vertical, normal is -1 if horizontal, normal is -1

lines = verticals + horizontals
normals = [[]]*len(lines)
normals[0] = [+1, 0]


def turn_left(normal):
    return [-normal[1], normal[0]]


def turn_right(normal):
    return [normal[1], -normal[0]]


def compute_normal(neigh_line, neigh_normal, curr_line):
    nSP = start_pos(neigh_line)
    nEP = end_pos(neigh_line)
    cSP = start_pos(curr_line)
    cEP = end_pos(curr_line)

    if is_horizontal(curr_line):
        if cSP == nEP: return turn_right(neigh_normal)
        elif cSP == nSP: return turn_left(neigh_normal)
        elif cEP == nEP: return turn_left(neigh_normal)
        elif cEP == nSP: return turn_right(neigh_normal)
    elif is_vertical(curr_line):
        if cSP == nEP: return turn_left(neigh_normal)
        elif cSP == nSP: return turn_right(neigh_normal)
        elif cEP == nEP: return turn_right(neigh_normal)
        elif cEP == nSP: return turn_left(neigh_normal)

    return []

# The outside wall is a connected graph so we can traverse it and induce the first normal to derive the rest
# Inner walls will have to be induced by some other method (possibly searching for opposite walls?)

curr_line = lines[0]
curr_normal = normals[0]

while curr_line:
    neighbours = find_adjoining(curr_line)
    nidx = list(map(lambda x: lines.index(x), neighbours))
    print("Curr line:", curr_line, "Curr N:", curr_normal, "Neighbours:", neighbours, "Neighbour Ids:", nidx)

    if not normals[nidx[0]]:
        normals[nidx[0]] = compute_normal(curr_line, curr_normal, lines[nidx[0]])
        curr_normal = normals[nidx[0]]
        curr_line = lines[nidx[0]]
    elif not normals[nidx[1]]:
        normals[nidx[1]] = compute_normal(curr_line, curr_normal, lines[nidx[1]])
        curr_normal = normals[nidx[1]]
        curr_line = lines[nidx[1]]
    else:
        # Search for the next line without a normal, it must, by inference, be a vertical line to the left and
        # therefore has a normal of [-1,0], repeat until no more vertical lines with null normals remain
        idx = normals.index([])
        curr_line = lines[idx] if is_vertical(lines[idx]) else None
        if curr_line:
            curr_normal = [-1, 0]
            normals[idx] = [-1, 0]

# Test the algorithm by outputting a diagram of the same size with the walls highlighted in red and normals in blue

img = Image.new('RGB', im.size, color='black')
px = img.load()

for h in horizontals:
    for col in range(h[0][0], h[0][1]+1):
        px[col, h[1]] = (255, 0, 0)

for v in verticals:
    for row in range(v[1][0], v[1][1]+1):
        px[v[0], row] = (255, 0, 0)

# Draw the normals

for l in range(len(lines)):
    if is_horizontal(lines[l]):
        nor = normals[l]
        lne = lines[l]
        #print(nor, lne)
        hw = (lne[0][0] + lne[0][1])/2
        if nor != [] and nor[1] == 1:
            for row in range(lne[1], lne[1] + 20):
                px[hw, row] = (0, 0, 255)
        elif nor != [] and nor[1] == -1:
            for row in range(lne[1] - 20, lne[1]):
                px[hw, row] = (0, 0, 255)
    elif is_vertical(lines[l]):
        nor = normals[l]
        lne = lines[l]
        #print(nor, lne)
        hh = (lne[1][0] + lne[1][1])/2
        if nor != [] and nor[0] == 1:
            for col in range(lne[0], lne[0] + 20):
                px[col, hh] = (0, 0, 255)
        elif nor != [] and nor[0] == -1:
            for col in range(lne[0] - 20, lne[0]):
                px[col, hh] = (0, 0, 255)


img.save("assets/test.png", "PNG")
