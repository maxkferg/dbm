import sys
from PIL import Image
import numpy as np
import math

'''Lines are based on the assumption that they are horizontal or vertical and
therefore are identified by a range over x or y and a single value for the coordinate
axis thus:

Vertical Lines   = [ x,              [y_min, y_max] ]
Horizontal Lines = [ [x_min, x_max], y              ]'''


def is_horizontal(line): return isinstance(line, list) and isinstance(line[0], list)


def is_vertical(line): return isinstance(line, list) and isinstance(line[1], list)


def start_pos(line):
    return [line[0][0], line[1]] if is_horizontal(line) else [line[0], line[1][0]]


def end_pos(line):
    return [line[0][1], line[1]] if is_horizontal(line) else [line[0], line[1][1]]


# Basically the 2D perp operator
def turn_left(normal):
    return [-normal[1], normal[0]]


# And the inverse thereof
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


def translation_matrix(pos):
    return np.array([[1., 0., 0., pos[0]],
                     [0., 1., 0., pos[1]],
                     [0., 0., 1., pos[2]],
                     [0., 0., 0., 1.]])


def scale_matrix(scale):
    return np.array([[scale[0], 0., 0., 0.],
                     [0., scale[1], 0., 0.],
                     [0., 0., scale[2], 0.],
                     [0., 0., 0., 1.]])


# Adapted to affine version from https://stackoverflow.com/questions/6802577/rotation-of-3d-vectors
def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac), 0.],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab), 0.],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc, 0.],
                     [0., 0., 0., 1.]])

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


# Compute the normals of the walls
#
# Notes:
# 1) Seed the first two lines with the correct normal
# 2) For each line, find all connectors and rotate normal accordingly
# 3) If no connector is found, assume must be interior wall, if vertical, normal is -1 if horizontal, normal is -1


# The outside wall is a connected graph so we can traverse it and induce the first normal to derive the rest
# Inner walls will have to be induced by making the assumption that the next line without a normal must be
# a vertical line (lines = verticals + horizontals) and due to the scan order, must be a wall facing the left because
# it must be an internal wall


class Generator:
    def __init__(self):
        self.horizontals = []
        self.verticals = []
        self.lines = []
        self.normals = []
        self.size = (0, 0)

    def clear(self):
        self.horizontals = []
        self.verticals = []
        self.lines = []
        self.normals = []
        self.size = (0, 0)

    def process_image(self, image_path="assets/Level 2 floor plan walls.png"):
        self.clear()

        image = Image.open(image_path)
        px = image.load()
        print("Image Size:", image.size)
        self.size = image.size

        for row in range(image.size[1]):
            if row % 10 == 0 and row != 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            if row % 500 == 0 and row != 0:
                print('#')
            for col in range(image.size[0]):
                if px[col, row][0] != 255:
                    verts = list(filter(lambda l: l[0] == col and l[1][1] == row - 1, self.verticals))
                    horzs = list(filter(lambda l: l[1] == row and l[0][1] == col - 1, self.horizontals))

                    if len(verts) == 0 and len(horzs) == 0:
                        self.verticals.append([col, [row, row]])  # Add a pixel
                        self.horizontals.append([[col, col], row])
                    else:
                        if len(verts) > 0:
                            verts[0][1][1] += 1
                        else:
                            self.verticals.append([col, [row, row]])

                        if len(horzs) > 0:
                            horzs[0][0][1] += 1
                        else:
                            self.horizontals.append([[col, col], row])

        # Remove all single-pixel lines
        clean_horizontals = list(filter(lambda h: h[0][0] != h[0][1], self.horizontals))
        clean_verticals = list(filter(lambda v: v[1][0] != v[1][1], self.verticals))

        self.horizontals = clean_horizontals
        self.verticals = clean_verticals
        self.lines = self.verticals + self.horizontals
        self.normals = [[]]*len(self.lines)

        self.compute_normals()

    def find_adjoining(self, line):
        if is_horizontal(line):  # [ [x_min, x_max], y ], [ x, [y_min, y_max] ]
            return list(filter(lambda l: (line[1] == l[1][0] or line[1] == l[1][1]) and
                                         (line[0][0] == l[0] or line[0][1] == l[0]), self.verticals))
        elif is_vertical(line):  # [ x, [y_min, y_max] ], [ [x_min, x_max], y ]
            return list(filter(lambda l: (line[0] == l[0][0] or line[0] == l[0][1]) and
                                         (line[1][0] == l[1] or line[1][1] == l[1]), self.horizontals))
        else:
            print("Error, not a valid vertical or horizontal line")
            return []

    def compute_normals(self, starting_normal=[+1, 0]):
        curr_line = self.lines[0]
        self.normals[0] = starting_normal
        curr_normal = starting_normal

        while curr_line:
            neighbours = self.find_adjoining(curr_line)
            nidx = list(map(lambda x: self.lines.index(x), neighbours))

            # print("Curr line:", curr_line, "Curr N:", curr_normal, "Neighbours:", neighbours, "Neighbour Ids:", nidx)

            if not self.normals[nidx[0]]:
                self.normals[nidx[0]] = compute_normal(curr_line, curr_normal, self.lines[nidx[0]])
                curr_normal = self.normals[nidx[0]]
                curr_line = self.lines[nidx[0]]
            elif len(nidx) > 1 and not self.normals[nidx[1]]:
                self.normals[nidx[1]] = compute_normal(curr_line, curr_normal, self.lines[nidx[1]])
                curr_normal = self.normals[nidx[1]]
                curr_line = self.lines[nidx[1]]
            else:
                # Search for the next line without a normal, it must, by inference, be a vertical line to the left and
                # therefore has a normal of [-1,0], repeat until no more vertical lines with null normals remain
                idx = self.normals.index([])
                curr_line = self.lines[idx] if is_vertical(self.lines[idx]) else None
                if curr_line:
                    curr_normal = [-1, 0]
                    self.normals[idx] = [-1, 0]

    def render_to_image(self, filename="assets/output.png", normal_len=5):
        img = Image.new('RGB', self.size, color='black')
        px = img.load()

        for h in self.horizontals:
            for col in range(h[0][0], h[0][1] + 1):
                px[col, h[1]] = (255, 0, 0)

        for v in self.verticals:
            for row in range(v[1][0], v[1][1] + 1):
                px[v[0], row] = (255, 0, 0)

        # Draw the normals

        for l in range(len(self.lines)):
            if is_horizontal(self.lines[l]):
                nor = self.normals[l]
                lne = self.lines[l]
                # print(nor, lne)
                hw = (lne[0][0] + lne[0][1]) / 2
                if nor != [] and nor[1] == 1:
                    for row in range(lne[1], lne[1] + normal_len):
                        px[hw, row] = (0, 0, 255)
                elif nor != [] and nor[1] == -1:
                    for row in range(lne[1] - normal_len, lne[1]):
                        px[hw, row] = (0, 0, 255)
            elif is_vertical(self.lines[l]):
                nor = self.normals[l]
                lne = self.lines[l]
                # print(nor, lne)
                hh = (lne[1][0] + lne[1][1]) / 2
                if nor != [] and nor[0] == 1:
                    for col in range(lne[0], lne[0] + normal_len):
                        px[col, hh] = (0, 0, 255)
                elif nor != [] and nor[0] == -1:
                    for col in range(lne[0] - normal_len, lne[0]):
                        px[col, hh] = (0, 0, 255)

        img.save(filename, "PNG")

    def export_to_object(self, filename="assets/output.obj"):
        # Normalise to the image size taking the longer axis as the dimension for the model
        dim = max(self.size[0], self.size[1])
        inv_dim = 1./dim

        planes = []         # len(planes) = len(lines)




