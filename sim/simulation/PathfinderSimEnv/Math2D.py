# Added all 2D vector maths here so we have one single place for it and added
# an implementation of the separating axis test for testing 2D polyhedra for intersection
# See: https://www.geometrictools.com/Documentation/MethodOfSeparatingAxes.pdf

import math


def rotate(points, angle, centre):
    cos_val = math.cos(angle)
    sin_val = math.sin(angle)
    cx, cy = centre
    new_points = []
    for x_old, y_old in points:
        x_old -= cx
        y_old -= cy
        x_new = x_old * cos_val - y_old * sin_val
        y_new = x_old * sin_val + y_old * cos_val
        new_points.append([x_new + cx, y_new + cy])
    return new_points


def rot_vec(vec, rot):
    st = math.sin(rot)
    ct = math.cos(rot)
    return [vec[0] * ct - vec[1] * st, vec[0] * st + vec[1] * ct]


def add(A, B):
    return [A[0] + B[0], A[1] + B[1]]


def sub(A, B):
    return [A[0] - B[0], A[1] - B[1]]


def mul(A, s):
    return [s*A[0], s*A[1]]


def scale_bias(v, s, b):
    return [v[0] * s[0] + b[0], v[1] * s[1] + b[1]]


def make_polar(orn):
    return [math.cos(orn), math.sin(orn)]


def dot(A, B):
    return A[0] * B[0] + A[1] * B[1]


def perp(A):
    return [A[1], -A[0]]


# pointset = [[x0,y0], [x1,y1], ...]
# D, P = [x,y], [x,y]
def which_side(pointset, D, P):
    pn = [0, 0]
    for i in range(len(pointset)):
        t = dot(D, sub(pointset[i], P))
        if t > 0: pn[0] += 1
        elif t < 0: pn[1] += 1
        if pn[0] > 0 and pn[1] > 0: return 0

    return 1 if pn[0] > 0 else -1


# polyA [ [x0, y0], [x1, y1], ... ] counter-clockwise ordered
# polyB ditto...
def test_intersection(polyA, polyB):
    i1 = len(polyA)-1
    for i0 in range(len(polyA)):
        D = perp(sub(polyA[i0], polyA[i1]))
        if which_side(polyB, D, polyA[0]) > 0:
            return False
        i1 = i0

    i1 = len(polyB)-1
    for i0 in range(len(polyB)):
        D = perp(sub(polyB[i0], polyB[i1]))
        if which_side(polyA, D, polyB[0]) > 0:
            return False
        i1 = i0

    return True
