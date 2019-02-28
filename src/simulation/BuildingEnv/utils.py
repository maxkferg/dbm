import os
import cv2
import gym
import time
import math
import random
import colorsys
import numpy as np
from random import random, randint



def get_translation_matrix(dx, dy):
    """
    Returns a numpy affine transformation matrix for a 2D translation of
    (dx, dy)
    """
    return np.matrix([[1, 0, dx], [0, 1, dy], [0, 0, 1]])


def rotate_image(image, angle):
    """
    Rotates the given image about it's centre
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])
    trans_mat = np.identity(3)

    w2 = image_size[0] * 0.5
    h2 = image_size[1] * 0.5

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    tl = (np.array([-w2, h2]) * rot_mat_notranslate).A[0]
    tr = (np.array([w2, h2]) * rot_mat_notranslate).A[0]
    bl = (np.array([-w2, -h2]) * rot_mat_notranslate).A[0]
    br = (np.array([w2, -h2]) * rot_mat_notranslate).A[0]

    x_coords = [pt[0] for pt in [tl, tr, bl, br]]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in [tl, tr, bl, br]]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))
    new_image_size = (new_w, new_h)

    new_midx = new_w * 0.5
    new_midy = new_h * 0.5

    dx = int(new_midx - w2)
    dy = int(new_midy - h2)

    trans_mat = getTranslationMatrix2d(dx, dy)
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    result = cv2.warpAffine(image, affine_mat, new_image_size, flags=cv2.INTER_LINEAR)

    return result


def random_color():
    h,s,l = random(), 0.5 + random()/2.0, 0.6 + random()/10.0
    r,g,b = colorsys.hls_to_rgb(h,l,s)
    return r, g, b, 0.5


def flatten(nested):
    """Flatten a nested Python list"""
    return [item for sublist in nested for item in sublist]


def normalize(vec):
    return np.multiply(vec, np.linalg.norm(vec))


def rotate_vector(quat, vec):
    n1 = quat[0] * 2.
    n2 = quat[1] * 2.
    n3 = quat[2] * 2.
    n4 = quat[0] * n1
    n5 = quat[1] * n2
    n6 = quat[2] * n3
    n7 = quat[0] * n2
    n8 = quat[0] * n3
    n9 = quat[1] * n3
    n10 = quat[3] * n1
    n11 = quat[3] * n2
    n12 = quat[3] * n3
    result = [0, 0, 0]
    result[0] = (1. - (n5 + n6)) * vec[0] + (n7 - n12) * vec[1] + (n8 + n11) * vec[2]
    result[1] = (n7 + n12) * vec[0] + (1. - (n4 + n6)) * vec[1] + (n9 - n10) * vec[2]
    result[2] = (n8 - n11) * vec[0] + (n9 + n10) * vec[1] + (1. - (n4 + n5)) * vec[2]
    return result


def make_quaternion(axis, angle_in_radians):
    n = (axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
    rad = angle_in_radians * .5
    sin_theta = np.sin(rad)
    cos_theta = np.cos(rad)
    s = sin_theta / np.sqrt(n)
    return [s*axis[0], s*axis[1], s*axis[2], cos_theta]


def mul_quat(qA, qB):
    return [
        qA[3]*qB[0] + qA[0]*qB[3] + qA[1]*qB[2] - qA[2]*qB[2],
        qA[3]*qB[1] + qA[1]*qB[3] + qA[2]*qB[0] - qA[0]*qB[2],
        qA[3]*qB[2] + qA[2]*qB[3] + qA[0]*qB[1] - qA[1]*qB[0],
        qA[3]*qB[3] - qA[0]*qB[0] - qA[1]*qB[1] - qA[2]*qB[2],
    ]


def positive_component(array):
    """Replace postive values with zero"""
    return (np.abs(array) + array)/2


def scale_vec(scale, vec):
    return [scale*vec[0], scale*vec[1], scale*vec[2]]


def add_vec(vA, vB):
    return [vA[0]+vB[0], vA[1]+vB[1], vA[2]+vB[2]]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def rotation_change(theta1,theta2):
    """Compute the change in rotation, assuming small angle change"""
    dt = theta2 - theta1
    dt1 = dt - 2*math.pi
    dt2 = dt + 2*math.pi
    return find_nearest([dt, dt1, dt2], 0)



def gen_start_position(radius, floor):
    def lerp(A, B, t):
        return (1 - t)*A + t*B

    # Select a random quad and generate a position on the quad with sufficient distance from the walls
    quad_count = len(floor[1])/2

    done = False
    ret = []
    while not done:
        quad = randint(0, quad_count - 1)
        qidx = 2 * quad
        f0, f1, f2 = floor[1][qidx]
        v0, v1, v2 = floor[0][f0], floor[0][f1], floor[0][f2]

        for i in range(20):
            u, v = random(), random()
            x, y = lerp(v0[0], v1[0], u), lerp(v0[1], v2[1], v)

            lb = [v0[0], v0[1]]
            rt = [v1[0], v2[1]]

            if lb[0] < x-radius and x+radius < rt[0] and lb[1] < y-radius and y+radius < rt[1]:
                ret = [x, y]
                done = True
                break
    return ret
