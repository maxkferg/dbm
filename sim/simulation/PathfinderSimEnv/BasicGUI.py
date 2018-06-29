import tkinter as tk
from tkinter import Tk, Canvas, Button
from random import randint, random
import math
import struct
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from OBJParser import OBJParser


def rgb2hex(rgb):
    return "#" + hex(rgb[0])[2:].rjust(2, '0') + hex(rgb[1])[2:].rjust(2, '0') + hex(rgb[2])[2:].rjust(2, '0')


def rand_colour():
    return rgb2hex((randint(0, 255), randint(0, 255), randint(0, 255)))


def rotate(points, angle, centre):
    angle = math.radians(angle)
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


class DisplayWindow:
    def __init__(self, master, floor_file, walls_file):
        self.master = master
        master.title("PathfinderSim Display Window")
        master.geometry("512x600")

        # Parse the input files
        self.floors = OBJParser(floor_file)
        self.walls = OBJParser(walls_file)
        self.floors.parse()
        self.walls.parse()
        self.walls_AABB = self.walls.model_AABB()       # cache this
        self.floors_AABB = self.floors.model_AABB()     # cached

        self.canvas = Canvas(master, width=512, height=512)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind('<Configure>', self.on_resize)
        self.width = self.canvas.winfo_width()
        self.height = self.canvas.winfo_height()

        self.ball_pos = [[randint(0, 5), randint(0, 5)], [randint(0, 5), randint(0, 5)]]
        self.balls = [
            self.canvas.create_oval(250, 250, 270, 270, fill="red"),
            self.canvas.create_oval(250, 250, 270, 270, fill="blue")
        ]

        self.button = Button(master, text="Quit", command=self.shutdown)
        self.button.pack()

        self.car = self.draw_rot_rect([10, 10], [30, 10], rand_colour())

        self.draw_map()

    def clear_map(self):
        self.canvas.delete("all")
        self.ball_pos = [[randint(0, 5), randint(0, 5)], [randint(0, 5), randint(0, 5)]]
        self.balls = [
            self.canvas.create_oval(250, 250, 270, 270, fill="red"),
            self.canvas.create_oval(250, 250, 270, 270, fill="blue")
        ]
        self.draw_map()

    def draw_map(self):
        # We need to draw the OBJ file in 2D
        centre = [
            (self.walls_AABB[1][0] - self.walls_AABB[0][0])/2 + self.walls_AABB[0][0],
            (self.walls_AABB[1][1] - self.walls_AABB[0][1])/2 + self.walls_AABB[0][1]
        ]

        bias = [self.width/2 - centre[0]*self.width, self.height/2 - centre[1]*self.height]
        scale = [self.width, self.height]

        def scale_bias(v, s, b): return [v[0]*s[0]+b[0], v[1]*s[1]+b[1]]

        for wall in range(int(self.walls.get_prim_count())):       # We actually want the quads and we know they're paired
            prim = self.walls.get_prim(wall)

            if len(prim) != 3: continue

            A = self.walls.get_position(prim[0])[:-1]
            B = self.walls.get_position(prim[1])[:-1]

            P0 = scale_bias(A, scale, bias)
            P1 = scale_bias(B, scale, bias)

            print(P0, P1)

            self.canvas.create_line(P0[0], P0[1], P1[0], P1[1], fill="red")

        for floor in range(int(self.floors.get_prim_count())):
            prim = self.floors.get_prim(floor)

            if len(prim) != 3: continue

            A = self.floors.get_position(prim[0])[:-1]
            B = self.floors.get_position(prim[2])[:-1]

            P0 = scale_bias(A, scale, bias)
            P1 = scale_bias(B, scale, bias)

            print(P0, P1)

            self.canvas.create_rectangle(P0[0], P0[1], P1[0], P1[1], fill=rand_colour())

    def draw_rot_rect(self, A, B, colour):
        verts = [A, [A[0], B[1]], [B[0], A[1]], B]
        centre = [(B[0] - A[0])/2, (B[1] - A[1])/2]
        return self.canvas.create_polygon(rotate(verts, (random()*math.pi*2), centre), fill=colour)

    def shutdown(self):
        self.master.destroy()

    def on_update(self):
        for i in range(len(self.balls)):
            delta_x = randint(-1, 1)
            delta_y = randint(-1, 1)
            self.canvas.move(self.balls[i], delta_x, delta_y)

        self.canvas.after(50, self.on_update)

    def on_resize(self, event):
        print("Resize:", event.width, event.height)
        print("Canvas size:", self.canvas.winfo_width(), self.canvas.winfo_height())

        self.width = self.canvas.winfo_width()
        self.height = self.canvas.winfo_height()

        self.clear_map()

        self.canvas.tag_raise(self.balls[0])
        self.canvas.tag_raise(self.balls[1])
        self.canvas.tag_raise(self.car)

root = Tk()
floors_file = '/Users/otgaard/Development/dbm/sim/assets/output_floors.obj'
walls_file = '/Users/otgaard/Development/dbm/sim/assets/output_walls.obj'

my_gui = DisplayWindow(root, floors_file, walls_file)
my_gui.on_update()

root.mainloop()
