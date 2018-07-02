import tkinter as tk
from tkinter import Tk, Canvas, Button
from random import randint, random
import math
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from OBJParser import OBJParser


def rgb2hex(rgb):
    return "#" + hex(rgb[0])[2:].rjust(2, '0') + hex(rgb[1])[2:].rjust(2, '0') + hex(rgb[2])[2:].rjust(2, '0')


def rand_colour():
    return rgb2hex((randint(0, 255), randint(0, 255), randint(0, 255)))


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


CAR_BODY = [[-20, -12.5], [20, 12.5]]
CAR_FRONT = [[-5, -10.], [5, 10]]
FRONT_TRANS = [7, -14.5]                       # The relative translation of the front to the body (tkinter doesn't have a scene graph)


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

        self.car_pos = [100, 100]
        self.car_orn = math.pi/3

        self.ball_pos = [[randint(0, 5), randint(0, 5)], [randint(0, 5), randint(0, 5)]]
        self.balls = [
            self.canvas.create_oval(250, 250, 270, 270, fill="red"),
            self.canvas.create_oval(250, 250, 270, 270, fill="blue")
        ]

        self.button = Button(master, text="Quit", command=self.shutdown)
        self.button.pack()

        self.draw_map()

        self.car = self.build_car(self.car_pos, 12)

    def clear_map(self):
        self.canvas.delete("all")
        self.ball_pos = [[randint(0, 5), randint(0, 5)], [randint(0, 5), randint(0, 5)]]
        self.balls = [
            self.canvas.create_oval(250, 250, 270, 270, fill="red"),
            self.canvas.create_oval(250, 250, 270, 270, fill="blue")
        ]
        self.draw_map()
        self.car = self.build_car(self.car_pos, 12)

    def draw_map(self):
        # We need to draw the OBJ file in 2D
        centre = [
            (self.walls_AABB[1][0] - self.walls_AABB[0][0])/2 + self.walls_AABB[0][0],
            (self.walls_AABB[1][1] - self.walls_AABB[0][1])/2 + self.walls_AABB[0][1]
        ]

        bias = [self.width/2 - centre[0]*self.width, self.height/2 - centre[1]*self.height]
        scale = [self.width, self.height]

        def scale_bias(v, s, b): return [v[0]*s[0]+b[0], v[1]*s[1]+b[1]]

        for floor in range(int(self.floors.get_prim_count())):
            prim = self.floors.get_prim(floor)

            if len(prim) != 3: continue

            A = self.floors.get_position(prim[0])[:-1]
            B = self.floors.get_position(prim[2])[:-1]

            P0 = scale_bias(A, scale, bias)
            P1 = scale_bias(B, scale, bias)

            self.canvas.create_rectangle(P0[0], P0[1], P1[0], P1[1], fill=rand_colour())

        for wall in range(int(self.walls.get_prim_count())):
            prim = self.walls.get_prim(wall)

            if len(prim) != 3: continue

            A = self.walls.get_position(prim[0])[:-1]
            B = self.walls.get_position(prim[1])[:-1]

            P0 = scale_bias(A, scale, bias)
            P1 = scale_bias(B, scale, bias)

            self.canvas.create_line(P0[0], P0[1], P1[0], P1[1], fill="red", width=2)

    def AABB_to_vertices(self, AABB):
        verts = [AABB[0], [AABB[1][0], AABB[0][1]], AABB[1], [AABB[0][0], AABB[1][1]]]
        centre = [(AABB[1][0] - AABB[0][0])/2, (AABB[1][1] - AABB[0][1])/2]
        return verts, centre

    def translate_vertices(self, verts, trans):
        verts = list(map(lambda x: [x[0]+trans[0], x[1]+trans[1]], verts))
        return [item for sublist in verts for item in sublist]

    def update_object_coords(self, obj, verts):
        self.canvas.coords(obj, verts)

    def rotate_polygon(self, AABB, rotation):
        verts, centre = self.AABB_to_vertices(AABB)
        return rotate(verts, rotation, centre)

    def build_car(self, position, rays):
        """Builds the mesh for the car and the view triangles for intersecting mesh geometry"""
        car = list()            # Two rectangles and rays - 1 triangles
        car.append(self.canvas.create_polygon(self.AABB_to_vertices(CAR_BODY), fill="blue"))
        car.append(self.canvas.create_polygon(self.AABB_to_vertices(CAR_FRONT), fill="red"))

        verts = self.translate_vertices(self.rotate_polygon(CAR_BODY, self.car_orn), position)
        self.update_object_coords(car[0], verts)

        # We have to rotate the local transformation of the front-AABB
        offset = add(position, rot_vec(FRONT_TRANS, self.car_orn))
        verts = self.translate_vertices(self.rotate_polygon(CAR_FRONT, self.car_orn), offset)
        self.update_object_coords(car[1], verts)

        return car

    # Draw the car along with its view intersection triangles
    def draw_car(self, position):
        """The draw_car method continually recreates the polygons used to display the car and view area because
        no other form of rotation is available in tkinter canvas"""
        pass

    def shutdown(self):
        self.master.destroy()

    def on_update(self):
        for i in range(len(self.balls)):
            delta_x = randint(-1, 1)
            delta_y = randint(-1, 1)
            self.canvas.move(self.balls[i], delta_x, delta_y)

        verts = self.translate_vertices(self.rotate_polygon(CAR_BODY, self.car_orn), self.car_pos)
        self.update_object_coords(self.car[0], verts)

        # We have to rotate the local transformation of the front-AABB
        offset = add(self.car_pos, rot_vec(FRONT_TRANS, self.car_orn))
        verts = self.translate_vertices(self.rotate_polygon(CAR_FRONT, self.car_orn), offset)
        self.update_object_coords(self.car[1], verts)

        self.car_orn += 0.01

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
