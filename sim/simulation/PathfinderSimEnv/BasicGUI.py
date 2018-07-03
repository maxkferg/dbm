import tkinter as tk
from tkinter import Tk, Canvas, Button
from random import randint, random
import math
import sys, os
from queue import Queue
from threading import Thread
from time import sleep, clock

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from OBJParser import OBJParser


def rgb2hex(rgb):
    return "#" + hex(rgb[0])[2:].rjust(2, '0') + hex(rgb[1])[2:].rjust(2, '0') + hex(rgb[2])[2:].rjust(2, '0')


def rand_colour():
    return rgb2hex((randint(0, 255), randint(0, 255), randint(0, 255)))


def rand_pos(min=(0, 0), max=(1, 1)):
    return [min[0] + (max[0] - min[0])*random(), min[1] + (max[1] - min[1])*random()]


def rand_orn(min=0, max=1):
    return min + (max - min)*random()

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


def scale_bias(v, s, b):
    return [v[0] * s[0] + b[0], v[1] * s[1] + b[1]]


CAR_BODY = [[-20, -12.5], [20, 12.5]]
CAR_FRONT = [[-5, -10.], [5, 10]]
FRONT_TRANS = [10, 0]                       # The relative translation of the front to the body (tkinter doesn't have a scene graph)
RAY_LINE = [100, 0]                         # A line of 100 pixels long


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

        # Intercept the destroy event so we can shutdown gracefully
        master.bind("<Destroy>", self.on_destroy)

        self.canvas = Canvas(master, width=512, height=512)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind('<Configure>', self.on_resize)
        self.width = self.canvas.winfo_width()
        self.height = self.canvas.winfo_height()

        self.car_pos = [100, 100]
        self.car_orn = 0
        self.car_rays = 12
        self.ray_dtheta = 2.*math.pi/self.car_rays

        # Async comms to class in separate thread
        self.command_q = Queue()
        self.response_q = Queue()
        self.shutdown_flag = False

        self.button = Button(master, text="Quit", command=self.shutdown)
        self.button.pack()

        self.draw_map()

        self.car = self.build_car(self.car_pos, 12)

    def clear_map(self):
        self.canvas.delete("all")
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
        return list(map(lambda x: [x[0]+trans[0], x[1]+trans[1]], verts))

    def flatten(self, verts):
        return [item for sublist in verts for item in sublist]

    def update_object_coords(self, obj, verts):
        self.canvas.coords(obj, self.flatten(verts))

    def rotate_polygon(self, AABB, rotation, centre):
        verts, centre2 = self.AABB_to_vertices(AABB)
        return rotate(verts, rotation, centre2 if not centre else centre)

    def build_car(self, position, rays):
        """Builds the mesh for the car and the view triangles for intersecting mesh geometry"""
        car = list()            # Two rectangles and rays - 1 triangles
        # car[0]
        car.append(self.canvas.create_polygon(self.AABB_to_vertices(CAR_BODY), fill="blue"))
        # car[1]
        car.append(self.canvas.create_polygon(self.AABB_to_vertices(CAR_FRONT), fill="red"))

        verts = self.translate_vertices(self.rotate_polygon(CAR_BODY, self.car_orn, [0, 0]), position)
        self.update_object_coords(car[0], verts)

        # We have to rotate the local transformation of the front-AABB
        verts = self.translate_vertices(self.rotate_polygon(CAR_FRONT, self.car_orn, [0, 0]), position)
        rot = rot_vec(FRONT_TRANS, self.car_orn)
        verts = list(map(lambda x: add(x, rot), verts))
        self.update_object_coords(car[1], verts)

        for i in range(rays):
            vec = rot_vec(RAY_LINE, i*self.ray_dtheta)
            verts = self.flatten([self.car_pos, add(self.car_pos, vec)])
            car.append(self.canvas.create_line(*verts, fill="black"))

        return car

    # Draw the car along with its view intersection triangles
    def draw_car(self):
        verts = self.translate_vertices(self.rotate_polygon(CAR_BODY, self.car_orn, [0, 0]), self.car_pos)
        self.update_object_coords(self.car[0], verts)

        # We have to rotate the local transformation of the front-AABB
        verts = self.translate_vertices(self.rotate_polygon(CAR_FRONT, self.car_orn, [0, 0]), self.car_pos)
        rot = rot_vec(FRONT_TRANS, self.car_orn)
        verts = list(map(lambda x: add(x, rot), verts))
        self.update_object_coords(self.car[1], verts)

        for i in range(2, 2 + self.car_rays):
            vec = rot_vec(RAY_LINE, self.car_orn + i*self.ray_dtheta)
            self.update_object_coords(self.car[i], [self.car_pos, add(self.car_pos, vec)])

    # Command interface for moving the vehicle
    # Note: the commands are synchronised via a queue because the GUI thread must run separately to the rest of
    # the simulation
    def cmd_move_car(self, target_vec):
        self.command_q.put(["move", target_vec])

    def cmd_turn_car(self, target_orn):
        self.command_q.put(["turn", target_orn])

    # Variables:
    # car_pos = the current position of the vehicle, [x, y]
    # car_orn = the current orientation of the vehicle, value
    # screen = the current screen size, [width, height]
    def cmd_read_var(self, variable="car_pos"):
        self.command_q.put(["read", variable])

    def get_response(self, block=True):
        return self.response_q.get(block)

    def has_response(self):
        return not self.response_q.empty()

    def shutdown(self):
        self.master.destroy()

    def on_destroy(self, arg):
        self.shutdown_flag = True
        self.response_q.put(["shutdown", self.shutdown_flag])

    def process_commands(self):
        while not self.command_q.empty():
            cmd = self.command_q.get()
            if cmd[0] == "move":
                #print("move:", cmd)
                self.car_pos = cmd[1]
            elif cmd[0] == "turn":
                #print("turn:", cmd)
                self.car_orn = cmd[1]
            elif cmd[0] == "read":
                #print("read", cmd)
                if cmd[1] == "car_pos":
                    self.response_q.put(["car_pos", self.car_pos])
                elif cmd[1] == "car_orn":
                    self.response_q.put(["car_orn", self.car_orn])
                elif cmd[1] == "screen":
                    self.response_q.put(["screen", [self.width, self.height]])
                elif cmd[1] == "shutdown":
                    self.response_q.put(["shutdown", self.shutdown])
                else:
                    print("Unknown command", cmd)

    def on_update(self):
        self.process_commands()
        self.draw_car()
        self.canvas.after(50, self.on_update)

    def on_resize(self, event):
        print("Resize:", event.width, event.height)
        print("Canvas size:", self.canvas.winfo_width(), self.canvas.winfo_height())

        self.width = self.canvas.winfo_width()
        self.height = self.canvas.winfo_height()

        self.response_q.put(["screen", [self.width, self.height]])

        self.clear_map()


root = Tk()
floors_file = '/Users/otgaard/Development/dbm/sim/assets/output_floors.obj'
walls_file = '/Users/otgaard/Development/dbm/sim/assets/output_walls.obj'

my_gui = DisplayWindow(root, floors_file, walls_file)
my_gui.on_update()


def lerp(A, B, u):
    if isinstance(A, (list,)):
        result = []
        for i in range(len(A)):
            result.append((1 - u) * A[i] + u * B[i])
        return result
    else:
        return (1 - u) * A + u * B


def test_thread_fnc():
    dim = [512, 512]
    # Move the vehicle according to random targets
    src_pos = rand_pos([0, 0], rand_pos(dim))
    src_orn = rand_orn(0., 2.*math.pi)
    trg_pos = rand_pos([0, 0], rand_pos(dim))
    trg_orn = rand_orn(0., 2. * math.pi)
    pos_dur = [.2, 0]
    orn_dur = [.2, 0]

    curr_t = clock()
    prev_t = clock()
    dt = curr_t - prev_t

    shutdown = False
    while not shutdown:
        dt = curr_t - prev_t
        prev_t = curr_t
        curr_t = clock()

        pos_dur[1] += dt
        orn_dur[1] += dt

        if pos_dur[1] > pos_dur[0]:
            src_pos = trg_pos
            trg_pos = rand_pos([0, 0], dim)
            print(src_pos, trg_pos)
            pos_dur[1] = 0

        if orn_dur[1] > orn_dur[0]:
            src_orn = trg_orn
            trg_orn = rand_orn(0, 2*math.pi)
            orn_dur[1] = 0

        my_gui.cmd_move_car(lerp(src_pos, trg_pos, pos_dur[1]/pos_dur[0]))
        my_gui.cmd_turn_car(lerp(src_orn, trg_orn, orn_dur[1]/orn_dur[0]))

        if my_gui.has_response():
            rsp = my_gui.get_response()
            if rsp[0] == "shutdown":
                shutdown = True
            if rsp[0] == "screen":
                dim = rsp[1]
            else:
                print(rsp)
        sleep(.016)


thread = Thread(target=test_thread_fnc)
thread.start()

root.mainloop()

thread.join()
