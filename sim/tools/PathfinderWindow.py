from random import randint, random
import math
import sys, os
import multiprocessing as mp
from PIL import ImageTk, Image

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.OBJModel import OBJModel
import tools.Math2D as m2d
from tools.TileGrid import TileGrid, compute_centre, AABB_to_vertices
import tools.TriangleRasteriser as tr


CAR_SCALE = .5
CAR_BODY = [[CAR_SCALE*-20, CAR_SCALE*-12.5], [CAR_SCALE*20, CAR_SCALE*12.5]]
CAR_FRONT = [[CAR_SCALE*-5, CAR_SCALE*-10.], [CAR_SCALE*5, CAR_SCALE*10]]
FRONT_TRANS = [CAR_SCALE*10, 0]  # Relative translation of the front to the body (tkinter doesn't have a scene graph)
RAY_LINE = [CAR_SCALE*100, 0]    # A line of 100 pixels long


def translate_vertices(verts, trans):
    return list(map(lambda x: [x[0]+trans[0], x[1]+trans[1]], verts))


def flatten(verts):
    return [item for sublist in verts for item in sublist]


def rotate_polygon(AABB, rotation, centre):
    verts, centre2 = AABB_to_vertices(AABB)
    return m2d.rotate(verts, rotation, centre2 if not centre else centre)


def pathfinder_executor(send_q, resp_q, floors_file, walls_file):
    pfw = PathfinderWindow(send_q, resp_q, floors_file, walls_file, False)
    pfw.run()


def scale_to_width(dims, width):
    height = (width * dims[1]) / dims[0]
    return (int(width), int(height))


def scale_to_height(dims, height):
    width = (height * dims[0]) / dims[1]
    return (int(width), int(height))


def scale_dims(dims, width, height):
    if width >= height:
        return scale_to_width(dims, width)
    else:
        return scale_to_height(dims, height)


def verts(AABB):
    return [[AABB[0][0], AABB[1][1]], AABB[1] , [AABB[1][0], AABB[0][1]], AABB[0]]


class PathfinderWindow:
    """This class displays a Tkinter window running in a separate process.  The Tkinter window contains a 2D top-down
    view of the model including the car, it's position, orientation, and visited grid blocks.

    Note: The grid will automatically be the smallest unit size (GCD).  Thus, when analysing the input model, the
    2D grid will be configured to be made up of blocks measuring the smallest unit as the width of the narrowest floor
    polygon by the height of the narrowest tile.  Thus, if we have two rects of 2 x 10, 4 x 2, the grid size will be set
    to 2 x 2 to ensure the entire floor can be covered with a uniform tile size.

    The GCD is computed in each axis to be the dimensions of the grid size.
    """
    def __init__(self, send_q, resp_q, floors, walls, is_test=True):
        # Tkinter must be imported only after the process has spawned
        from tkinter import Tk
        """Initialise the PathfinderWindow with the floors and walls, both being paths because the OBJ files are
        loaded into a separate process in a separate python interpreter to avoid mixing two different GUI event loops.
        Set is_test equal to True when wishing to control the car with the keyboard."""
        # Tkinter resources
        self.master = Tk()
        self.master.title("Grid Display")
        self.master.geometry("512x600")
        self.is_test = is_test      # Allow single-threaded operation using keyboard
        self.canvas = None
        self.width = None           # Screen space width (i.e. canvas, not whole window)
        self.height = None          # SS height
        self.timer_freq = 50        # The frequency of the redraw event (20 Hz)

        # Multiprocessing resources
        self.send_q = send_q        # The queue in which commands arrive for processing
        self.resp_q = resp_q        # The queue used to send responses to queries
        self.shutdown_flag = False  # Used in mp mode to shutdown the system

        # Model Resources
        self.wall_model = None      # The loaded OBJ file mdoel for the walls
        self.wall_bound = None      # The wall AABB world bound
        self.tile_grid = None       # TileGrid instance
        self.car = None

        self.car_pos = [0, 0]       # The position of the car in screen coordinates
        self.car_orn = 0            # The orientation of the car in screen space
        self.car_rays = 12          # Number of rays to simulate
        self.scale = [1, 1]         # The scale between the model and the display
        self.ray_dtheta = 2.*math.pi/self.car_rays

        self.tile_polygons = []     # Pre-built floor polygons
        self.images = []            # The source image, used to store visited pixels in image scale
        self.visited = []           # A set of PhotoImages, for each polygon, for displaying the visited pixels

        if not self.setup_window():
            print("Failed to setup Tkinter Display")
            return

        if not self.setup(floors, walls):
            print("Failed to initialise Display Model")
            return

    def run(self):
        self.on_update()
        self.master.mainloop()

    def setup_window(self):
        from tkinter import Canvas
        self.master.bind('<Destroy>', self.on_destroy)
        if self.is_test:
            self.master.bind('<Left>', lambda x: self.cmd_turn_car(self.car_orn - .1))
            self.master.bind('<Right>', lambda x: self.cmd_turn_car(self.car_orn + .1))
            self.master.bind('<Up>', lambda x: self.cmd_move_car(m2d.add(self.car_pos, m2d.mul(m2d.make_polar(self.car_orn), 5))))
            self.master.bind('<Down>', lambda x: self.cmd_move_car(m2d.add(self.car_pos, m2d.mul(m2d.make_polar(self.car_orn), -5))))

        self.canvas = Canvas(self.master, width=512, height=512)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind('<Configure>', self.on_resize)
        self.width = self.canvas.winfo_width()
        self.height = self.canvas.winfo_height()
        return True

    def setup(self, floor_file, wall_file):
        if not self.build_tiles(floor_file):
            return False

        self.wall_model = OBJModel(wall_file)
        if not self.wall_model.parse():
            return False

        self.wall_bound = self.wall_model.model_AABB()

        return True

    def build_tiles(self, floor_file):
        self.tile_grid = TileGrid(floor_file)
        if not self.tile_grid.is_valid():
            print("Failed to initialise TileGrid, aborting")
            return False

        self.tile_grid.build_grid()
        self.car = self.build_car(self.car_pos, self.car_rays)
        return True

    def on_update(self):
        """The on_update method reads the input command queue and updates the display with the new rendering"""
        self.process_commands()
        self.update_car()
        self.canvas.after(self.timer_freq, self.on_update)
        pass

    def on_resize(self, event):
        """Resizes the Tkinter display window"""
        self.width = self.canvas.winfo_width()
        self.height = self.canvas.winfo_height()
        self.resp_q.put(["screen", [self.width, self.height]])
        self.clear_map()
        pass

    def process_commands(self):
        """The process_commands method reads input commands from the input command queue.  Note that the input command
        queue is thread-safe and therefore may be accessed from other concurrent threads."""
        while not self.send_q.empty():
            print('msg')
            cmd = self.send_q.get()
            if cmd[0] == "move":
                self.car_pos = cmd[1]
            elif cmd[0] == "turn":
                self.car_orn = cmd[1]
            elif cmd[0] == "shutdown":
                self.master.destroy()
            elif cmd[0] == "scale":
                self.scale = cmd[1]
            elif cmd[0] == "read":
                if cmd[1] == "car_pos":
                    self.resp_q.put(["car_pos", self.car_pos])
                elif cmd[1] == "car_orn":
                    self.resp_q.put(["car_orn", self.car_orn])
                elif cmd[1] == "screen":
                    self.resp_q.put(["screen", [self.width, self.height]])
                elif cmd[1] == "shutdown":
                    self.resp_q.put(["shutdown", self.shutdown])
                else:
                    print("Unknown command", cmd)

    def clear_map(self):
        """The Tkinter window only draws the map during initialisation, clear the map to draw the map again."""
        self.canvas.delete("all")
        self.draw_map()
        self.car = self.build_car(self.car_pos, 12)

    def draw_map(self):
        """The draw_map routine should not need to be called except via the clear_map function."""
        centre = compute_centre(self.wall_bound)

        bias = [self.width/2 - centre[0]*self.width, self.height/2 - centre[1]*self.height]
        dims = self.tile_grid.get_map_dims()
        dims_i = [int(dims[0]), int(dims[1])]
        scale = [self.width/dims[0], self.height/dims[1]]
        print("DIMS:", dims, scale)
        self.tile_polygons.clear()

        first_run = len(self.visited) == 0

        if first_run:
            # The images are stored on a 1 to 1 pixel scale and are resized if the screen is resized.
            # TODO: Check very large images and test size reduction
            self.tile_grid.set_screen_scale([1, 1])  # Use [1, 1] and scale
            for floor in range(self.tile_grid.poly_count()):
                LB, TR = self.tile_grid.get_poly(floor)
                self.tile_polygons.append(verts([LB, TR]))
                # Create an image the same size as the rectangle and map pixels 1 to 1
                w = int(TR[0] - LB[0])
                h = int(LB[1] - TR[1])      # Y is flipped
                if w < 1 or h < 0:
                    continue

                img = Image.new("RGB", (w, h))

                pixels = [None] * (w * h)

                half = w*h/2
                A = m2d.rand_colour3()
                B = m2d.rand_colour3()
                for i in range(w*h):
                    pixels[i] = A if i < half else B
                img.putdata(pixels)
                self.images.append(img)
                photo = ImageTk.PhotoImage(image=img)
                self.visited.append(photo)
                self.canvas.create_image(int(LB[0])+w/2, int(LB[1])-h/2, image=photo)

            self.draw_map()                         # Do the scaling
        else:
            self.tile_grid.set_screen_scale(scale)  # Use [1, 1] and scale
            for floor in range(self.tile_grid.poly_count()):
                LB, TR = self.tile_grid.get_poly(floor)
                self.tile_polygons.append(verts([LB, TR]))
                w = int(TR[0] - LB[0])
                h = int(LB[1] - TR[1])      # Y is flipped
                img = self.images[floor]
                new_size = (int(scale[0] * img.width), int(scale[1] * img.height))
                print(new_size)
                img = img.resize(new_size, Image.NEAREST)
                photo = ImageTk.PhotoImage(image=img)
                self.visited[floor] = photo
                self.canvas.create_image(int(LB[0])+w/2, int(LB[1])-h/2, image=photo)

    def update_object_coords(self, obj, verts):
        self.canvas.coords(obj, flatten(verts))

    def build_car(self, position, rays):
        """Builds the Tkinter objects for displaying the car.  This must be called prior to draw_car."""
        car = list()
        car.append(self.canvas.create_polygon(AABB_to_vertices(CAR_BODY), fill="blue"))
        car.append(self.canvas.create_polygon(AABB_to_vertices(CAR_FRONT), fill="red"))

        verts = translate_vertices(rotate_polygon(CAR_BODY, self.car_orn, [0, 0]), position)
        self.update_object_coords(car[0], verts)

        # We have to rotate the local transformation of the front-AABB
        verts = translate_vertices(rotate_polygon(CAR_FRONT, self.car_orn, [0, 0]), position)
        rot = m2d.rot_vec(FRONT_TRANS, self.car_orn)
        verts = list(map(lambda x: m2d.add(x, rot), verts))
        self.update_object_coords(car[1], verts)

        ray_points = []
        for i in range(rays):
            vec = m2d.rot_vec(RAY_LINE, i*self.ray_dtheta)
            verts = flatten([self.car_pos, m2d.add(self.car_pos, vec)])
            ray_points.append(m2d.add(self.car_pos, vec))
            car.append(self.canvas.create_line(*verts, fill="blue"))

        i1 = len(ray_points) - 1
        for i0 in range(len(ray_points)):
            car.append(self.canvas.create_line(*[ray_points[i1], ray_points[i0]], fill="blue"))
            i1 = i0

        return car

    def update_car(self):
        """Updates the car each frame by updating the scene graph created by build_car."""
        verts = translate_vertices(rotate_polygon(CAR_BODY, self.car_orn, [0, 0]), self.car_pos)
        self.update_object_coords(self.car[0], verts)

        # We have to rotate the local transformation of the front-AABB
        verts = translate_vertices(rotate_polygon(CAR_FRONT, self.car_orn, [0, 0]), self.car_pos)
        rot = m2d.rot_vec(FRONT_TRANS, self.car_orn)
        verts = list(map(lambda x: m2d.add(x, rot), verts))
        self.update_object_coords(self.car[1], verts)

        ray_points = []
        for i in range(2, 2 + self.car_rays):
            vec = m2d.rot_vec(RAY_LINE, self.car_orn + i*self.ray_dtheta)
            self.update_object_coords(self.car[i], [self.car_pos, m2d.add(self.car_pos, vec)])
            ray_points.append(m2d.add(self.car_pos, vec))

        i1 = len(ray_points) - 1
        for i0 in range(self.car_rays):
            i = i0 + 2 + self.car_rays
            self.update_object_coords(self.car[i], [ray_points[i1], ray_points[i0]])
            i1 = i0

        self.visit_tiles(ray_points)

        for i in range(len(self.car)):
            self.canvas.tag_raise(self.car[i])


    # The new visit_tiles method must do the following:
    # 1) For each tile:
    #       a) Test if tri intersects tile
    #       b) if intersection:
    #            i) rasterise triangle
    def visit_tiles(self, ray_points):
        dims = self.tile_grid.get_map_dims()
        dims_i = [int(dims[0]), int(dims[1])]
        scale = [self.width/dims[0], self.height/dims[1]]

        # Test each rectangle that has not been seen against the ray triangles
        i1 = len(ray_points) - 1
        for i0 in range(self.car_rays):
            tri = [self.car_pos, ray_points[i1], ray_points[i0]]
            if not m2d.is_ccw(tri):
                tri.reverse()

            for tile in range(len(self.tile_polygons)):
                #if i0 == 0:
                #    print(tri, self.tile_polygons[tile])

                if m2d.test_intersection(tri, self.tile_polygons[tile]):
                    print("Intersection")
                    LB, TR = self.tile_grid.get_poly(tile)
                    w = int(TR[0] - LB[0])
                    h = int(LB[1] - TR[1])  # Y is flipped
                    img = self.images[tile]
                    pixels = list(img.getdata())
                    cx = int(LB[0] + w/2)
                    cy = int(LB[1] - h/2)

                    print("centre:", cx, cy, "car_pos:", self.car_pos)

                    def cb(coord):
                        idx = (coord[1] - cy)*w + (coord[0] + cx)
                        if 0 < idx < len(pixels):
                            print(idx, coord[0], coord[1])
                            pixels[idx] = (0, 0, 0)

                    itri = list(map(lambda x: [int(round(x[0] - w/2, 4)), int(round(x[1] - h/2, 4))], tri))
                    print("TRIANGLES:", tri, itri)

                    tr.rasterise(itri, cb)
                    img.putdata(pixels)

                    new_size = (int(scale[0] * img.width), int(scale[1] * img.height))
                    print(new_size)
                    img = img.resize(new_size, Image.NEAREST)
                    photo = ImageTk.PhotoImage(image=img)
                    self.visited[tile] = photo
                    id = self.canvas.create_image(cx, cy, image=photo)

            i1 = i0

    def has_response(self):
        """Returns whether or not there is a response for a read_var command in the response queue."""
        return not self.resp_q.empty()

    def read_response(self, block=True):
        """Returns a response from the response queue."""
        return self.resp_q.get(block)

    def shutdown(self):
        """Shutdown the Tkinter application and resources."""
        self.master.destroy()

    def on_destroy(self, arg):
        """Event triggered by closing window, etc. used to kill the thread and clean up the Tkinter resources."""
        self.shutdown_flag = True
        self.resp_q.put(["shutdown", self.shutdown_flag])

    def cmd_move_car(self, target_vec):
        print("move", target_vec)
        self.send_q.put(["move", target_vec])

    def cmd_turn_car(self, target_orn):
        print("turn", target_orn)
        self.send_q.put(["turn", target_orn])

    def cmd_read_var(self, variable="car_pos"): # car_pos, car_orn, or screen (dims)
        print("read")
        self.send_q.put(["read", variable])


if __name__ == '__main__':
    pfw = PathfinderWindow(mp.Queue(), mp.Queue(), '../output/test2_floors.obj', '../output/test2_walls.obj', True)
