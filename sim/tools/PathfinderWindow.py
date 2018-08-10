from tkinter import Tk, Canvas
from random import randint, random
import math
import sys, os
import multiprocessing as mp

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.OBJModel import OBJModel
import tools.Math2D as m2d
from tools.TileGrid import TileGrid


CAR_SCALE = .5
CAR_BODY = [[CAR_SCALE*-20, CAR_SCALE*-12.5], [CAR_SCALE*20, CAR_SCALE*12.5]]
CAR_FRONT = [[CAR_SCALE*-5, CAR_SCALE*-10.], [CAR_SCALE*5, CAR_SCALE*10]]
FRONT_TRANS = [CAR_SCALE*10, 0]  # Relative translation of the front to the body (tkinter doesn't have a scene graph)
RAY_LINE = [CAR_SCALE*100, 0]    # A line of 100 pixels long


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

        # Multiprocessing resources
        self.send_q = send_q        # The queue in which commands arrive for processing
        self.resp_q = resp_q        # The queue used to send responses to queries
        self.shutdown_flag = False  # Used in mp mode to shutdown the system

        # Model Resources
        self.wall_model = None      # The loaded OBJ file mdoel for the walls
        self.wall_bound = None      # The wall AABB world bound
        self.tile_grid = None       # TileGrid instance

        self.car_pos = [0, 0]       # The position of the car in screen coordinates
        self.car_orn = 0            # The orientation of the car in screen space
        self.car_rays = 12          # Number of rays to simulate
        self.ray_dtheta = 2.*math.pi/self.car_rays

        if not self.setup_window():
            print("Failed to setup Tkinter Display")
            return

        if not self.setup(floors, walls):
            print("Failed to initialise Display Model")
            return

        self.on_update()
        self.master.mainloop()

    def setup_window(self):
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

        return True

    def build_tiles(self, floor_file):
        self.tile_grid = TileGrid(floor_file)
        if not self.tile_grid.is_valid():
            print("Failed to intialize TileGrid, aborting")
            return False

        return True

    def on_update(self):
        """The on_update method reads the input command queue and updates the display with the new rendering"""
        pass

    def on_resize(self, event):
        """Resizes the Tkinter display window"""
        pass

    def process_commands(self):
        """The process_commands method reads input commands from the input command queue.  Note that the input command
        queue is threadsafe and therefore may be accessed from other concurrent threads."""
        pass

    def clear_map(self):
        """The Tkinter window only draws the map during initialisation, clear the map to draw the map again."""
        pass

    def draw_map(self):
        """The draw_map routine should not need to be called except via the clear_map function."""
        pass

    def build_car(self):
        """Builds the Tkinter objects for displaying the car.  This must be called prior to draw_car."""
        pass

    def draw_car(self):
        """Draws the car each frame by updating the scene graph created by build_car."""
        pass

    def has_response(self):
        """Returns whether or not there is a response for a read_var command in the response queue."""
        pass

    def read_response(self):
        """Returns a response from the response queue."""
        pass

    def shutdown(self):
        """Shutdown the Tkinter application and resources."""
        pass

    def on_destroy(self, arg):
        """Event triggered by closing window, etc. used to kill the thread and clean up the Tkinter resources."""
        pass

    def cmd_move_car(self, target_vec):
        self.send_q.put(["move", target_vec])

    def cmd_turn_car(self, target_orn):
        self.send_q.put(["turn", target_orn])

    def cmd_read_var(self, variable="car_pos"): # car_pos, car_orn, or screen (dims)
        self.send_q.put(["read", variable])


if __name__ == '__main__':
    pfw = PathfinderWindow(mp.Queue(), mp.Queue(), '../output/test2_floors.obj', '../output/test2_walls.obj', True)
