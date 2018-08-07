from tkinter import Tk, Canvas
from random import randint, random
import math
import sys, os
from multiprocessing import Process


class PathfinderWindow:
    """This class displays a Tkinter window running in a separate thread.  The Tkinter window contains a 2D top-down
    view of the model including the car, it's position, orientation, and visited grid blocks.

    Note: The grid will automatically be the smallest unit size possible.  Thus, when analysing the input model, the
    2D grid will be configured to be made up of blocks measuring the smallest unit as the width of the narrowest floor
    polygon by the height of the narrowest tile.  Thus, if we have two rects of 2 x 10, 4 x 2, the grid size will be set
    to 2 x 2 to ensure the entire floor can be covered with a uniform tile size.
    """
    def __init__(self, floors, walls, is_test=True):
        """Initialise the PathfinderWindow with the floors and walls, both being paths because the OBJ files are
        loaded into a separate process in a separate python interpreter to avoid mixing two different GUI event loops.
        Set is_test equal to True when wishing to control the car with the keyboard."""
        self.master = Tk()
        self.on_update()

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

    def cmd_move_car(self, target):
        """Issue an async move command to the car.  Updates the current position of the car model in world space."""
        pass

    def cmd_turn_car(self, orn):
        """Issue an async rotation command to the car.  Updates the current orientation of the car model."""
        pass

    def cmd_read_var(self, variable="car_pos"):
        """Return the value of a variable asynchronously.
        variable can be "car_pos" - the position of the car.
                        "car_orn" = the orientation
                        "screen"  = the dimensions of the Tkinter display."""
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

    def on_destroy(self):
        """Event triggered by closing window, etc. used to kill the thread and clean up the Tkinter resources."""
        pass


