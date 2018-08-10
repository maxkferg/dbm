# Communicates via multiprocessing.Queue to the Tkinter display window

import multiprocessing as mp
from time import sleep
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.PathfinderWindow import pathfinder_executor


class MPQueue:
    def __init__(self):
        """Simply initialises the MPQueue, really a composite of two multiprocessing.Queue() objects with some light
        wrapping to make it easier to work with."""
        mp.set_start_method('spawn')
        self.send_q = mp.Queue()
        self.resp_q = mp.Queue()
        self.proc = None

    def run(self, floors_file, walls_file):
        self.proc = mp.Process(target=pathfinder_executor, args=(self.send_q, self.resp_q, floors_file, walls_file))
        self.proc.start()

    def join(self):
        self.proc.join()

    def command_move(self, car_pos):
        pass

    def command_turn(self, car_orn):
        pass

    def command_shutdown(self):
        pass

    def command_scale(self):
        pass

    def read_screensize(self):
        pass


if __name__ == '__main__':
    floors_file = "../output/test2_floors.obj"
    walls_file = "../output/test2_walls.obj"

    pf_queue = MPQueue()
    pf_queue.run(floors_file, walls_file)

    pf_queue.command_move([100, 100])
    sleep(.16)
    pf_queue.command_move([200, 200])
    sleep(.16)

    pf_queue.join()