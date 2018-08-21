# Communicates via multiprocessing.Queue to the Tkinter display window
# Running this file will open a PathfinderWindow, loading the passed map files, and will display the
# car using the passed movement and orientation commands.

import multiprocessing as mp
from time import sleep
import sys
import os
from queue import Empty

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
        self.send_q.put(['move', car_pos])

    def command_turn(self, car_orn):
        self.send_q.put(['turn', car_orn])

    def command_shutdown(self):
        self.send_q.put(['shutdown'])

    def command_scale(self, scale):
        self.send_q.put(['scale', scale])

    def read_variable(self, var):
        self.send_q.put(['read', var])
        try:
            resp = self.resp_q.get(True, 1)     # Timeout after a second
            return resp[1]
        except Empty:
            return None

    def read_pos(self):
        return self.read_variable('car_pos')

    def read_orn(self):
        return self.read_variable('car_orn')

    def read_screen(self):
        return self.read_variable('screen')


if __name__ == '__main__':
    floors_file = "../output/test2_floors.obj"
    walls_file = "../output/test2_walls.obj"

    pf_queue = MPQueue()
    pf_queue.run(floors_file, walls_file)
    sleep(1)

    pf_queue.command_scale([2, 2])
    sleep(.16)
    pf_queue.command_move([100, 100])
    sleep(.16)
    for i in range(100):
        pf_queue.command_move([100+i, 200])
        sleep(.16)

    print("POS:", pf_queue.read_pos())

    pf_queue.join()