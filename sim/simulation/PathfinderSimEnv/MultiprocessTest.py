# This is a small test of running two separate processes and initiating IPC between the two.

from multiprocessing import Process, Queue
import multiprocessing as mp
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '.'))
from BasicGUI import rand_pos, rand_orn
from time import sleep, clock
import math
from Math2D import rotate, rot_vec, add, sub, dot, perp, mul, scale_bias, make_polar, lerp, test_intersection, is_ccw


def gui_process(q):
    # Note: must import Tkinter *after* forking!!!
    sys.path.insert(1, os.path.join(sys.path[0], '.'))
    from tkinter import Tk
    from BasicGUI import DisplayWindow

    root = Tk()
    floors_file = '../../assets/output_floors.obj'
    walls_file = '../../assets/output_walls.obj'

    my_gui = DisplayWindow(root, q, floors_file, walls_file)
    my_gui.on_update()

    root.mainloop()


def cmd_move_car(q, target_vec):
    q.put(["move", target_vec])


def cmd_turn_car(q, target_orn):
    q.put(["turn", target_orn])


def test_fnc(q):
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

        cmd_move_car(q, lerp(src_pos, trg_pos, pos_dur[1]/pos_dur[0]))
        cmd_turn_car(q, lerp(src_orn, trg_orn, orn_dur[1]/orn_dur[0]))

        sleep(.016)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    queue = mp.Queue()
    p = Process(target=gui_process, args=(queue,))
    p.start()
    test_fnc(queue)
    p.join()
