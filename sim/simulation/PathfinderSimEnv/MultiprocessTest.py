# This is a small test of running two separate processes and initiating IPC between the two.

from multiprocessing import Process, Queue
import multiprocessing as mp
import sys
import os


def gui_process(name):
    print("RUNNING GUI")
    # Note: must import Tkinter *after* forking!!!
    sys.path.insert(1, os.path.join(sys.path[0], '.'))
    from tkinter import Tk
    from BasicGUI import DisplayWindow

    root = Tk()
    floors_file = '../../assets/output_floors.obj'
    walls_file = '../../assets/output_walls.obj'

    my_gui = DisplayWindow(root, floors_file, walls_file)
    my_gui.on_update()

    root.mainloop()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    p = Process(target=gui_process, args=('GUI',))
    p.start()
    p.join()



