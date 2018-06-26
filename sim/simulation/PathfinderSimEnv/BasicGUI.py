import tkinter as tk
from tkinter import Tk, Label, Canvas
from PIL import ImageTk, Image


class DisplayWindow:
    def __init__(self, master):
        self.master = master
        master.title("PathfinderSim Display Window")
        master.geometry("256x256")

        self.canvas = Canvas(master, width=256, height=256)
        self.canvas.pack()

        self.img = ImageTk.PhotoImage(image=Image.open('/Users/otgaard/Development/dbm/sim/assets/test2.png'))
        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)


root = Tk()
my_gui = DisplayWindow(root)
root.mainloop()
