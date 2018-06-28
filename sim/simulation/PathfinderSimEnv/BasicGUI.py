import tkinter as tk
from tkinter import Tk, Label, Canvas
from PIL import ImageTk, Image
from random import randint
import time

class DisplayWindow:
    def __init__(self, master):
        self.master = master
        master.title("PathfinderSim Display Window")
        master.geometry("512x512")

        self.canvas = Canvas(master, width=512, height=512)
        self.canvas.pack()

        self.img = ImageTk.PhotoImage(image=Image.open('/Users/otgaard/Development/dbm/sim/assets/test2.png'))
        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)

        self.ball_pos = [[randint(0,5), randint(0,5)], [randint(0,5), randint(0,5)]]
        self.balls = [
            self.canvas.create_oval(250, 250, 270, 270, fill="red"),
            self.canvas.create_oval(250, 250, 270, 270, fill="blue")
        ]

    def update_display(self):
        for i in range(len(self.balls)):
            delta_x = randint(-1, 1)
            delta_y = randint(-1, 1)
            self.canvas.move(self.balls[i], delta_x, delta_y)

        self.canvas.after(50, self.update_display)


root = Tk()
my_gui = DisplayWindow(root)

my_gui.update_display()

root.mainloop()
