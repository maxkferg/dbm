import tkinter as tk
from tkinter import Tk, Canvas, Button
from PIL import ImageTk, Image
from random import randint
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from OBJParser import OBJParser

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

        self.canvas = Canvas(master, width=512, height=512)
        self.canvas.pack(fill="both", expand=True)

        # Resize to fit the window
        self.original = Image.open('/Users/otgaard/Development/dbm/sim/assets/test2.png')
        self.image = self.original.resize((512, 512))
        self.img = ImageTk.PhotoImage(image=self.image)

        self.bk = self.canvas.create_image(256, 256, image=self.img, anchor=tk.CENTER)
        self.canvas.bind('<Configure>', self.on_resize)

        self.ball_pos = [[randint(0, 5), randint(0, 5)], [randint(0, 5), randint(0, 5)]]
        self.balls = [
            self.canvas.create_oval(250, 250, 270, 270, fill="red"),
            self.canvas.create_oval(250, 250, 270, 270, fill="blue")
        ]

        self.button = Button(master, text="Quit", command=self.shutdown)
        self.button.pack()

    def draw_map(self):
        # We need to draw the OBJ file in 2D
        pass

    def shutdown(self):
        self.master.destroy()

    def on_update(self):
        for i in range(len(self.balls)):
            delta_x = randint(-1, 1)
            delta_y = randint(-1, 1)
            self.canvas.move(self.balls[i], delta_x, delta_y)

        self.canvas.after(50, self.on_update)

    def on_resize(self, event):
        print("Resize:", event.width, event.height)
        self.image = self.original.resize((event.width, event.height))
        self.img = ImageTk.PhotoImage(self.image)

        self.bk = self.canvas.create_image(event.width/2, event.height/2, image=self.img, anchor=tk.CENTER)
        for i in range(len(self.balls)):
            self.canvas.tag_raise(self.balls[i])


root = Tk()
floors_file = '/Users/otgaard/Development/dbm/sim/assets/output_floors.obj'
walls_file = '/Users/otgaard/Development/dbm/sim/assets/output_walls.obj'
my_gui = DisplayWindow(root, floors_file, walls_file)

my_gui.on_update()

root.mainloop()
