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

        self.width = 512
        self.height = 512

        # Parse the input files
        self.floors = OBJParser(floor_file)
        self.walls = OBJParser(walls_file)
        self.floors.parse()
        self.walls.parse()

        self.canvas = Canvas(master, width=self.width, height=self.height)
        self.canvas.pack(fill="both", expand=True)

        # Resize to fit the window
        #self.original = Image.open('/Users/otgaard/Development/dbm/sim/assets/test2.png')
        #self.image = self.original.resize((512, 512))
        #self.img = ImageTk.PhotoImage(image=self.image)

        #self.bk = self.canvas.create_image(256, 256, image=self.img, anchor=tk.CENTER)
        self.canvas.bind('<Configure>', self.on_resize)

        self.ball_pos = [[randint(0, 5), randint(0, 5)], [randint(0, 5), randint(0, 5)]]
        self.balls = [
            self.canvas.create_oval(250, 250, 270, 270, fill="red"),
            self.canvas.create_oval(250, 250, 270, 270, fill="blue")
        ]

        self.button = Button(master, text="Quit", command=self.shutdown)
        self.button.pack()

        self.draw_map()

    def clear_map(self):
        self.canvas.delete("all")

    def draw_map(self):
        # We need to draw the OBJ file in 2D
        bias = [self.width/2, self.height/2]
        scale = [self.width, self.height]
        scale_bias = lambda v, s, b: [v[0]*s[0]+b[0], v[1]*s[1]+b[1]]

        print("Walls", int(self.walls.get_prim_count()/2))

        for wall in range(int(self.walls.get_prim_count()/2)):       # We actually want the quads and we know they're paired
            prim = self.walls.get_prim(wall)
            print(prim)
            if len(prim) != 3: continue

            A = self.walls.get_position(prim[0])[:-1]
            B = self.walls.get_position(prim[1])[:-1]

            P0 = scale_bias(A, scale, bias)
            P1 = scale_bias(B, scale, bias)

            print(P0, P1)

            self.canvas.create_line(P0[0], P0[1], P1[0], P1[1], fill="red")

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
        print("Canvas size:", self.canvas.winfo_width(), self.canvas.winfo_height())

        self.width = self.canvas.winfo_width()
        self.height = self.canvas.winfo_height()

        #self.image = self.original.resize((self.width, self.height))
        #self.img = ImageTk.PhotoImage(self.image)
        #self.bk = self.canvas.create_image(self.width/2, self.height/2, image=self.img, anchor=tk.CENTER)

        for i in range(len(self.balls)):
            self.canvas.tag_raise(self.balls[i])


root = Tk()
floors_file = '/Users/otgaard/Development/dbm/sim/assets/output_floors.obj'
walls_file = '/Users/otgaard/Development/dbm/sim/assets/output_walls.obj'

my_gui = DisplayWindow(root, floors_file, walls_file)
my_gui.on_update()

root.mainloop()
