from tkinter import Tk, Label
from PIL import ImageTk, Image


class DisplayWindow:
    def __init__(self, master):
        self.master = master
        master.title("PathfinderSim Display Window")
        master.geometry("300x300")

        img = ImageTk.PhotoImage(Image.open('/Users/otgaard/Development/dbm/sim/assets/test2.png'))
        panel = Label(master, image=img)
        panel.pack(side="bottom", fill="both", expand="yes")


root = Tk()
my_gui = DisplayWindow(root)
root.mainloop()
