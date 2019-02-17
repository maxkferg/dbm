#!/usr/bin/env python
from __future__ import print_function
import math
import sys, gym, time
import numpy as np
import tkinter
from PIL import Image, ImageTk
from gym.envs.registration import registry
from simulation.BuildingEnv import BuildingEnv
tkinter.NoDefaultRoot()

RENDER_WIDTH = 640
RENDER_HEIGHT = 480
RENDER_SIZE = (RENDER_HEIGHT, RENDER_WIDTH)


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


register(id='BuildingEnv-v0',
    entry_point='simulation.BuildingEnv:BuildingEnv',
    reward_threshold=.5)

env = gym.make('BuildingEnv-v0')
env.debug = 2



class ViewWindow():
    times=1
    timestart=time.clock()

    def __init__(self, mapw, width, height):
        self.action = [0,0]
        self.width = width
        self.height = height
        self.root = tkinter.Tk()
        self.frame = tkinter.Frame(self.root, width=width, height=height)
        self.frame.pack()
        self.canvas = tkinter.Canvas(self.frame, width=width, height=height)
        self.canvas.place(x=-2, y=-2)
        self.map = mapw

    def start(self):
        self.root.after(0, self.step) # INCREASE THE 0 TO SLOW IT DOWN
        self.root.mainloop()

    def render(self, pixels):
        self.im = Image.fromarray(pixels)
        self.photo = ImageTk.PhotoImage(master=self.root, image=self.im)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

    def step(self):
        obser, r, done, info = env.step(self.action)
        #print("Observation:", obser)

        self.action[0] = obser[0]/math.pi / 4
        self.action[1] = 0.6
        self.render(env.render(mode="rgb_array", width=self.width, height=self.height))
        self.map.render(env.render_map(mode="rgb_array", width=self.width, height=self.height))
        self.root.update()
        self.times+=1
        if self.times%33==0:
            print("%.02f FPS"%(self.times/(time.clock()-self.timestart)))
        self.root.after(10, self.step)
        time.sleep(.5)
        if done:
            print("--- Resetting ---")
            env.reset()



class MapWindow():
    times = 1
    timestart = time.clock()

    def __init__(self, width, height):
        self.action = [0,0]
        self.width = width
        self.height = height
        self.root = tkinter.Tk()
        self.frame = tkinter.Frame(self.root, width=width, height=height)
        self.frame.pack()
        self.canvas = tkinter.Canvas(self.frame, width=width, height=height)
        self.canvas.place(x=-2, y=-2)

    def start(self):
        self.root.after(0, self.step) # INCREASE THE 0 TO SLOW IT DOWN
        self.root.mainloop()

    def render(self, pixels):
        self.im = Image.fromarray(pixels)
        self.photo = ImageTk.PhotoImage(master=self.root, image=self.im)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)


mapw = MapWindow(RENDER_WIDTH, RENDER_HEIGHT)
view = ViewWindow(mapw, RENDER_WIDTH, RENDER_HEIGHT)
view.start()
