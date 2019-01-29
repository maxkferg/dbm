#!/usr/bin/env python
from __future__ import print_function

import sys, gym, time
import numpy as np
import tkinter
from PIL import Image, ImageTk
from gym.envs.registration import registry


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


register(id='SeekerSimEnv-v0',
     entry_point='simulation.SeekerSimEnv:SeekerSimEnv',
     reward_threshold=.5)


env = gym.make('SeekerSimEnv-v0')
env.debug = True


class mainWindow():
    times=1
    timestart=time.clock()

    def __init__(self):
        self.action = [0,0]
        self.root = tkinter.Tk()
        self.frame = tkinter.Frame(self.root, width=960, height=720)
        self.frame.pack()
        self.canvas = tkinter.Canvas(self.frame, width=960, height=720)
        self.canvas.place(x=-2, y=-2)
        self.root.after(0,self.start) # INCREASE THE 0 TO SLOW IT DOWN
        self.root.mainloop()

    def start(self):
        obser, r, done, info = env.step(self.action)
        print(obser,"-->")
        self.action[0] = obser[3]/10
        self.action[1] = 0.5
        colors = env.render(mode="rgb_array")

        self.im = Image.frombytes('L', (colors.shape[1], colors.shape[0]), colors.astype('b').tostring())
        self.im = Image.fromarray(colors)
        self.photo = ImageTk.PhotoImage(image=self.im)
        self.canvas.create_image(0,0,image=self.photo,anchor=tkinter.NW)
        self.root.update()
        self.times+=1
        if self.times%33==0:
            print("%.02f FPS"%(self.times/(time.clock()-self.timestart)))
        self.root.after(10,self.start)
        if done:
            env.reset()


mainWindow()
