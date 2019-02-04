import os
import subprocess

from nonlinear import NonLinear
from animator import Animator
import numpy as np

RG = NonLinear()

RG.dx = 0.1
RG.dy = 0.1
x = np.arange(-10, 10, RG.dy)
y = np.arange(-10, 10, RG.dy)
RG.x, RG.y = np.meshgrid(x, y)


RG.dz = 0.02
RG.z = np.arange(0, 6, RG.dz)

# RG.eta = 0.25065
RG.eta = 0.3
RG.a0 = np.exp(- RG.eta * (RG.x**2 + RG.y**2))

RG.kappa = 1
RG.integrate()


filename = 'repulse.mp4'

anim = Animator(RG)
anim.start_animation(filename)

devnull = open(os.devnull, 'w')
subprocess.call(['mpv', filename], stdout=devnull, stderr=devnull)
