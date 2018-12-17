from nonlinear import NonLinear
from animator import Animator
import numpy as np

RG = NonLinear()

RG.dt = 0.01
RG.times = np.arange(-50, 50, RG.dt)

RG.dz = 0.0001
RG.z = np.arange(0, 1, RG.dz)

# RG.T0 = 7
# RG.a0 = np.exp(-RG.times**2 / RG.T0**2)
size = len(RG.times)
RG.a0 = np.zeros(size)
RG.a0[np.where(abs(RG.times) < .5)] = 0.5 * np.pi

RG.integrate()

anim = Animator(RG)
anim.start_animation()
