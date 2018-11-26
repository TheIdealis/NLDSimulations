from nonlinear import NonLinear
from animator import Animator
import numpy as np

RG = NonLinear()

RG.dt = 0.1
RG.times = np.arange(-25, 25, RG.dt)

RG.dz = 0.01
RG.z = np.arange(0, 50, RG.dz)

RG.T0 = 7
RG.a0 = np.exp(-RG.times**2 / RG.T0**2)

RG.integrate('d')

anim = Animator(RG)
anim.start_animation()
