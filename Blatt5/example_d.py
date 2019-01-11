from nonlinear import NonLinear
from animator import Animator
import numpy as np

RG = NonLinear()

RG.dt = 0.02
RG.times = np.arange(-10, 10, RG.dt)

RG.dz = 0.02
RG.z = np.arange(0, 10, RG.dz)

RG.eta = 1
RG.a0 = RG.eta / np.cosh(RG.times * RG.eta)

RG.integrate()

anim = Animator(RG)
anim.start_animation()
