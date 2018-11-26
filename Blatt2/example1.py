import numpy as np

from nonlinear import NonLinear
from animator1 import Animator

RG = NonLinear()

RG.dt = 0.1
RG.times = np.arange(-150, 150, RG.dt)

RG.dz = 0.02
RG.z = np.arange(0, 20, RG.dz)

T0 = 1.5
RG.a0 = T0 * np.exp(-RG.times**2 * T0)

RG.integrate('NLS')

anim = Animator(RG)
anim.start_animation()
