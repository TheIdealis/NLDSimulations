import numpy as np

from linear import Linear
from animator import Animator

RG = Linear()

RG.A0 = 1
# RG.omega = 1.41
RG.omega = 3.

RG.dt = .02
RG.T = 2 * np.pi / RG.omega
periods = 3
RG.periodic_time = periods * RG.T
RG.times = np.arange(-RG.periodic_time, RG.periodic_time, RG.dt)[:-1]

RG.dz = 0.005
RG.z = np.arange(0, 5, RG.dz)


RG.a0 = np.exp(- 1j * RG.omega * RG.times)

RG.integrate()

anim = Animator(RG)
anim.start_animation()
