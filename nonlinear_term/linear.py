import numpy as np
from qutip import Odeoptions
from scipy.integrate import ode
from tqdm import trange


class Linear():
    def __init__(self):
        self.A0 = 1# np.exp(1j * 0.54)
        self.Omega = 1
        self.kappa = 1
        self.dt = .05
        self.times = np.arange(-50, 50, self.dt)
        # self.times = np.arange(-25, 25, self.dt)

        self.dz = 0.005
        self.z = np.arange(0, 10, self.dz)

        self.Y = None

        self.T0 = 2.
        # self.a0 = np.exp(-self.times**2 / self.T0**2)
        W = 1
        self.a0 = np.cos(2 * self.times)

    def __func_LS(self, z, y):
        return self.Omega * np.gradient(y, self.dt) + \
               1j/2 * np.gradient(np.gradient(y, self.dt), self.dt) + \
               1j * self.kappa * abs(self.A0)**2 * y + \
               1j * self.kappa * self.A0**2 * np.conj(y)

    def integrate(self):
        r = ode(self.__func_LS)
        opt = Odeoptions()
        opt.atol = 1e-12
        opt.rtol = 1e-10
        opt.max_step = .1

        r.set_integrator('zvode', method=opt.method, order=opt.order,
                         atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                         first_step=opt.first_step, min_step=opt.min_step,
                         max_step=opt.max_step)

        r.set_initial_value(self.a0, 0)

        self.Y = [r.y]

        for i in trange(1, len(self.z)):
            self.Y.append(r.integrate(self.z[i]))

        self.Y = np.array(self.Y)
