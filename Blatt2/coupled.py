import numpy as np
from qutip import Odeoptions
from scipy.integrate import ode
from tqdm import trange


class Coupled():
    def __init__(self):
        self.A0 = 1
        self.Omega = 1
        self.kappa = 1
        self.dt = 0.1
        self.times = np.arange(-500, 500, self.dt)
        # self.times = np.arange(-25, 25, self.dt)

        self.dz = 0.005
        self.z = np.arange(0, 10, self.dz)

        self.Y = None

        self.a0 = np.zeros((2, len(self.times)))

        self.a0

    def __func_b(self, z, y):
        return 1j * self.c * np.gradient(np.gradient(y, self.dt), self.dt)

    def __func_c(self, z, y):
        return 1j * self.c * abs(y)**2 * y

    def __func_d(self, z, y):
        return self.c * y**2 * np.gradient(y, self.dt)

    def __func_bc(self, z, y):
        return 1j * self.c * abs(y)**2 * y + \
               - 0.2 * 1j * self.c * np.gradient(np.gradient(y, self.dt), self.dt)

    def __func_bcd(self, z, y):
        return 1j * self.c * abs(y)**2 * y + \
               - 0.2 * 1j * self.c * np.gradient(np.gradient(y, self.dt), self.dt) + \
               0.2 * self.c * y**2 * np.gradient(y, self.dt)

    def __func_NLS(self, z, y):
        return 1j / 2 * np.gradient(np.gradient(y, self.dt), self.dt) \
               + 1j * abs(y)**2 * y

    def integrate(self):
        r = ode(self.__func_NLS)
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
