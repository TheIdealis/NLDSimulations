import numpy as np
from qutip import Odeoptions
from scipy.integrate import ode
from tqdm import trange


class NonLinear():
    def __init__(self):
        self.c = .25

        self.dx = None
        self.dy = None

        self.dz = 0.01
        self.z = np.arange(0, 10, self.dz)
        self.a0 = None

        self.Nx = None
        self.Ny = None

        self.kappa = 1

    def _func_NLS(self, z, y):
        y = y.reshape(self.Nx, self.Ny)
        mat =  1j / 2 * np.gradient(np.gradient(y, self.dx, axis=0), self.dx, axis=0) \
               + 1j / 2 * np.gradient(np.gradient(y, self.dy, axis=1), self.dy, axis=1) \
               + self.kappa * 1j * abs(y) ** 2 * y
        return mat.reshape(self.Nx * self.Ny)

    def integrate(self, type='NLS'):
        self.Nx, self.Ny = self.x.shape
        r = ode(getattr(self, '_func_' + type))
        opt = Odeoptions()
        opt.atol = 1e-12
        opt.rtol = 1e-10
        opt.max_step = self.dz

        r.set_integrator('zvode', method=opt.method, order=opt.order,
                         atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                         first_step=opt.first_step, min_step=opt.min_step,
                         max_step=opt.max_step)

        r.set_initial_value(self.a0.reshape(self.Nx * self.Ny), 0)

        self.Y = [self.a0]

        for i in trange(1, len(self.z)):
            self.Y.append(r.integrate(self.z[i]).reshape(self.Nx, self.Ny))

        self.Y = np.array(self.Y)
