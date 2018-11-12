import numpy as np
from numba import jit
from qutip import Odeoptions
from scipy.integrate import ode
from tqdm import trange

@jit(nopython=True, cache=True)
def cyc_grad(vec, eps):
    N = len(vec)
    new_vec = np.zeros(N, dtype=np.complex128)

    new_vec[0] = (vec[1] - vec[N-1]) / (2*eps)
    new_vec[N-1] = (vec[0] - vec[N-2]) / (2 * eps)

    for i in range(1, N-1):
        new_vec[i] = (vec[i+1] - vec[i-1]) / (2 * eps)

    return new_vec

class Linear():
    def __init__(self):
        self.A0 = 1# np.exp(1j * 0.54)
        self.Omega = 0
        self.kappa = 1
        self.dt = .02
        self.omega = 1.41

        # time of one period
        self.T = 2 * np.pi / self.omega
        periods = 3
        self.periodic_time = periods * self.T

        self.times = np.arange(-self.periodic_time, self.periodic_time, self.dt)[:-1]
        # self.times = np.arange(-50, 50, self.dt)

        self.dz = 0.005
        self.z = np.arange(0, 5, self.dz)

        self.Y = None

        self.T0 = 2.
        # self.a0 = np.exp(-self.times**2 / self.T0**2)
        W = 1
        # self.a0 = np.cos(self.omega * self.times)
        self.a0 = np.exp(- 1j * self.omega * self.times)

    def __func_LS(self, z, y):
        return self.Omega * np.gradient(y, self.dt) + \
               1j/2 * np.gradient(np.gradient(y, self.dt), self.dt) + \
               1j * self.kappa * abs(self.A0)**2 * y + \
               1j * self.kappa * self.A0**2 * np.conj(y)

    def __func_LS_cyc(self, z, y):
        return self.Omega * cyc_grad(y, self.dt) + \
               1j/2 * cyc_grad(cyc_grad(y, self.dt), self.dt) + \
               1j * self.kappa * abs(self.A0)**2 * y + \
               1j * self.kappa * self.A0**2 * np.conj(y)

    def integrate(self):
        r = ode(self.__func_LS_cyc)
        opt = Odeoptions()
        opt.atol = 1e-10
        opt.rtol = 1e-8
        opt.max_step = .005

        r.set_integrator('zvode', method=opt.method, order=opt.order,
                         atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                         first_step=opt.first_step, min_step=opt.min_step,
                         max_step=opt.max_step)

        r.set_initial_value(self.a0, 0)

        self.Y = [r.y]

        for i in trange(1, len(self.z)):
            self.Y.append(r.integrate(self.z[i]))

        self.Y = np.array(self.Y)
