import numpy as np
from scipy.integrate import ode
from tqdm import trange


class NonLinear():
    def __init__(self):
        self.c = .25

        self.dt = 0.1
        self.times = np.arange(-15, 15, self.dt)

        self.dz = 0.005
        self.z = np.arange(0, 30, self.dz)

        self.Y = None

        self.T0 = 1
        self.a0 = np.exp(-self.times**2 / self.T0**2)

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

    def integrate(self):
        r = ode(self.__func_b).set_integrator('zvode', method='adams')

        r.set_initial_value(self.a0, 0)

        self.Y = [r.y]

        for i in trange(1, len(self.z)):
            self.Y.append(r.integrate(self.z[i]))

        self.Y = np.array(self.Y)
