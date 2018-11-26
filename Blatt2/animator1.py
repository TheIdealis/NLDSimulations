import pylab as plt
from matplotlib import animation
from matplotlib import rcParams
from scipy.optimize import minimize
from scipy.signal import get_window
import scipy.fftpack as ft
import numpy as np

plt.style.use('default')

rcParams['font.size'] = 15
rcParams['axes.labelsize'] = 15
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = [r'\usepackage{garamondx}',
                                   r'\usepackage{amsmath}',
                                   r'\usepackage{nicefrac}',
                                   r'\usepackage{siunitx}']


class Animator:
    """base class for animating the GPE"""

    def __init__(self, RG, steps=1, frames=300, skip=1, filename=None, spec_cut=2):
        """Constructor for Animator"""

        self.filename = filename

        self.steps = steps
        self.skip = skip
        self.Y = RG.Y
        self.frames = self.Y.shape[0]
        self.times = RG.times
        self.grid_points = len(self.Y[0])
        self.window = get_window('blackman', self.grid_points)
        self.dw = RG.dt / (2 * np.pi)
        self.w = ft.fftfreq(self.grid_points, d=self.dw)
        self.w = ft.fftshift(self.w)
        self.tslice = slice(int(self.grid_points*0.4), int(self.grid_points*0.6))
        self.test_old = 1
        self._create_figure()

    def _create_figure(self):
        self.fig, self.axes = plt.subplots(
            figsize=(10., 5), nrows=1, ncols=2,
        )

        plt.subplots_adjust(
            wspace=0.25
        )

        self._a_line, = self.axes[0].plot(self.times[self.tslice], self.Y[0][self.tslice])
        self._a_line2, = self.axes[0].plot(self.times[self.tslice], abs(self.Y[0][self.tslice]),
                                           'k--')
        self._a_line3, = self.axes[0].plot(self.times[self.tslice], abs(self.Y[0][self.tslice]),
                                           'g')

        self._spec_line, = self.axes[1].plot(self.w, abs(ft.fft(self.Y[0])))

        # Axis text
        self.axes[0].set_ylabel('$a(\\tau, z)$')
        self.axes[0].set_xlabel('$\\tau$')

        self.axes[1].set_ylabel('$I(\omega, z)$')
        self.axes[1].set_xlabel('$\omega$')

        self.axes[0].set_yticks([])
        # self.axes[1].set_yticks([])
        # self.axes[1].set_yticklabels([])

    def __animate(self, step):

        def cost_func(lam):
            test_func = lam * 1/(np.cosh(self.times * lam))
            return np.sum((test_func - abs(self.Y[step]))**2)

        res = minimize(cost_func, self.test_old, method='Nelder-Mead', tol=1e-6,
                       options={'disp': False})
        # print(res.)
        fit_func = res.x[0] * 1/(np.cosh(self.times[self.tslice] * res.x[0]))
        self.test_old = res.x[0]

        self._a_line.set_data(self.times[self.tslice], self.Y[step][self.tslice])
        self._a_line2.set_data(self.times[self.tslice], abs(self.Y[step][self.tslice]))
        self._a_line3.set_data(self.times[self.tslice], abs(fit_func))
        fft_func = ft.fftshift(abs(ft.fft(self.window * self.Y[step])))
        self._spec_line.set_data(self.w, fft_func)
        self.axes[0].relim()

        lim = abs(self.Y[step]).max()
        self.axes[0].set_ylim([-0.4*lim, 1.5*lim])
        # self.fig.canvas.draw()

        return (self._a_line, self._a_line2, self._a_line3, self._spec_line)

    def start_animation(self, offset=0):

        anim = animation.FuncAnimation(
            self.fig, self.__animate, frames=self.frames, interval=1, blit=True)

        plt.show()
