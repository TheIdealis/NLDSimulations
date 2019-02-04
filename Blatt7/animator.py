import pylab as plt
from matplotlib import animation
from matplotlib import rcParams
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tqdm import tqdm

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

        self.x = RG.x
        self.y = RG.y
        self.steps = steps
        self.skip = skip
        self.Y = RG.Y
        self.frames = self.Y.shape[0]
        self.tqdm = tqdm(total=self.frames)
        self.stride = 1
        self._create_figure()

    def _create_figure(self):
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        self.plot = self.ax.plot_surface(self.x, self.y, np.abs(self.Y[0]),
                                         rstride=self.stride, cstride=self.stride,
                                         linewidth=0, antialiased=False,
                                         cmap=cm.coolwarm)

        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zlim([0, 1])

    def __animate(self, step):
        self.plot.remove()
        self.plot = self.ax.plot_surface(self.x, self.y, np.abs(self.Y[step]),
                                         rstride=self.stride, cstride=self.stride,
                                         linewidth=0, antialiased=False,
                                         cmap=cm.coolwarm)
        self.tqdm.update()
        return self.plot,

    def start_animation(self, filename=None):
        anim = animation.FuncAnimation(
            self.fig, self.__animate, frames=self.frames, interval=1, blit=False)

        if filename is None:
            plt.show()
        else:
            anim.save(filename, fps=20)
