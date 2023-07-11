# %%
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
import numba as nb
randomSeed = 100

%matplotlib inline
%config InlineBackend.figure_format = "svg"

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.jet(np.linspace(0, 1, 256)) * 0.85, N=256
)

import seaborn as sns

sns.set(font_scale=1.1, rc={
    'figure.figsize': (10, 6),
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'grid.color': '#dddddd',
    'grid.linewidth': 0.5,
    "lines.linewidth": 1.5,
    'text.color': '#000000',
    'figure.titleweight': "bold",
    'xtick.color': '#000000',
    'ytick.color': '#000000'
})

# %%
class Swarmalators2D():
    def __init__(self, agentsNum: int, dt: float, K: float) -> None:
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * 2 - 1
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi
        self.agentsNum = agentsNum
        self.dt = dt
        self.K = K

    @staticmethod
    @nb.njit
    def _delta_theta(phaseTheta):
        dim = phaseTheta.shape[0]
        subTheta = phaseTheta - np.repeat(phaseTheta, dim).reshape(dim, dim)

        deltaTheta = np.zeros((dim, dim - 1))
        for i in np.arange(dim):
            deltaTheta[i, :i], deltaTheta[i, i:] = subTheta[i, :i], subTheta[i, i + 1 :]
        return deltaTheta

    @staticmethod
    @nb.njit
    def _delta_x(positionX):
        dim = positionX.shape[0]
        subX = positionX - np.repeat(positionX, dim).reshape(dim, 2, dim).transpose(0, 2, 1)
        deltaX = np.zeros((dim, dim - 1, 2))
        for i in np.arange(dim):
            deltaX[i, :i], deltaX[i, i:] = subX[i, :i], subX[i, i + 1 :]
        return deltaX

    @staticmethod
    @nb.njit
    def distance_x_2(deltaX):
        return np.sqrt(deltaX[:, :, 0] ** 2 + deltaX[:, :, 1] ** 2).reshape(deltaX.shape[0], deltaX.shape[1], 1)

    @staticmethod
    @nb.njit
    def distance_x(deltaX):
        return np.sqrt(deltaX[:, :, 0] ** 2 + deltaX[:, :, 1] ** 2)

    @property
    def deltaTheta(self) -> np.ndarray:
        """Phase difference between agents"""
        return self._delta_theta(self.phaseTheta)

    @property
    def deltaX(self) -> np.ndarray:
        """Spatial difference between agents"""
        return self._delta_x(self.positionX)

    @property
    def Fatt(self) -> np.ndarray:
        """Effect of phase similarity on spatial attraction"""
        pass

    @property
    def Frep(self) -> np.ndarray:
        """Effect of phase similarity on spatial repulsion"""
        pass

    @property
    def Iatt(self) -> np.ndarray:
        """Spatial attraction"""
        pass

    @property
    def Irep(self) -> np.ndarray:
        """Spatial repulsion"""
        pass

    @property
    def H(self) -> np.ndarray:
        """Phase interaction"""
        pass

    @property
    def G(self) -> np.ndarray:
        """Effect of spatial similarity on phase couplings"""
        pass

    @property
    def velocity(self) -> np.ndarray:
        """Self propulsion velocity"""
        pass

    @property
    def omega(self) -> np.ndarray:
        """Natural intrinsic frequency"""
        pass

    @staticmethod
    @nb.njit
    def _update(
        positionX: np.ndarray, phaseTheta: np.ndarray,
        velocity: np.ndarray, omega: np.ndarray,
        Iatt: np.ndarray, Irep: np.ndarray,
        Fatt: np.ndarray, Frep: np.ndarray,
        H: np.ndarray, G: np.ndarray,
        K: float, dt: float
    ):
        dim = positionX.shape[0]
        pointX = velocity + np.sum(
            Iatt * Fatt.reshape((dim, dim - 1, 1)) - Irep * Frep.reshape((dim, dim - 1, 1)),
            axis=1
        ) / (dim - 1)
        positionX += pointX * dt
        pointTheta = omega + K * np.sum(H * G, axis=1) / (dim - 1)
        phaseTheta = np.mod(phaseTheta + pointTheta * dt, 2 * np.pi)
        return positionX, phaseTheta
    

    def update(self) -> None:
        self.positionX, self.phaseTheta = self._update(
            self.positionX, self.phaseTheta,
            self.velocity, self.omega,
            self.Iatt, self.Irep,
            self.Fatt, self.Frep,
            self.H, self.G,
            self.K, self.dt
        )

    def plot(self) -> None:
        plt.figure(figsize=(6, 5))

        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        plt.scatter(self.positionX[:, 0], self.positionX[:, 1],
                    c=self.phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)

        cbar = plt.colorbar(ticks=[0, np.pi, 2*np.pi])
        cbar.ax.set_ylim(0, 2*np.pi)
        cbar.ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])
        
        # plt.show()


class ChiralSwarmalators2D(Swarmalators2D):
    def __init__(self, agentsNum: int, dt: float,
                 J: float, K: float,
                 A: float, B: float) -> None:
        super().__init__(agentsNum, dt, K)
        self.J = J
        self.K = K
        self.A = A
        self.B = B * np.ones((agentsNum, agentsNum - 1))
        self.omegaValue = np.ones(agentsNum)
        randomHalfIdx = np.random.choice(agentsNum, agentsNum // 2, replace=False)
        self.omegaValue[randomHalfIdx] = -1

    @property
    def omega(self) -> np.ndarray:
        return self.omegaValue

    @property
    def Fatt(self) -> np.ndarray:
        """
        Effect of phase similarity on spatial attraction
        A + J * cos(theta_j - theta_i)
        """
        return self.A + self.J * np.cos(self.deltaTheta)
    
    @property
    def Frep(self) -> np.ndarray:
        """Effect of phase similarity on spatial repulsion: B"""
        return self.B

    @property
    def Iatt(self) -> np.ndarray:
        """Spatial attraction: (x_j - x_i) / |x_j - x_i|"""

        return self.deltaX / self.distance_x_2(self.deltaX)
    # self.deltaX / np.linalg.norm(self.deltaX, axis=-1, keepdims=True)
    
    @property
    def Irep(self) -> np.ndarray:
        """Spatial repulsion: (x_j - x_i) / |x_j - x_i| ^ 2"""
        return self.deltaX / self.distance_x_2(self.deltaX) ** 2

    @property
    def H(self) -> np.ndarray:
        """Phase interaction: sin(theta_j - theta_i)"""
        return np.sin(self.deltaTheta)
    
    @property
    def G(self) -> np.ndarray:
        """Effect of spatial similarity on phase couplings: 1 / |x_i - x_j|"""
        return 1 / self.distance_x(self.deltaX)
    
    @property
    def velocity(self) -> np.ndarray:
        """Self propulsion velocity: 0"""
        return 0
    

def plot_model(positionX, phaseTheta, xlim, ylim, title):
    plt.figure(figsize=(6, 5))

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    plt.scatter(positionX[:, 0], positionX[:, 1],
                c=phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)

    cbar = plt.colorbar(ticks=[0, np.pi, 2 * np.pi])
    cbar.ax.set_ylim(0, 2* np.pi)
    cbar.ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)

# %%
J = 1
K = -0.5
dt = 0.1

model = ChiralSwarmalators2D(agentsNum=500, dt=dt, J=J, K=K, A=1, B=1)
# 迭代1000次，每隔5次用记录模型的positionX，phaseTheta，最后用plot_model画出每次记录的图像，xlim，ylim分别是x轴和y轴的范围，title是次数序号
import imageio
import os
import shutil

if os.path.exists("images"):
    shutil.rmtree("images")
os.mkdir("images")

positions = []
phases = []

for i in tqdm(range(1000)):
    model.update()
    if i % 5 == 0:
        positions.append(model.positionX.copy())
        phases.append(model.phaseTheta.copy())

xOfPositions = np.array(positions)[:, :, 0]
yOfPositions = np.array(positions)[:, :, 1]
xlim = (np.min(xOfPositions), np.max(xOfPositions))
ylim = (np.min(yOfPositions), np.max(yOfPositions))
images = []

for i, (position, phase) in tqdm(enumerate(zip(positions, phases)), total=len(positions)):
    plot_model(position, phase, xlim=xlim, ylim=ylim, title=f"{i}. J={J}, K={K}, dt={dt * 5 * i}")
    plt.savefig(f"images/{i}.png")
    plt.close()
    images.append(imageio.imread(f"images/{i}.png"))
    
imageio.mimsave('chiral_swarmalators.gif', images, fps=10)

# %%
!ffmpeg -i chiral_swarmalators.gif chiral_swarmalators.mp4


