import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from itertools import product
import numpy as np
import numba as nb
import imageio
import os
import shutil

randomSeed = 100

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.jet(np.linspace(0, 1, 256)) * 0.85, N=256
)

@nb.njit
def colors_idx(phaseTheta):
    return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

import seaborn as sns

sns.set(font_scale=1.1, rc={
    'figure.figsize': (6, 5),
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

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['animation.ffmpeg_path'] = "D:/Programs/ffmpeg/bin/ffmpeg.exe"


def runge_kutta_4(func, y, h):
    k1 = func(y)
    k2 = func(y + h / 2 * k1)
    k3 = func(y + h / 2 * k2)
    k4 = func(y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

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

        plt.scatter(self.positionX[:, 0], self.positionX[:, 1],
                    c=self.phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)

        cbar = plt.colorbar(ticks=[0, np.pi, 2*np.pi])
        cbar.ax.set_ylim(0, 2*np.pi)
        cbar.ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])


class SpatialGroups(Swarmalators2D):
    def __init__(self, strengthLambda: float, distanceD0: float, 
                 agentsNum: int=1000, dt: float=0.01, tqdm: bool = False) -> None:
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * 10
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi - np.pi
        self.agentsNum = agentsNum
        self.dt = dt
        self.speedV = 0.03
        self.distanceD0 = distanceD0
        self.omegaTheta = np.concatenate([
            np.random.normal(loc=3, scale=0.5, size=agentsNum // 2),
            np.random.normal(loc=-3, scale=0.5, size=agentsNum // 2)
        ])
        self.strengthLambda = strengthLambda
        self.tqdm = tqdm

    @property
    def K(self):
        return self._K(self.distance_x(self.deltaX), self.distanceD0).astype(float)
    # (self.distance_x(self.deltaX) <= self.distanceD0).astype(float)

    @staticmethod
    @nb.njit
    def _K(distanceX, distanceD0):
        return (distanceX <= distanceD0)

    @staticmethod
    @nb.njit
    def _delta_x(positionX):
        dim = positionX.shape[0]
        subX = positionX - np.repeat(positionX, dim).reshape(dim, 2, dim).transpose(0, 2, 1)
        return subX

    @property
    def pointTheta(self):
        return self._pointTheta(self.phaseTheta, self.omegaTheta, self.strengthLambda, self.dt, self.K)

    @staticmethod
    @nb.njit
    def _pointTheta(phaseTheta: np.ndarray, omegaTheta: np.ndarray, strengthLambda: float, 
                    h: float, K: np.ndarray):
        adjMatrixTheta = np.repeat(phaseTheta, phaseTheta.shape[0]).reshape(phaseTheta.shape[0], phaseTheta.shape[0])
        k1 = omegaTheta + strengthLambda * np.sum(K * np.sin(
            adjMatrixTheta - phaseTheta
        ), axis=1)
        k2 = omegaTheta + strengthLambda * np.sum(K * np.sin(
            adjMatrixTheta - (phaseTheta + h / 2 * k1)
        ), axis=1)
        k3 = omegaTheta + strengthLambda * np.sum(K * np.sin(
            adjMatrixTheta - (phaseTheta + h / 2 * k2)
        ), axis=1)
        k4 = omegaTheta + strengthLambda * np.sum(K * np.sin(
            adjMatrixTheta - (phaseTheta + h * k3)
        ), axis=1)
        return (k1 + 2 * k2 + 2 * k3 + k4) * h / 6

    def update(self):
        self.positionX[:, 0] += self.speedV * np.cos(self.phaseTheta)
        self.positionX[:, 1] += self.speedV * np.sin(self.phaseTheta)
        self.positionX = np.mod(self.positionX, 10)
        self.phaseTheta += self.pointTheta
        self.phaseTheta = np.mod(self.phaseTheta + np.pi, 2 * np.pi) - np.pi

    def plot(self) -> None:
        plt.figure(figsize=(6, 5))

        plt.scatter(self.positionX[:, 0], self.positionX[:, 1],
                    c=self.phaseTheta, cmap=new_cmap, alpha=0.8, vmin=-np.pi, vmax=np.pi)

        cbar = plt.colorbar(ticks=[-np.pi, 0, np.pi])
        cbar.ax.set_ylim(-np.pi, np.pi)
        cbar.ax.set_yticklabels(['$\pi$', '$0$', '$\pi$'])

    def plot_update(self, _):
        if self.tqdm:
            pbar.update(1)
        pointTheta = self.pointTheta
        self.update()
        clockWise, antiClockWise = np.where(pointTheta > 0), np.where(pointTheta < 0)
        plt.cla()
        line = plt.quiver(
            self.positionX[clockWise, 0], self.positionX[clockWise, 1],
            np.cos(self.phaseTheta[clockWise]), np.sin(self.phaseTheta[clockWise]), color='tomato'
        )
        line = plt.quiver(
            self.positionX[antiClockWise, 0], self.positionX[antiClockWise, 1],
            np.cos(self.phaseTheta[antiClockWise]), np.sin(self.phaseTheta[antiClockWise]), color='dodgerblue'
        )
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        return line

    def run_mp4(self, TNum: int):
        file_name = f"spatial_groups_{self.strengthLambda:2f}_{self.distanceD0:2f}.mp4"

        if self.tqdm:
            global pbar
            pbar = tqdm(total=TNum)

        fig, ax = plt.subplots()
        ani = ma.FuncAnimation(fig, self.plot_update, frames=np.arange(0, TNum, 1), interval=10, repeat=False)
        ani.save(file_name, dpi=200)
