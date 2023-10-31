import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import imageio
import sys
import os
import shutil

randomSeed = 100

# %matplotlib inline
# %config InlineBackend.figure_format = "svg"

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

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

sys.path.append("..")
from swarmalatorlib.template import Swarmalators2D


import numpy as np


class Strogatz2017(Swarmalators2D):
    def __init__(self, agentsNum: int, dt: float, 
                 K: float, J: float, fixSpiralPhaseCoupling: bool = False,
                 randomSeed: int = 100, tqdm: bool = False, savePath: str = None, shotsnaps: int = 5) -> None:
        super().__init__(agentsNum, dt, K, randomSeed, tqdm, savePath, shotsnaps)
        self.J = J
        self.fixSpiralPhaseCoupling = fixSpiralPhaseCoupling
        self.one = np.ones((agentsNum, agentsNum))
        self.phaseTheta = np.zeros(agentsNum)

    @property
    def Fatt(self) -> np.ndarray:
        """
        Effect of phase similarity on spatial attraction

        if fixSpiralPhaseCoupling is True, then Fatt = 1 + J

        else Fatt = 1 + J * cos(theta_j - theta_i)
        """
        if self.fixSpiralPhaseCoupling:
            return 1 + self.J * self.one
        else:
            return 1 + self.J * np.cos(self.deltaTheta)
    
    @property
    def Frep(self) -> np.ndarray:
        """Effect of phase similarity on spatial repulsion: 1"""
        return self.one
    
    @property
    def Iatt(self) -> np.ndarray:
        """Spatial attraction: (x_j - x_i) / |x_j - x_i|"""
        return self.div_distance_power(numerator=self.temp["deltaX"], power=1)
    
    @property
    def Irep(self) -> np.ndarray:
        """Spatial repulsion: (x_j - x_i) / |x_j - x_i| ^ 2"""
        return self.div_distance_power(numerator=self.temp["deltaX"], power=2)
    
    @property
    def velocity(self) -> np.ndarray:
        """Self propulsion velocity: 0"""
        return 0

    def update_temp(self):
        self.temp["deltaX"] = self.deltaX
        self.temp["distanceX2"] = self.distance_x_2(self.temp["deltaX"])


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
            Iatt * Fatt.reshape((dim, dim, 1)) - Irep * Frep.reshape((dim, dim, 1)),
            axis=1
        ) / (dim - 1)
        positionX += pointX * dt
        return positionX, phaseTheta
    
    def __str__(self) -> str:
        if self.fixSpiralPhaseCoupling:
            return f"Strogatz2017_a{self.agentsNum}_K{self.K}_J{self.J:.2f}_fix"
        else:
            return f"Strogatz2017_a{self.agentsNum}_K{self.K}_J{self.J:.2f}"


from itertools import product

Jrange = np.concatenate([
    np.arange(0.01, 0.1, 0.01),
    np.arange(0.1, 1.1, 0.1),
    np.arange(1, 11, 1)
])
agentsNumRange = np.arange(1000, 5001, 1000)


models = [
    Strogatz2017(agentsNum=agentsNum, dt=0.1, K=1, J=J, 
                 fixSpiralPhaseCoupling=True, randomSeed=randomSeed, 
                 tqdm=False, savePath="data", shotsnaps=1)
    for J, agentsNum in tqdm(product(Jrange, agentsNumRange), total=len(Jrange) * len(agentsNumRange))
]

def run_model(model):
    model.run(2000)

with Pool(40) as p:
    p.map(run_model, models)