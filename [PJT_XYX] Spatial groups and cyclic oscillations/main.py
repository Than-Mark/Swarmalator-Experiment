import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import imageio
import sys
import os
import shutil

randomSeed = 10

if "ipykernel_launcher.py" in sys.argv[0]:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

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
if os.path.exists("/opt/conda/bin/ffmpeg"):
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"
else:
    plt.rcParams['animation.ffmpeg_path'] = "D:/Programs/ffmpeg/bin/ffmpeg.exe"

sys.path.append("..")
from swarmalatorlib.template import Swarmalators2D


class SpatialGroups(Swarmalators2D):
    def __init__(self, strengthLambda: float, distanceD0: float, boundaryLength: float = 10, 
                 omegaTheta2Shift: float = 0, agentsNum: int=1000, dt: float=0.01, 
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 5, 
                 uniform: bool = True, randomSeed: int = 10, overWrite: bool = False) -> None:
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * 10
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi - np.pi
        self.agentsNum = agentsNum
        self.dt = dt
        self.speedV = 0.03
        self.distanceD0 = distanceD0
        if uniform:
            self.omegaTheta = np.concatenate([
                np.random.uniform(1, 3, size=agentsNum // 2),
                np.random.uniform(-3, -1, size=agentsNum // 2)
            ])
        else:
            self.omegaTheta = np.concatenate([
                np.random.normal(loc=3, scale=0.5, size=agentsNum // 2),
                np.random.normal(loc=-3, scale=0.5, size=agentsNum // 2)
            ])
        self.uniform = uniform
        self.strengthLambda = strengthLambda
        self.tqdm = tqdm
        self.savePath = savePath
        self.temp = np.zeros(agentsNum)
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.omegaTheta[:self.agentsNum // 2] += omegaTheta2Shift
        self.omegaTheta2Shift = omegaTheta2Shift
        self.boundaryLength = boundaryLength
        self.halfBoundaryLength = boundaryLength / 2
        self.overWrite = overWrite

    @property
    def K(self):
        return self.distance_x(self.deltaX) <= self.distanceD0

    @property
    def deltaX(self) -> np.ndarray:
        return self._delta_x(self.positionX, self.positionX[:, np.newaxis], 
                             self.boundaryLength, self.halfBoundaryLength)

    @staticmethod
    @nb.njit
    def _delta_x(positionX: np.ndarray, others: np.ndarray,
                 boundaryLength: float, halfBoundaryLength: float) -> np.ndarray:
        subX = positionX - others
        return positionX - (
            others * (-halfBoundaryLength <= subX) * (subX <= halfBoundaryLength) + 
            (others - boundaryLength) * (subX < -halfBoundaryLength) + 
            (others + boundaryLength) * (subX > halfBoundaryLength)
        )

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
        ), axis=0)
        return k1 * h

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))
            self.store.append(key="pointTheta", value=pd.DataFrame(self.temp))

    def update(self):
        self.positionX[:, 0] += self.speedV * np.cos(self.phaseTheta)
        self.positionX[:, 1] += self.speedV * np.sin(self.phaseTheta)
        self.positionX = np.mod(self.positionX, 10)
        self.temp = self.pointTheta
        self.phaseTheta += self.temp
        self.phaseTheta = np.mod(self.phaseTheta + np.pi, 2 * np.pi) - np.pi

    def __str__(self) -> str:
        
        if self.uniform:
            name =  f"CorrectCoupling_uniform_{self.strengthLambda:.3f}_{self.distanceD0:.2f}"
        else:
            name =  f"CorrectCoupling_normal_{self.strengthLambda:.3f}_{self.distanceD0:.2f}"

        if self.omegaTheta2Shift != 0:
            name += f"_shift_{self.omegaTheta2Shift:.2f}"

        return name

    def close(self):
        if self.store is not None:
            self.store.close()

    
class CorrectCouplingAfter(SpatialGroups):
    def __init__(self, strengthLambda: float, distanceD0: float, boundaryLength: float = 10,
                 omegaTheta2Shift: float = 0, agentsNum: int=1000, dt: float=0.01, 
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 5, 
                 uniform: bool = True, randomSeed: int = 10, overWrite: bool = False) -> None:
        super().__init__(strengthLambda, distanceD0, boundaryLength, omegaTheta2Shift, 
                         agentsNum, dt, tqdm, savePath, shotsnaps, uniform, randomSeed, overWrite)

        targetPath = f"./data/{self.oldName}.h5"
        totalPositionX = pd.read_hdf(targetPath, key="positionX")
        totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")

        TNum = totalPositionX.shape[0] // self.agentsNum
        totalPositionX = totalPositionX.values.reshape(TNum, self.agentsNum, 2)
        totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, self.agentsNum)
        
        self.positionX = totalPositionX[-1]
        self.phaseTheta = totalPhaseTheta[-1]

    @property
    def oldName(self) -> str:
        
        if self.uniform:
            name =  f"CorrectCoupling_uniform_{self.strengthLambda:.3f}_{self.distanceD0:.2f}"
        else:
            name =  f"CorrectCoupling_normal_{self.strengthLambda:.3f}_{self.distanceD0:.2f}"

        if self.omegaTheta2Shift != 0:
            name += f"_shift_{self.omegaTheta2Shift:.2f}"

        return name

    def __str__(self) -> str:
            
            if self.uniform:
                name =  f"CorrectCouplingAfter_uniform_{self.strengthLambda:.3f}_{self.distanceD0:.2f}"
            else:
                name =  f"CorrectCouplingAfter_normal_{self.strengthLambda:.3f}_{self.distanceD0:.2f}"
    
            return name
    
    def run(self, enhancedLambdas: np.ndarray):

        if not self.init_store():
            return
        
        TNum = enhancedLambdas.shape[0]
        if self.tqdm:
            iterRange = tqdm(range(TNum))
        else:
            iterRange = range(TNum)

        for idx in iterRange:
            self.strengthLambda = enhancedLambdas[idx]
            self.update()
            self.append()
            self.counts = idx

        self.close()