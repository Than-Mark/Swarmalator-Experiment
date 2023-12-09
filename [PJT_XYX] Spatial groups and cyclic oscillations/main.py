import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from itertools import product
from typing import List
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
        self.positionX = np.random.random((agentsNum, 2)) * self.boundaryLength
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
        self.randomSeed = randomSeed
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
        self.positionX = np.mod(self.positionX, self.boundaryLength)
        self.temp = self.pointTheta
        self.phaseTheta += self.temp
        self.phaseTheta = np.mod(self.phaseTheta + np.pi, 2 * np.pi) - np.pi

    def __str__(self) -> str:
        
        if self.uniform:
            name =  f"CorrectCoupling_uniform_{self.strengthLambda:.3f}_{self.distanceD0:.2f}_{self.randomSeed}"
        else:
            name =  f"CorrectCoupling_normal_{self.strengthLambda:.3f}_{self.distanceD0:.2f}_{self.randomSeed}"

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
            name =  f"CorrectCoupling_uniform_{self.strengthLambda:.3f}_{self.distanceD0:.2f}_{self.randomSeed}"
        else:
            name =  f"CorrectCoupling_normal_{self.strengthLambda:.3f}_{self.distanceD0:.2f}_{self.randomSeed}"

        if self.omegaTheta2Shift != 0:
            name += f"_shift_{self.omegaTheta2Shift:.2f}"

        return name

    def __str__(self) -> str:
            
            if self.uniform:
                name =  f"CorrectCouplingAfter_uniform_{self.strengthLambda:.3f}_{self.distanceD0:.2f}_{self.randomSeed}"
            else:
                name =  f"CorrectCouplingAfter_normal_{self.strengthLambda:.3f}_{self.distanceD0:.2f}_{self.randomSeed}"
    
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


class SingleDistribution(SpatialGroups):
    def __init__(self, strengthLambda: float, distanceD0: float, boundaryLength: float = 5, 
                 agentsNum: int=500, dt: float=0.01, 
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 5, 
                 distributType: str = "const", randomSeed: int = 10, overWrite: bool = False) -> None:
        assert distributType in ["const", "normal", "uniform"], "distributType must be const, normal or uniform"
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * boundaryLength
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi - np.pi
        self.agentsNum = agentsNum
        self.dt = dt
        self.speedV = 0.03
        self.distanceD0 = distanceD0
        if distributType == "uniform":
            self.omegaTheta = np.random.uniform(1, 3, size=agentsNum)
        elif distributType == "normal":
            self.omegaTheta = np.random.normal(loc=3, scale=0.5, size=agentsNum)
        elif distributType == "const":
            self.omegaTheta = np.ones(agentsNum) * 3
        self.distributType = distributType
        self.strengthLambda = strengthLambda
        self.tqdm = tqdm
        self.savePath = savePath
        self.temp = np.zeros(agentsNum)
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.boundaryLength = boundaryLength
        self.halfBoundaryLength = boundaryLength / 2
        self.randomSeed = randomSeed
        self.overWrite = overWrite

    def __str__(self) -> str:
        
        
        name =  f"SingleDistribution_{self.distributType}_{self.strengthLambda:.3f}_{self.distanceD0:.2f}_{self.randomSeed}"
        
        return name
        

class StateAnalysis:
    def __init__(self, model: SpatialGroups, classDistance: float = 2, lookIndex: int = -1, tqdm: bool = True):
        self.model = model
        self.classDistance = classDistance
        self.lookIndex = lookIndex
        self.tqdm = tqdm
        
        targetPath = f"{self.model.savePath}/{self.model}.h5"
        totalPositionX = pd.read_hdf(targetPath, key="positionX")
        totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
        totalPointTheta = pd.read_hdf(targetPath, key="pointTheta")
        
        TNum = totalPositionX.shape[0] // self.model.agentsNum

        self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum, 2)
        self.totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, self.model.agentsNum)
        self.totalPointTheta = totalPointTheta.values.reshape(TNum, self.model.agentsNum)

        self.centersValue = None
        self.classesValue = None

        if self.tqdm:
            self.iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
        else:
            self.iterObject = range(1, self.totalPhaseTheta.shape[0])

    def get_state(self, index: int = -1):
        positionX = self.totalPositionX[index]
        phaseTheta = self.totalPhaseTheta[index]
        pointTheta = self.totalPointTheta[index]

        return positionX, phaseTheta, pointTheta
    
    # @staticmethod
    # @nb.njit
    # def _calc_centers(positionX, phaseTheta, pointTheta, speedV):
    #     centers = np.zeros((positionX.shape[0], 2))

    #     for i in range(positionX.shape[0]):
    #         counts = max(200, abs(2 * np.pi // pointTheta[i]))
    #         phaseChanges = np.arange(0, counts * pointTheta[i], pointTheta[i])

    #         trackX = positionX[i, 0] + np.cumsum(speedV * np.cos(phaseTheta[i] + phaseChanges))
    #         trackY = positionX[i, 1] + np.cumsum(speedV * np.sin(phaseTheta[i] + phaseChanges))

    #         centers[i, 0] = np.mean(trackX)  # np.sum(np.array([trackX, trackY]), axis=1) / counts
    #         centers[i, 1] = np.mean(trackY)  

    #     return centers

    @staticmethod
    @nb.njit
    def _calc_centers(positionX, phaseTheta, pointTheta, speedV):
        centers = np.zeros((positionX.shape[0], 2))

        for i in range(positionX.shape[0]):
            position, phase, point = positionX[i], phaseTheta[i], pointTheta[i]
            point1 = position
            velocity1 = speedV * np.array([np.cos(phase), np.sin(phase)])
            point2 = position + velocity1
            velocity2 = speedV * np.array([np.cos(phase + point), np.sin(phase + point)])

            unit_velocity1 = velocity1 / np.linalg.norm(velocity1)
            unit_velocity2 = velocity2 / np.linalg.norm(velocity2)

            coefficients = np.array([
                [unit_velocity1[0], unit_velocity1[1]],
                [unit_velocity2[0], unit_velocity2[1]]
            ])

            constants = np.array([
                np.sum(unit_velocity1 * point1),
                np.sum(unit_velocity2 * point2)
            ])

            center = np.linalg.solve(coefficients, constants)

            centers[i] = center

        return centers

    @property
    def centers(self):
        
        lastPositionX, lastPhaseTheta, lastPointTheta = self.get_state(self.lookIndex)
        
        centers = self._calc_centers(
            lastPositionX, lastPhaseTheta, lastPointTheta, self.model.speedV
        )

        return np.mod(centers, self.model.boundaryLength)

    @property
    def centersNoMod(self):
            
        lastPositionX, lastPhaseTheta, lastPointTheta = self.get_state(self.lookIndex)
        
        centers = self._calc_centers(
            lastPositionX, lastPhaseTheta, lastPointTheta, self.model.speedV
        )

        return centers
         
    
    def adj_distance(self, positionX, others):
        return self._adj_distance(
            positionX, others, self.model.boundaryLength, self.model.halfBoundaryLength
        )

    @staticmethod
    @nb.njit
    def _adj_distance(positionX, others, boundaryLength, halfLength):
        subX = positionX - others
        adjustOthers = (
            others * (-halfLength <= subX) * (subX <= halfLength) + 
            (others - boundaryLength) * (subX < -halfLength) + 
            (others + boundaryLength) * (subX > halfLength)
        )
        adjustSubX = positionX - adjustOthers
        return np.sqrt(np.sum(adjustSubX ** 2, axis=-1))
    
    @staticmethod
    @nb.njit
    def _calc_classes(centers, classDistance, totalDistances):
        classes = [[0]]
        classNum = 0
        nonClassifiedOsci = np.arange(1, centers.shape[0])

        for i in nonClassifiedOsci:
            newClass = True

            for classI in range(len(classes)):
                distance = classDistance
                for j in classes[classI]:
                    if totalDistances[i, j] < distance:
                        distance = totalDistances[i, j]
                if distance < classDistance:
                    classes[classI].append(i)
                    newClass = False
                    break

            if newClass:
                classNum += 1
                classes.append([i])

        return classes

    def get_classes_centers(self):
        centers = self.centers
        classes = self._calc_classes(
            centers, self.classDistance, self.adj_distance(centers, centers[:, np.newaxis])
        )
        return {i + 1: classes[i] for i in range(len(classes))}, centers

    def plot_spatial(self, ax: plt.Axes = None, oscis: np.ndarray = None, index: int = -1, **kwargs):
        positionX, phaseTheta, pointTheta = self.get_state(index)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        if oscis is None:
            oscis = np.arange(self.model.agentsNum)

        ax.quiver(
            positionX[oscis, 0], positionX[oscis, 1],
            np.cos(phaseTheta[oscis]), np.sin(phaseTheta[oscis]), **kwargs
        )
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)    

    def plot_centers(self, ax: plt.Axes = None, index: int = -1):
        positionX, phaseTheta, pointTheta = self.get_state(index)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        quiverColors = ["tomato"] * 500 + ["dodgerblue"] * 500
        ax.quiver(
            positionX[:, 0], positionX[:, 1],
            np.cos(phaseTheta[:]), np.sin(phaseTheta[:]), color=quiverColors, alpha=0.5
        )
        centerColors = ["orange"] * 500 + ["blue"] * 500
        centers = self.centers
        ax.scatter(centers[:, 0], centers[:, 1], color=centerColors, s=5)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)    
    
    def center_radius_op(self, classOsci: np.ndarray):
        centers = self.centers
        return self.adj_distance(centers[classOsci], self.totalPositionX[:, classOsci]).mean(axis=1)
    
    def tv_center_radius_op(self, step: int = 10):
        centerRadios = []

        if self.tqdm:
            iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
        else:
            iterObject = range(1, self.totalPhaseTheta.shape[0])

        for i in iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i
            centers = self.centersNoMod
            modCenters = np.mod(centers, self.model.boundaryLength)
            distance = self.adj_distance(modCenters, self.totalPositionX[i])
            d = np.mean(distance)
            centerRadios = centerRadios + [d] * step

        return np.array(centerRadios)
    
    def tv_center_position(self, step: int = 30):
        color = ["tomato"] * 500 + ["dodgerblue"] * 500

        t = []
        positionX = []
        positionY = []
        colors = []

        if self.tqdm:
            iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
        else:
            iterObject = range(1, self.totalPhaseTheta.shape[0])

        for i in iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i

            centers = self.centers
            t.append(np.ones(self.model.agentsNum) * i)
            positionX.append(centers[:, 0])
            positionY.append(centers[:, 1])
            colors.append(color)

        t = np.concatenate(t, axis=0)
        positionX = np.concatenate(positionX, axis=0)
        positionY = np.concatenate(positionY, axis=0)
        colors = np.concatenate(colors, axis=0)

        return np.array([t, positionX, positionY]).T, colors

    def tv_center_radius(self, step: int = 30):
        
        color = ["tomato"] * 500 + ["dodgerblue"] * 500

        t = []
        centerRadios = []
        colors = []

        if self.tqdm:
            iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
        else:
            iterObject = range(1, self.totalPhaseTheta.shape[0])

        for i in iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i

            centers = self.centers
            t.append(np.ones(self.model.agentsNum) * i)
            centerRadios.append(self.adj_distance(self.totalPositionX[i], centers))
            colors.append(color)

        t = np.concatenate(t, axis=0)
        centerRadios = np.concatenate(centerRadios, axis=0)
        colors = np.concatenate(colors, axis=0)

        return np.array([t, centerRadios]).T, colors
    
    def phase_agg_op(self):
        theta = self.totalPhaseTheta[self.lookIndex]
        return self._clac_phase_sync_op(self, theta)
    
    @staticmethod
    @nb.njit
    def _clac_phase_sync_op(theta):
        N = theta.shape[0]
        return (
            (np.sum(np.sin(theta)) / N) ** 2 + 
            (np.sum(np.cos(theta)) / N) ** 2
        )**0.5

    @staticmethod
    @nb.njit
    def _delta_x(positionX, others, boundaryLength, halfLength):
        subX = positionX - others
        adjustOthers = (
            others * (-halfLength <= subX) * (subX <= halfLength) + 
            (others - boundaryLength) * (subX < -halfLength) + 
            (others + boundaryLength) * (subX > halfLength)
        )
        adjustSubX = positionX - adjustOthers
        return adjustSubX

    def delta_x(self, positionX, others):
        return self._delta_x(
            positionX, others, self.model.boundaryLength, self.model.halfBoundaryLength
        )

    @property
    def center_agg1(self):

        centers = self.centers
        deltaX = self.delta_x(centers, centers[:, np.newaxis])

        return np.sqrt(np.sum(deltaX ** 2, axis=2)).mean(axis=1)

    @property
    def center_agg2(self):

        centers = self.centers
        deltaX = self.delta_x(centers, centers[:, np.newaxis])

        return np.sqrt(np.sum(deltaX.mean(axis=1) ** 2, axis=-1))
    
    @property
    def dis_reciwgt_phase_agg(self):
        centers = self.centers
        deltaX = self.delta_x(centers, centers[:, np.newaxis])
        distance = np.sqrt(np.sum(deltaX ** 2, axis=-1))
        centerDisMat = distance.copy()
        centerDisMat[centerDisMat != 0] = 1 / centerDisMat[centerDisMat != 0]
        N = centers.shape[0]
        theta = self.totalPhaseTheta[self.lookIndex]
        return [
            (
                (np.sum(np.sin(theta) * centerDisMat[rowIdx]) / centerDisMat[rowIdx].sum()) ** 2 + 
                (np.sum(np.cos(theta) * centerDisMat[rowIdx]) / centerDisMat[rowIdx].sum()) ** 2
            )**0.5
            for rowIdx in range(N)
        ]

    def tv_dis_reciwgt_phase_agg(self, step: int = 30):
        color = ["tomato"] * 500 + ["dodgerblue"] * 500

        t = []
        opValues = []
        colors = []

        for i in self.iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i

            opValue = self.dis_reciwgt_phase_agg

            t.append(np.ones(self.model.agentsNum) * i)
            opValues.append(opValue)
            colors.append(color)

        t = np.concatenate(t, axis=0)
        opValues = np.concatenate(opValues, axis=0)
        colors = np.concatenate(colors, axis=0)

        return np.array([t, opValues]).T, colors

    def tv_dis_reciwgt_phase_agg_op(self, step: int = 10):
        t = []
        r = []

        for i in self.iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i

            opValue = self.dis_reciwgt_phase_agg

            r.append(np.mean(opValue))
            t.append(i)

        return np.array([t, r]).T

    def tv_center_agg(self, opType: int, step: int = 30):
        color = ["tomato"] * 500 + ["dodgerblue"] * 500

        t = []
        centerAggs = []
        colors = []

        if self.tqdm:
            iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
        else:
            iterObject = range(1, self.totalPhaseTheta.shape[0])

        for i in iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i

            if opType == 1:
                caValue = self.center_agg1
            elif opType == 2:
                caValue = self.center_agg2

            t.append(np.ones(self.model.agentsNum) * i)
            centerAggs.append(caValue)
            colors.append(color)

        t = np.concatenate(t, axis=0)
        centerAggs = np.concatenate(centerAggs, axis=0)
        colors = np.concatenate(colors, axis=0)

        return np.array([t, centerAggs]).T, colors
    
    def tv_center_agg_op(self, opType: int, step: int = 10):
        assert opType in [1, 2], "opType must be 1 or 2"

        t = []
        r = []

        if self.tqdm:
            iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
        else:
            iterObject = range(1, self.totalPhaseTheta.shape[0])

        for i in iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i

            if opType == 1:
                caValue = self.center_agg1
            elif opType == 2:
                caValue = self.center_agg2

            r.append(np.mean(caValue))
            t.append(i)

        return np.array([t, r]).T

def plot_drpaop(models: List[SpatialGroups], savePath: str = None):
    _ = plt.figure(figsize=(3 * 5, len(models) * 5))

    idx = 0

    for model in tqdm(models):
        ax1 = plt.subplot2grid((len(models), 3), (idx, 0), colspan=2)
        ax2 = plt.subplot2grid((len(models), 3), (idx, 2))

        sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
        drpa = sa.tv_dis_reciwgt_phase_agg_op(20)
        ax1.plot(drpa[:, 0], drpa[:, 1])
        ax1.set_ylim(0, 1)
        ax1.set_title(f"{model},     t: 0-12000, Dis ReciWgt Phase Agg")

        sa.plot_centers(ax=ax2, index=-1)
        ax2.set_title(f"snapshot at 12000")

        idx += 1

    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=100, bbox_inches="tight")
    plt.close()

def plot_drpa(models: List[SpatialGroups], savePath: str = None):
    _ = plt.figure(figsize=(3 * 5, len(models) * 5))

    idx = 0

    for model in tqdm(models):
        ax1 = plt.subplot2grid((len(models), 3), (idx, 0), colspan=2)
        ax2 = plt.subplot2grid((len(models), 3), (idx, 2))

        sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
        drpa, colors = sa.tv_dis_reciwgt_phase_agg(30)
        ax1.scatter(drpa[:, 0], drpa[:, 1], s=0.5, alpha=0.01, c=colors)
        ax1.set_ylim(0, 1)
        ax1.set_title(f"{model},     t: 0-12000, Dis ReciWgt Phase Agg")

        sa.plot_centers(ax=ax2, index=-1)
        ax2.set_title(f"snapshot at 12000")

        idx += 1

    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=100, bbox_inches="tight")
    plt.close()

def plot_tvcp(models: List[SpatialGroups], savePath: str = None):
    _ = plt.figure(figsize=(3 * 5, len(models) * 5))

    idx = 0

    for model in tqdm(models):
        ax1Row1 = plt.subplot2grid((len(models) * 2, 3), (idx, 0), colspan=2)
        ax1Row2 = plt.subplot2grid((len(models) * 2, 3), (idx + 1, 0), colspan=2)
        ax2 = plt.subplot2grid((len(models) * 2, 3), (idx, 2), rowspan=2)

        sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
        cp1, colors = sa.tv_center_position(step=30)
        sa.plot_centers(ax=ax2, index=-1)
        ax2.set_title(f"snapshot at 12000")

        ax1Row1.scatter(cp1[:, 0], cp1[:, 1], s=0.5, alpha=0.01, c=colors)
        ax1Row1.set_ylim(0, 10)
        ax1Row1.set_title(f"{model}\t, t: 0-12000, PositionX")
        ax1Row2.scatter(cp1[:, 0], cp1[:, 2], s=0.5, alpha=0.01, c=colors)
        ax1Row2.set_ylim(0, 10)
        ax1Row2.set_title(f"PositionY")

        idx += 2

    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=100, bbox_inches="tight")
    plt.close()

def plot_tvca(models: List[SpatialGroups], opType: int, savePath: str = None):
    assert opType in [1, 2], "opType must be 1 or 2"
    _ = plt.figure(figsize=(3 * 5, len(models) * 5))

    idx = 0

    for model in tqdm(models):
        ax1 = plt.subplot2grid((len(models), 3), (idx, 0), colspan=2)
        ax2 = plt.subplot2grid((len(models), 3), (idx, 2))

        sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
        ca, colors = sa.tv_center_agg(opType, 30)
        ax1.scatter(ca[:, 0], ca[:, 1], s=0.5, alpha=0.01, c=colors)
        ax1.set_title(f"{model},     t: 0-12000, Center Agg{opType}")

        sa.plot_centers(ax=ax2, index=-1)
        ax2.set_title(f"snapshot at 12000")

        idx += 1

    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=100, bbox_inches="tight")
    plt.close()


def plot_tvcaop(models: List[SpatialGroups], opType: int, savePath: str = None):
    assert opType in [1, 2], "opType must be 1 or 2"
    _ = plt.figure(figsize=(3 * 5, len(models) * 5))

    idx = 0

    for model in tqdm(models):
        ax1 = plt.subplot2grid((len(models), 3), (idx, 0), colspan=2)
        ax2 = plt.subplot2grid((len(models), 3), (idx, 2))

        sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
        caop = sa.tv_center_agg_op(opType, 20)
        ax1.plot(caop[:, 0], caop[:, 1])
        ax1.set_title(f"{model},     t: 0-12000, Center Agg Order Parameter{opType}")

        sa.plot_centers(ax=ax2, index=-1)
        ax2.set_title(f"snapshot at 12000")

        idx += 1

    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=100, bbox_inches="tight")
    plt.close()

# def plot_tvcr(models: List[SpatialGroups], savePath: str = None):
#     _ = plt.figure(figsize=(3 * 5, len(models) * 5))

#     idx = 0

#     for model in tqdm(models):
#         ax1 = plt.subplot2grid((len(models), 3), (idx, 0), colspan=2)
#         ax2 = plt.subplot2grid((len(models), 3), (idx, 2))

#         sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
#         cr, colors = sa.tv_center_radius(30)
#         ax1.scatter(cr[:, 0], cr[:, 1], s=0.5, alpha=0.01, c=colors)
#         ax1.set_title(f"{model},     t: 0-12000, Global center distance")

#         sa.plot_centers(ax=ax2, index=-1)
#         ax2.set_title(f"snapshot at 12000")

#         idx += 1

#     plt.tight_layout()
#     if savePath is not None:
#         plt.savefig(savePath, dpi=100, bbox_inches="tight")
#     plt.close()


# def plot_tvccr(models: List[SpatialGroups], savePath: str = None):
#     _ = plt.figure(figsize=(3 * 5, len(models) * 5))

#     idx = 0

#     for model in tqdm(models):
#         ax1 = plt.subplot2grid((len(models), 3), (idx, 0), colspan=2)
#         ax2 = plt.subplot2grid((len(models), 3), (idx, 2))

#         sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
#         tvccr = sa.tv_class_center_radius(35)
#         ax1.plot(tvccr[:, 0], tvccr[:, 1])
#         ax1.set_title(f"{model}, t: 0-12000, Class mean center distance")

#         sa.plot_centers(ax=ax2, index=-1)
#         ax2.set_title(f"snapshot at 12000")

#         idx += 1

#     plt.tight_layout()
#     if savePath is not None:
#         plt.savefig(savePath, dpi=100, bbox_inches="tight")
#     plt.close()

# def plot_tvccp(models: List[SpatialGroups], savePath: str = None):
#     _ = plt.figure(figsize=(3 * 5, len(models) * 5))

#     idx = 0

#     for model in tqdm(models):
#         ax1Row1 = plt.subplot2grid((len(models) * 2, 3), (idx, 0), colspan=2)
#         ax1Row2 = plt.subplot2grid((len(models) * 2, 3), (idx + 1, 0), colspan=2)
#         ax2 = plt.subplot2grid((len(models) * 2, 3), (idx, 2), rowspan=2)

#         sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
#         cp1 = sa.tv_class_center_position(step=30)
#         sa.plot_centers(ax=ax2, index=-1)

#         ax1Row1.scatter(cp1[:, 0], cp1[:, 1], s=0.5, alpha=0.01)
#         ax1Row1.set_ylim(0, 10)
#         ax1Row1.set_title(f"{model}, t: 0-12000, Class center PositionX")
#         ax1Row2.scatter(cp1[:, 0], cp1[:, 2], s=0.5, alpha=0.01)
#         ax1Row2.set_ylim(0, 10)
#         ax1Row2.set_title(f"PositionY")

#         idx += 2

#     plt.tight_layout()
#     if savePath is not None:
#         plt.savefig(savePath, dpi=100, bbox_inches="tight")
#     plt.close()