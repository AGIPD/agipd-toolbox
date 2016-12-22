from agipdCalibration.helperFunctions import *

from agipdCalibration.algorithms.moKAlphaFitting import *
from agipdCalibration.batchProcessing.gatherMokalphaData import *


# for interactive pycharm: %matplotlib tk


localityRadius = 800
samplePointsCount = 1000
print(getOnePhotonAdcCountsMoKAlpha(analog, localityRadius, samplePointsCount))
