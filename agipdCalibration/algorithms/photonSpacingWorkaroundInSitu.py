from agipdCalibration.algorithms.xRaxTubeDataFitting import *

def getOnePhotonAdcCountsInSitu(analog):
    upperCutoffPart = 0.01
    upperCutoffRank = int(np.round(analog.size*upperCutoffPart))
    analog_cleaned=np.sort(analog)[:-upperCutoffRank]

    return getOnePhotonAdcCountsXRayTubeData(analog, applyLowpass=False)