from agipdCalibration.algorithms.moKAlphaFitting import *

def getOnePhotonAdcCountsInSitu(analog):
    upperCutoffPart = 0.01
    upperCutoffRank = int(np.round(analog.size*upperCutoffPart))
    analog_cleaned=np.sort(analog)[:-upperCutoffRank]

    return getOnePhotonAdcCountsMoKAlpha(analog, applyLowpass=False)