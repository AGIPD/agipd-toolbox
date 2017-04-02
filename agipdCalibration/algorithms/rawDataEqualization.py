import numpy as np

#todo: currently using same dark offset for all gain stages. This is wrong, but no algoritms that provide different dark offsets exist at the time of writing this message, so the dark offset of the high gain is used for all gain stages
def equalizeRawData_oneBurst(analog, digital, analogGains_keV, digitalThresholds, darkOffset):
    gainStage = computeGainStage(digital, digitalThresholds)

    # following code mimics "analogCorrected = (analog - darkOffset) * analogGains_keV[gainStage]"
    analogCorrected = (analog - darkOffset) * np.choose(gainStage, (analogGains_keV[0, ...], analogGains_keV[1, ...], analogGains_keV[2, ...]))

    return analogCorrected, gainStage


def computeGainStage(digital, digitalThresholds):
    gainStage = np.zeros((352, 128, 512), dtype='uint8')
    gainStage[digital > digitalThresholds[0, ...]] = 1
    gainStage[digital > digitalThresholds[1, ...]] = 2

    return gainStage
