import numpy as np

# following code mimics "analogCorrected = (analog - darkOffset) * analogGains_keV[gainStage]"
def equalizeRawData_oneBurst(analog, digital, analogGains_keV, digitalThresholds, darkOffsets):
    gainStage = computeGainStage(digital, digitalThresholds)

    analogCorrected = (analog - np.choose(gainStage, (darkOffsets[0, ...], darkOffsets[1, ...], darkOffsets[2, ...]))) \
                      * np.choose(gainStage, (analogGains_keV[0, ...], analogGains_keV[1, ...], analogGains_keV[2, ...]))

    return analogCorrected, gainStage


def computeGainStage(digital, digitalThresholds):
    gainStage = np.zeros((352, 128, 512), dtype='uint8')
    gainStage[digital > digitalThresholds[0, ...]] = 1
    gainStage[digital > digitalThresholds[1, ...]] = 2

    return gainStage
