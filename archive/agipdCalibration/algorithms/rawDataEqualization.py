import numpy as np


# following code mimics "analogCorrected = (analog - darkOffset) * analogGains_keV[gainStage]"
def equalizeRawData_oneBurst(analog, digital, analogGains_keV, digitalThresholds, darkOffsets):
    gainStage = computeGainStage_oneBurst(digital, digitalThresholds)

    analogCorrected = (analog - np.choose(gainStage, (darkOffsets[0, ...], darkOffsets[1, ...], darkOffsets[2, ...]))) \
                      * np.choose(gainStage, (analogGains_keV[0, ...], analogGains_keV[1, ...], analogGains_keV[2, ...]))

    return analogCorrected, gainStage


def computeGainStage_oneBurst(digital, digitalThresholds):
    gainStage = np.zeros((352, 128, 512), dtype='uint8')
    gainStage[digital > digitalThresholds[0, ...]] = 1
    gainStage[digital > digitalThresholds[1, ...]] = 2

    return gainStage


# following code mimics "analogCorrected = (analog - darkOffset) * analogGains_keV[gainStage]"
def equalizeRawData_oneCell(analog, digital, analogGains_keV, digitalThresholds, darkOffsets, cellNumber):
    gainStage = computeGainStage_oneCell(digital, digitalThresholds, cellNumber)

    analogCorrected = (analog - np.choose(gainStage, (darkOffsets[0, cellNumber, ...], darkOffsets[1, cellNumber, ...], darkOffsets[2, cellNumber, ...]))) \
                      * np.choose(gainStage, (analogGains_keV[0, cellNumber, ...], analogGains_keV[1, cellNumber, ...], analogGains_keV[2, cellNumber, ...]))

    return analogCorrected, gainStage


def computeGainStage_oneCell(digital, digitalThresholds, cellNumber):
    gainStage = np.zeros((128, 512), dtype='uint8')
    gainStage[digital > digitalThresholds[0, cellNumber, ...]] = 1
    gainStage[digital > digitalThresholds[1, cellNumber, ...]] = 2

    return gainStage
