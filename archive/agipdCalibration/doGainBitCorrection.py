# -*- coding: utf-8 -*-
"""
Does gain bit correction on data

Created on Wed May 10 09:53:12 2017

@author: Jennifer Pöhlsen (jennifer.poehlsen@desy.de)
"""

import h5py
import numpy as np
import time
import matplotlib.pyplot as plt


dataDir = '/home/jsibille/AGIPD/Calibration/GainBitCorrection/H5Files/'


fileName = dataDir + 'linear3_352images.hdf5'
dataPathInFile = '/entry/instrument/detector/data'
gainFileName = dataDir + 'combinedCalibrationConstants.h5'
gainPath = '/analogGains_keV'
darkPath = '/darkOffsets'
saveFileName = dataDir + 'linear3_decoded.h5'
calibFileName = gainFileName
calibDataPathInFile = '/GainBitCorrection'

memCells = 352
x_pix = 128
y_pix = 512

doPlot = True
plotCell = 175


print('\n\n\nstart gainBitCorrection')
print('fileName = ', fileName)
print('saveFileName = ', saveFileName)
print('')

totalTime = time.time()


# Load raw data
print('start loading ', dataPathInFile, ' from ', fileName)
f = h5py.File(fileName, 'r')
rawData = f[dataPathInFile][()]
f.close()
print('loading done')


# Load calibration
print('start loading ', calibDataPathInFile, ' from ', calibFileName)
f2 = h5py.File(calibFileName, 'r')
calibTable = f2[calibDataPathInFile][()]
# In future these should all be in the same calibration file!
f3 = h5py.File(gainFileName, 'r')
gain = f3[gainPath][()]
dark = f3[darkPath][()]
f2.close()
f3.close()
print('loading done')


# decode gain information
print('Decoding digital and analog data, correcting wrong gain stages')
digital = np.empty([memCells, x_pix, y_pix], dtype='int64')
analog = np.empty([memCells, x_pix, y_pix])

badpix = 0
wrongpix = 0

# Loop over pixels
for i in range(memCells):
    for j in range(x_pix):
        for k in range(y_pix):

            # Separate digital and analog information
            encoded = format(rawData[i, j, k], '016b')
            digital[i, j, k] = int(encoded[0:2], 2)
            analog[i, j, k] = int(encoded[2:], 2)

            # Check if gain stage is correct
            if calibTable[i, j, k] == 1:
            #if calibTable[j, k] == 1:
                digital[i, j, k] = digital[i, j, k] - 1
                wrongpix = wrongpix + 1
            elif calibTable[i, j, k] > 1:
            #elif calibTable[j, k] > 1:
                badpix = badpix + 1
                analog[i, j, k] = 0


print('Decoding and correcting gain stages done')
print('Wrong pixels: ', wrongpix)
print('Bad pixels: ', badpix)



# Apply dark correction and gain calibration
image = np.ones([memCells, x_pix, y_pix])
image = (analog - np.choose(digital, (dark[0, ...], dark[1, ...], dark[2, ...]))) * np.choose(digital, (gain[0, ...], gain[1, ...], gain[2, ...]))/100


# Plots
if doPlot:
    # Corrected image
    fig1 = plt.figure(1)
    sp1 = fig1.add_subplot(111)
    pt1 = sp1.matshow(image[plotCell,...], vmax=8000)
    fig1.colorbar(pt1, orientation='horizontal')
    sp1.set_title('Dark and gain corrected image [# photons]')
    sp1.xaxis.tick_bottom()
    plt.show()
    # Analog values
    fig2 = plt.figure(2)
    sp2 = fig2.add_subplot(111)
    pt2 = sp2.matshow(analog[plotCell,...])
    fig2.colorbar(pt2, orientation='horizontal')
    sp2.set_title('Analog values [ADU]')
    sp2.xaxis.tick_bottom()
    plt.show()
    # Digital values
    fig3 = plt.figure(3)
    sp3 = fig3.add_subplot(111)
    pt3 = sp3.matshow(digital[plotCell,...])
    fig3.colorbar(pt3, orientation='horizontal')
    sp3.set_title('Gain Stage: 0 = high, 1 = med, 2 = low, 3 = 4th region')
    sp3.xaxis.tick_bottom()
    plt.show()

# Save corrected data in output file
saveFile = h5py.File(saveFileName, 'w', libver='latest')
dset_dig = saveFile.create_dataset('digital', shape=digital.shape, dtype='uint8')
dset_ana = saveFile.create_dataset('analog', shape=analog.shape)
dset_image = saveFile.create_dataset('corrected_image', shape=image.shape)

print('start saving to ', saveFileName)
dset_dig[...] = digital
dset_ana[...] = analog
dset_image[...] = image
saveFile.close()
print('saving done')


print('gainBitCorrection took time: ', time.time() - totalTime, '\n\n')
