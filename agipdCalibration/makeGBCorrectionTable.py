# -*- coding: utf-8 -*-
"""
Creates the lookup table for the gain bit correction

Created on Wed May 10 09:53:12 2017

@author: Jennifer Pöhlsen (jennifer.poehlsen@desy.de)
"""

import h5py
import time
import numpy as np
import matplotlib.pyplot as plt

inFileName = 'digitalMeans_currentSource_2.h5'
outFileName = 'GB_lookup_table_currentSource_2.h5'

memCells = 352
x_pix = 128
y_pix = 512

# Plot histogram of gain bits first?
plot = True

totalTime = time.time()


print('Start loading gain bits')
file = h5py.File(inFileName, 'r')
dset_gb = file['/digitalMeans'][()]
dset_stdev = file['/digitalStdDeviations'][()]
file.close()
print('Finished loading gain bits')


if(plot):
    print('Plotting histogram of gain bits, close histogram to continue')
    hi = dset_gb[1:351, 0, :, :].flatten()
    med = dset_gb[1:351, 1, :, :].flatten()
    low = dset_gb[1:351, 2, :, :].flatten()
    plt.hist(hi, bins=100, range=[4000, 14000], histtype='step')
    plt.hist(med, bins=100, range=[4000, 14000], histtype='step')
    plt.hist(low, bins=100, range=[4000, 14000], histtype='step')
    plt.yscale('log')
    plt.show()
    print('Finished plotting')


# thresholds hard-coded for now
threshold = ([6800, 8000, 10000])
# initialize counters
shifted = 0
toofar = 0
wrongdir = 0
tooclose = 0
failed_h = 0
failed_m = 0
failed_l = 0
close_h = close_m = close_l = 0

print('Start sorting by region')
gbtable = np.zeros([352, 128, 512])
for i in range(memCells):
    if i % 50 == 0:
        print('Memcell ', i)
    for j in range(x_pix):
        for k in range(y_pix):

            # check if low gain is shifted
            if dset_gb[i, 2, j, k] > threshold[2]:
                gbtable[i, j, k] = 1
                shifted = shifted + 1

            # check if high or low gain more than 1 region wrong
            if dset_gb[i, 1, j, k] > threshold[2] or dset_gb[i, 0, j, k] > threshold[1]:
                gbtable[i, j, k] = 2
                toofar = toofar + 1

            # check if gain wrong in wrong direction
            if dset_gb[i, 2, j, k] < threshold[1] and dset_gb[i, 2, j, k] > 0:
                gbtable[i, j, k] = 3
                wrongdir = wrongdir + 1

            # check if gain too close to threshold
            h = m = l = False
            if dset_gb[i, 0, j, k] > threshold[0] - 1*dset_stdev[i, 0, j, k] and dset_gb[i, 0, j, k] < threshold[0] + 1*dset_stdev[i, 0, j, k]:
                h = True
            if dset_gb[i, 1, j, k] > threshold[1] - 1*dset_stdev[i, 1, j, k] and dset_gb[i, 1, j, k] < threshold[1] + 1*dset_stdev[i, 1, j, k]:
                m = True
            if dset_gb[i, 2, j, k] > threshold[2] - 1*dset_stdev[i, 2, j, k] and dset_gb[i, 2, j, k] < threshold[2] + 1*dset_stdev[i, 2, j, k]:
                l = True
            if h or m or l:
                gbtable[i, j, k] = 4
                tooclose = tooclose + 1
            if h:
                close_h = close_h + 1
            if m:
                close_m = close_m + 1
            if l:
                close_l = close_l + 1

            # check if gain bit fitting failed (gain=0)
            if dset_gb[i, 0, j, k] == 0:
                failed_h = failed_h + 1
            if dset_gb[i, 1, j, k] == 0:
                failed_m = failed_m + 1
            if dset_gb[i, 2, j, k] == 0:
                failed_l = failed_l + 1


badpix = toofar + wrongdir + tooclose
totalpix = 3*memCells*x_pix*y_pix
print('Finished sorting')
print('')
print('Shifted pixels: ', shifted, ' (', (shifted/totalpix)*100, '%)')
print('Bad pixels (excluding failed fits): ', badpix, ' (', (badpix/totalpix)*100, '%)')
print('Shifted by more than 1 region: ', toofar, ' (', (toofar/totalpix)*100, '%)')
print('Shifted in the wrong direction: ', wrongdir, ' (', (wrongdir/totalpix)*100, '%)')
print('Too close to threshold (h, m, l): ', close_h, close_m, close_l)
print('Too close to threshold: ', tooclose, ' (', (tooclose/totalpix)*100, '%)')
print('Gain bit fitting failed (h, m, l): ', failed_h, failed_m, failed_l)
print('Gain bit fitting failed high gain: ', (failed_h/(totalpix/3))*100, '%')
print('')


print('Start saving to ', outFileName)
outFile = h5py.File(outFileName, 'w')
dset_gbtable = outFile.create_dataset('GainBitCorrection', shape=gbtable.shape, dtype='uint8')
dset_gbtable[...] = gbtable
outFile.close()
print('Saving done')


print('makeGBCorrectionTable took time: ', time.time() - totalTime, '\n\n')
