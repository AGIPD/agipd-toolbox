import gc
import time
from multiprocessing import Pool
import sys
import os
import h5py

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print("BASE_PATH {}".format(BASE_PATH))

if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)

from agipdCalibration.algorithms.rangeScansFitting import *

def compute_drscs_per_mc(analog, digital, linearIndex, per_mill_interval):
    fit_slopes_result = fit3DynamicScanSlopes(analog, digital)

    # if np.mod(linearIndex, per_mill_interval) == 0:
    #     print(0.1 * linearIndex / per_mill_interval, '%')

    return fit_slopes_result


class ProcessDrscs():
    def __init__(self, source_fname, analog_gains_fname, digital_means_fname):
        self.source_fname = source_fname
        self.a_gains_fname = analog_gains_fname
        self.d_means_fname = digital_means_fname

        print("\n\n\nstart processing drscs")
        print("source_fname = {}".format(self.source_fname))
        print("a_gains_fname = {}".format(self.a_gains_fname))
        print("d_means_fname = {}".format(self.d_means_fname))
        print()

        self.n_mem_cells = 352
        self.module_h = 128  # in pixels
        self.module_l = 512  # in pixels
        self.asic_size = 64  # in pixels

        self.workerCount = 3  # python parallelizes the rest! One worker makes 30 threads!
        self.shape = (3, 352, 128, 512)
        self.shape_means = (352, 3, 128, 512)
        self.shape_thresholds = (2, 352, 128, 512)
        self.shape_safty_factor = (352, 2, 128, 512),

        start_total = time.time()
        try:
            self.run()
        except KeyboardInterrupt:
            pass


        print("processing drscs took time: {}\n\n".format(time.time() - start_total))

    def run(self):

        self.a_file = h5py.File(self.a_gains_fname, "w", libver='latest')
        dset_gains = self.a_file.create_dataset("analogGains",
                                                shape=(3, 352, 128, 512),
                                                #shape=self.shape,
                                                dtype="float32")
        dset_line_offsets = self.a_file.create_dataset("anlogLineOffsets",
                                                       shape=(3, 352, 128, 512),
                                                       #shape=self.shape,
                                                       dtype="float32")
        dset_fit_std_devs = self.a_file.create_dataset("analogFitStdDevs",
                                                       shape=(3, 352, 128, 512),
                                                       #shape=self.shape,
                                                       dtype="float32")

        self.d_file = h5py.File(self.d_means_fname, "w", libver="latest")
        dset_means = self.d_file.create_dataset("digitalMeans",
                                                shape=(352, 3, 128, 512),
                                                #shape=self.shape_means,
                                                dtype="uint16")
        dset_thresholds = self.d_file.create_dataset("digitalThresholds",
                                                     shape=(2, 352, 128, 512),
                                                     #shape=self.shape_thresholds,
                                                     dtype="uint16")
        dset_std_devs = self.d_file.create_dataset("digitalStdDeviations",
                                                   shape=(352, 3, 128, 512),
                                                   #shape=self.shape_means,
                                                   dtype="float32")
        dset_safety_factors = self.d_file.create_dataset("digitalSpacingsSafetyFactors",
                                                         shape=(352, 2, 128, 512),
                                                         #shape=self.shape_safty_factor,
                                                         dtype="float32")

        self.pool = Pool(self.workerCount)
        source_file = h5py.File(self.source_fname, "r", libver="latest")
        columnsToLoadPerIteration = 64
        rowsToLoadPerIteration = 64

        for column in np.arange(512 / columnsToLoadPerIteration):
            considered_pixels_x = (int(column * columnsToLoadPerIteration),
                                   int((column + 1) * columnsToLoadPerIteration))
            for row in np.arange(128 / rowsToLoadPerIteration):
                considered_pixels_y = (int(row * rowsToLoadPerIteration),
                                       int((row + 1) * rowsToLoadPerIteration))

                print("loading data, rows {} - {} columns {} - {} from {}"
                      .format(considered_pixels_y[0], considered_pixels_y[1],
                              considered_pixels_x[0], considered_pixels_x[1],
                              source_fname))

                t = time.time()
                analog = source_file["/analog"][:, :,
                                                considered_pixels_y[0]:considered_pixels_y[1],
                                                considered_pixels_x[0]:considered_pixels_x[1]]
                digital = source_file["/digital"][:, :,
                                                  considered_pixels_y[0]:considered_pixels_y[1],
                                                  considered_pixels_x[0]:considered_pixels_x[1]]
                print("took time: {}".format(time.time() - t))

                print("creating linear index, rows {} - {} columns {} - {}"
                      .format(considered_pixels_y[0], considered_pixels_y[1],
                              considered_pixels_x[0], considered_pixels_x[1]))
                linear_indices = np.arange(352 * columnsToLoadPerIteration * rowsToLoadPerIteration)
                matrix_indices = np.unravel_index(linear_indices,
                                                 (352, columnsToLoadPerIteration, rowsToLoadPerIteration))

                cell_list = []
                per_mill_interval = int(np.round(linear_indices.size / 1000))
                for i in np.arange(linear_indices.size):
                    idx = (slice(None),
                           matrix_indices[0][i],
                           matrix_indices[1][i],
                           matrix_indices[2][i])
                    cell_list.append((analog[idx],
                                      digital[idx],
                                      i,
                                      per_mill_interval))

                # for i in np.arange(linear_indices.size):
                #     print(i, (slice(None),
                #               matrix_indices[0][i],
                #               matrix_indices[1][i],
                #               matrix_indices[2][i]))
                #     compute_drscs_per_mc(*(cell_list[i]))

                print("start manual garbage collection")
                t = time.time()
                gc.collect()
                print("took time: {}".format(time.time() - t))

                print("start parallel computations, rows {} - {} columns {} - {}"
                      .format(considered_pixels_y[0], considered_pixels_y[1],
                              considered_pixels_x[0], considered_pixels_x[1]))
                t = time.time()
                parallel_result = self.pool.starmap(compute_drscs_per_mc,
                                                    cell_list,
                                                    chunksize=352 * 4)
                print("took time: {}".format(time.time() - t))

                print("all calculation done, rows {} - {} columns {} - {}"
                      .format(considered_pixels_y[0], considered_pixels_y[1],
                              considered_pixels_x[0], considered_pixels_x[1]))

                result_size = (352,
                               (considered_pixels_y[1] - considered_pixels_y[0]),
                               (considered_pixels_x[1] - considered_pixels_x[0]))

                # (high_gain, medium_gain, low_gain)
                gain = (np.empty(result_size, dtype='float32'),
                        np.empty(result_size, dtype='float32'),
                        np.empty(result_size, dtype='float32'))

                # (high_gain, medium_gain, low_gain)
                offset = (np.empty(result_size, dtype='float32'),
                          np.empty(result_size, dtype='float32'),
                          np.empty(result_size, dtype='float32'))

                # (high_gain, medium_gain, low_gain)
                fit_error = (np.empty(result_size, dtype='float32'),
                             np.empty(result_size, dtype='float32'),
                             np.empty(result_size, dtype='float32'))

                # (high_gain, medium_gain, low_gain)
                std_dev = (np.empty(result_size, dtype='float32'),
                           np.empty(result_size, dtype='float32'),
                           np.empty(result_size, dtype='float32'))

                # (high_gain, medium_gain, low_gain)
                means = (np.empty(result_size, dtype='uint16'),
                         np.empty(result_size, dtype='uint16'),
                         np.empty(result_size, dtype='uint16'))

                for i in np.arange(linear_indices.size):
                    idx = (matrix_indices[0][i], matrix_indices[1][i], matrix_indices[2][i])

                    (((gain[0][idx], offset[0][idx]),
                      (gain[1][idx], offset[1][idx]),
                      (gain[2][idx], offset[2][idx])),

                     (means[0][idx],
                      means[1][idx],
                      means[2][idx]),

                     (fit_error[0][idx],
                      fit_error[1][idx],
                      fit_error[2][idx]),

                     (std_dev[0][idx],
                      std_dev[1][idx],
                      std_dev[2][idx])
                     ) = parallel_result[i]

                print("start saving, rows {} - {} columns {} - {} in {} and {}"
                      .format(considered_pixels_y[0], considered_pixels_y[1],
                              considered_pixels_x[0], considered_pixels_x[1],
                              self.a_gains_fname, self.d_means_fname))

                t = time.time()
                idx = (slice(0, 3), slice(None),
                       slice(considered_pixels_y[0], considered_pixels_y[1]),
                       slice(considered_pixels_x[0], considered_pixels_x[1]))
                dset_gains[idx] = np.stack((high_gain, medium_gain, low_gain), axis=0)
                dset_line_offsets[idx] = np.stack(offset, axis=0)

                print("gains and offsets saved")

                dset_fit_std_devs[idx] = np.stack(fit_error, axis=0)

                print("analog fit errors saved")

                idx = (slice(None),
                       slice(0, 3),
                       slice(considered_pixels_y[0], considered_pixels_y[1]),
                       slice(considered_pixels_x[0], considered_pixels_x[1]))
                dset_means[idx] = np.stack(means, axis=1)
                dset_std_devs[idx] = np.stack(std_dev, axis=1)

                print("means and standard deviations saved")

                print("flushing")

                self.a_file.flush()
                self.d_file.flush()

                print("took time: {}".format(time.time() - t))

                print("saved, rows {} - {} columns {} - {}"
                      .format(considered_pixels_y[0], considered_pixels_y[1],
                              considered_pixels_x[0], considered_pixels_x[1]))
                print()

        means = dset_means[...]
        dset_thresholds[0, ...] = ((means[:, 0, ...].astype('float32')
                                      + means[:, 1, ...].astype('float32'))
                                     / 2).astype('uint16')
        dset_thresholds[1, ...] = ((means[:, 1, ...].astype('float32')
                                      + means[:, 2, ...].astype('float32'))
                                     / 2).astype('uint16')

        print("digital threshold computed and saved")

        std_devs = dset_std_devs[...]

        dset_safety_factors[:, 0, ...] = (
            (means[:, 1, ...] - means[:, 0, ...])
            / (std_devs[:, 1, ...] + std_devs[:, 0, ...]))

        dset_safety_factors[:, 1, ...] = (
            (means[:, 2, ...] - means[:, 1, ...])
            / (std_devs[:, 2, ...] + std_devs[:, 1, ...]))

        print("digital spacings safety factors computed and saved")

        source_file.close()
        self.pool.close()
        a_file.close()
        d_file.close()

        print('\ndone with processing ', self.source_fname)
        print('generated output: ', self.a_gains_fname)
        print('generated output: ', self.d_means_fname)

if __name__ == '__main__':
#    workspacePath = '/gpfs/cfel/fsds/labs/processed/kuhnm/'
    workspacePath = "/gpfs/cfel/fsds/labs/processed/calibration/processed/M314/temperature_40C/drscs/itestc150/"
    source_fname = workspacePath + 'M314_drscs_itestc150_chunked.h5'
    target_fname_a = workspacePath + 'analogGains_currentSource.h5'
    target_fname_d = workspacePath + 'digitalMeans_currentSource.h5'

    ProcessDrscs(source_fname, target_fname_a, target_fname_d)
