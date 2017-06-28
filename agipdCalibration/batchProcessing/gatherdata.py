from __future__ import print_function

import h5py
import time
import sys
import numpy as np
import glob
import os


class GatherData():
    def __init__(self, rel_file_path, file_base_name, output_file_name, col_spec, max_part):
        base_path = "/gpfs/cfel/fsds/labs/calibration/current/"
        self.data_path_in_file = "/entry/instrument/detector/data"
        file_path = os.path.join(base_path, rel_file_path, file_base_name)

        self.saveFileName = output_file_name
        # [[<column>, <file index>],...]
        self.column_specs = col_spec

        self.file_name_prefix = []

        for c, i in self.column_specs:
            # The method zfill() pads string on the left with zeros to fill width.
            self.file_name_prefix.append("{}_col{}_{}".format(file_path,
                                                              c,
                                                              str(i).zfill(5)))

        print("\nUsed prefixes")
        self.files = []
        for p in self.file_name_prefix:
            self.files.append(glob.glob("{}*".format(p)))
            print(p)

        for file_list in self.files:
            file_list.sort()

        if max_part:
            for i in range(len(self.files)):
                self.files[i] = self.files[i][:max_part]

        self.check_file_number()

        print('\n\nstart gatherCurrentSourceScanData')
        print('saveFileName =', self.saveFileName)
        print('')

        self.run()

    def check_file_number(self):
        len_col15 = len(self.files[0])
        len_col26 = len(self.files[1])
        len_col37 = len(self.files[2])
        len_col48 = len(self.files[3])

        if any(len(file_list) != len_col15 for file_list in self.files):
            print("Number of files does not match")
            print("file for col15", len_col15)
            print("file for col26", len_col26)
            print("file for col37", len_col37)
            print("file for col48", len_col48)
        else:
            self.number_of_files = len_col15
            print("{} file parts found".format(self.number_of_files))

    def run(self):

        # shape of input data: <total image number (analog and digital)>,
        #                      <module height>,
        #                      <module length>
        #
        #                  ____________
        #                 |___________ |
        #                |___________ ||
        #               |            |||    analog and digital images
        # module height |            ||
        #               |____________|
        #                module_length

        # shape of reshaped data: <number of bursts>,
        #                         <images_per_burst>,
        #                         <distinction between analog and digital>,
        #                         <module_height>,
        #                         <module_length>
        # for each burst:  ____________                          ____________
        #                 |___________ |                        |___________ |
        #                |___________ ||                       |___________ ||
        #               |            |||   images_per_burst   |            |||
        # module height |   analog   ||                       |  digital   ||
        #               |____________|                        |____________|
        #                module_length

        # e.g.totalimages, 128, 512 -> charges_per_file, 352, 2, 128, 512

        self.memory_cells = 352
        self.module_height = 128  # in pixels
        self.module_length = 512  # in pixels
        self.asic_size = 64  # in pixels

        f = h5py.File(self.files[0][0], 'r', libver='latest')

        charges_per_file = int(f[self.data_path_in_file].shape[0] / 2 / 352)
        charges = charges_per_file * self.number_of_files

        self.shape = (charges, 352, 128, 512)
        self.chunksize = (charges, 352, 64, 64)

        f.close()

        saveFile = h5py.File(self.saveFileName, "w", libver='latest')
        print("Create analog data set")
        dset_analog = saveFile.create_dataset("analog",
                                              shape=self.shape,
                                              chunks=self.chunksize,
                                              compression=None, dtype='int16')
        print("Create digital data set")
        dset_digital = saveFile.create_dataset("digital",
                                               shape=self.shape,
                                               chunks=self.chunksize,
                                               compression=None, dtype='int16')

        print("Initiate analog data")
        t = time.time()
        analog = np.zeros(self.shape, dtype='int16')
        print('took time:  ' + str(time.time() - t))

        print("Initiate digital data")
        t = time.time()
        # Creating the array with np.zero is faster than copying the array from analog
        digital = np.zeros(self.shape, dtype='int16')
        print('took time:  ' + str(time.time() - t))

        try:
            for i in np.arange(len(self.files)):
                index_upper_row = np.arange(3 - i, 512, 4)
                # the asics of the lower row are upside down
                index_lower_row = np.arange(i, 512, 4)

                for j in np.arange(self.number_of_files):
                    t = time.time()
                    fileName = self.files[i][j]

                    print('start loading', fileName)
                    f = h5py.File(fileName, 'r', libver='latest')
                    rawData = f[self.data_path_in_file][..., 0:128, 0:512]
                    print('took time:  ' + str(time.time() - t))

                    t = time.time()
                    print('start reshaping')
                    rawData.shape = (charges_per_file, 352, 2, 128, 512)

                    dc_start = j * charges_per_file
                    dc_stop = (j + 1) * charges_per_file

                    print('Reshaping analog data')
                    tmp = rawData[:, :, 0, :, :]
                    analog[dc_start:dc_stop, :, 0:64, index_upper_row] = tmp[..., 0:64, index_upper_row]
                    analog[dc_start:dc_stop, :, 64:, index_lower_row] = tmp[..., 64:, index_lower_row]

                    print('Reshaping digital data')
                    tmp = rawData[:, :, 1, :, :]
                    digital[dc_start:dc_stop, :, 0:64, index_upper_row] = tmp[..., 0:64, index_upper_row]
                    digital[dc_start:dc_stop, :, 64:, index_lower_row] = tmp[..., 64:, index_lower_row]
                    print('took time:  ' + str(time.time() - t))

                    f.close()
                    f = None

            t = time.time()
            print('')
            print('start saving')
            dset_analog[...] = analog
            dset_digital[...] = digital
            saveFile.flush()
            print('took time:  ' + str(time.time() - t))
        finally:
            if f is not None:
                f.close()
            saveFile.close()
