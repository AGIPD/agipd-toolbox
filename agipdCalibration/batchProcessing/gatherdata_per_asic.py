from __future__ import print_function

import h5py
import time
import sys
import numpy as np
import glob
import os


class GatherData():
    def __init__(self, asic, rel_file_path, file_base_name, output_file_name, col_spec, max_part):

        base_path = "/gpfs/cfel/fsds/labs/calibration/current/"
        filename_template = Template("${prefix}_col${column}")
        self.data_path_in_file = "/entry/instrument/detector/data"

        file_path = os.path.join(base_path, rel_file_path, file_base_name)

        self.save_file = output_file_name
        self.check_output_file_exist()

        # [[<column>, <file index>],...]
        self.column_specs = col_spec

        self.n_mem_cells = 352
        self.module_h = 128  # in pixels
        self.module_l = 512  # in pixels
        self.asic_size = 64  # in pixels

        self.asic = asic
        #                    [rows, columns]
        self.asics_per_module = [2, 8]

        self.file_name_prefix = []
        if type(self.column_specs[0]) == list:
            for c, i in self.column_specs:
                file_prefix = filename_template.substitute(prefix=file_path, column=c)

                # The method zfill() pads string on the left with zeros to fill width.
                self.file_name_prefix.append("{}_{}".format(file_prefix,
                                                            str(i).zfill(5)))
#                self.file_name_prefix.append("{}_col{}_{}".format(file_path,
#                                                                  c,
#                                                                  str(i).zfill(5)))
        else:
            for c in self.column_specs:
                self.file_name_prefix.append(
                    filename_template.substitute(prefix=file_path, column=c))

        print("\nUsed prefixes")
        self.files = []
        for p in self.file_name_prefix:
            self.files.append(glob.glob("{}*".format(p)))
            print(p)

        # to make sure that part_00000 comes before part_00001
        for file_list in self.files:
            file_list.sort()

        self.check_for_single_index()

        if max_part:
            for i in range(len(self.files)):
                self.files[i] = self.files[i][:max_part]

        self.check_file_number()

        try:
            f = h5py.File(self.files[0][0], 'r', libver='latest')
            #TODO: check that shape for charges_per_file is the same for all files
            self.charges_per_file = int(f[self.data_path_in_file].shape[0] / 2 / self.n_mem_cells)
            self.charges = self.charges_per_file * self.number_of_files
        except:
            print("error when determining charge numbers")
            raise
        finally:
            f.close()

        self.shape = (self.charges, self.n_mem_cells, self.asic_size, self.asic_size)
        #self.shape = (self.charges, 352, 128, 512)
        self.chunksize = (self.charges, self.n_mem_cells, self.asic_size, self.asic_size)

        #       ____ ____ ____ ____ ____ ____ ____ ____
        # 0x64 |    |    |    |    |    |    |    |    |
        #      | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  |
        # 1x64 |____|____|____|____|____|____|____|____|
        #      | 9  | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
        # 2x64 |____|____|____|____|____|____|____|____|
        #      0*64 1x64 2x64 3x64 4x64 5x64 6x64 7x64 8x64

        # asic counting starts with 1 and index with 0
        self.col_progress = (self.asic - 1) / self.asics_per_module[1]
        self.row_progress = (self.asic - 1) % self.asics_per_module[1]
        print("col_progress:", self.col_progress)
        print("row_progress:", self.row_progress)

        self.a_col_start = self.col_progress * self.asic_size
        self.a_col_stop = (self.col_progress + 1) * self.asic_size
        self.a_row_start = self.row_progress * self.asic_size
        self.a_row_stop = (self.row_progress + 1) * self.asic_size

        print("asic_size", self.asic_size)
        print("a_col_start:", self.a_col_start)
        print("a_col_stop:", self.a_col_stop)
        print("a_row_start:", self.a_row_start)
        print("a_row_stop:", self.a_row_stop)

        self.raw_data_shape = (self.charges_per_file, self.n_mem_cells, 2, self.module_h, self.module_l)

        print("\n\nStart gathering")
        print("save_file =", self.save_file)
        print("")

        t = time.time()
        self.run()
        print("Total run time: ", time.time()-t)

    def check_output_file_exist(self):
        print("save_file =", self.save_file)
        if os.path.exists(self.save_file):
            print("Output file already exists\n")
            sys.exit(1)
        else:
            print("Output file: ok")

    def check_for_single_index(self):
        indices = []

        for i in range(len(self.files)):
            indices.append([])
            for f in self.files[i]:
                index = f.split("_")[-2]
                if index not in indices[i]:
                    indices[i].append(index)

        for index_list in indices:
            if len(index_list) < 1:
                print("More than one index found: {}\n".format(index_list))
                sys.exit(1)
            elif len(index_list) > 1:
                print("More than one index found: {}\n".format(index_list))
                sys.exit(1)

        print (indices)

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
            sys.exit(1)
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

        # shape of reshaped data: <number of mem cells>,
        #                         <images_per_mcell>,
        #                         <distinction between analog and digital>,
        #                         <asic_height>,
        #                         <asic_length>
        # for each memory cell
        #                  ____________                         ____________
        #                 |___________ |                       |___________ |
        #                |___________ ||                      |___________ ||
        #               |            |||   images_per_mcell  |            |||
        # module height |   analog   ||                      |  digital   ||
        #               |____________|                       |____________|
        #                module_length

        # i.e. totalframes, module_h, module_l
        #          -> charges_per_file, n_mem_cells, 2, asic_h, asic_l
        # e.g.totalframes, 128, 512 -> charges_per_file, 352, 2, 64, 64

        saveFile = h5py.File(self.save_file, "w", libver='latest')
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
        print("took time:", str(time.time() - t))

        print("Initiate digital data")
        t = time.time()
        # Creating the array with np.zero is faster than copying the array from analog
        digital = np.zeros(self.shape, dtype='int16')
        print("took time:", str(time.time() - t))

        n_cols = len(self.files)
        f = None

        print("n_cols", n_cols)

        try:
            for i in np.arange(n_cols):
                #TODO: find a generic solution for this
                if self.a_col_start == 0:
                    row_index_source = np.arange(self.a_row_start + (n_cols - 1) - i,
                                                 self.a_row_stop,
                                                 n_cols)
                    row_index_target = np.arange((n_cols - 1) - i,
                                                 self.asic_size,
                                                 n_cols)
                    #index_upper_row = np.arange(3 - i, 512, 4)
                else:
                    # the asics of the lowers row are upside down
                    row_index_source = np.arange(self.a_row_start + i,
                                                 self.a_row_stop,
                                                 n_cols)
                    row_index_target = np.arange(i, self.asic_size, n_cols)
                    #index_lower_row = np.arange(i, 512, 4)

                print("row_index_source", row_index_source)
                print("row_index_target", row_index_target)

                for j in np.arange(self.number_of_files):

                    print("Start loading", self.files[i][j])
                    t = time.time()
                    f = h5py.File(self.files[i][j], 'r', libver='latest')
                    raw_data = f[self.data_path_in_file][..., 0:self.module_h, 0:self.module_l]
                    print("took time:", str(time.time() - t))

                    print("Start reshaping")
                    reshaping_t_start = time.time()
                    raw_data.shape = self.raw_data_shape
                    #raw_data.shape = (self.charges_per_file, 352, 2, 128, 512)

                    dc_start = j * self.charges_per_file
                    dc_stop = (j + 1) * self.charges_per_file

                    print("Reshaping analog data")
                    t = time.time()
                    tmp = raw_data[:, :, 0, :, :]
                    analog[dc_start:dc_stop,
                           :,
                           :, #self.a_col_start:self.a_col_stop,
                           row_index_target] = tmp[..., self.a_col_start:self.a_col_stop,
                                                        row_index_source]
                    #analog[dc_start:dc_stop, :, 0:64, index_upper_row] = tmp[..., 0:64, index_upper_row]
                    #analog[dc_start:dc_stop, :, 64:, index_lower_row] = tmp[..., 64:, index_lower_row]
                    print("Reshaping of analog data took time:", str(time.time() - t))

                    print("Reshaping digital data")
                    t = time.time()
                    tmp = raw_data[:, :, 1, :, :]
                    digital[dc_start:dc_stop,
                            :,
                            :, #self.a_col_start:self.a_col_stop,
                            row_index_target] = tmp[..., self.a_col_start:self.a_col_stop,
                                                         row_index_source]
                    #digital[dc_start:dc_stop, :, 0:64, index_upper_row] = tmp[..., 0:64, index_upper_row]
                    #digital[dc_start:dc_stop, :, 64:, index_lower_row] = tmp[..., 64:, index_lower_row]
                    print("Reshaping of digital data took time:", str(time.time() - t))

                    print("Reshaping of data took time:", str(time.time() - reshaping_t_start))
                    f.close()
                    f = None

            t = time.time()
            print("\nStart saving")
            dset_analog[...] = analog
            dset_digital[...] = digital
            saveFile.flush()
            print("took time:", str(time.time() - t))
        finally:
            if f is not None:
                f.close()
            saveFile.close()


if __name__ == "__main__":
    rel_file_path = "311-312-301-300-310-234/temperature_m20C/drscs/itestc150"
    file_base_name = "M234_m8_drscs_itestc150"
    #file_base_name = "M301_m3_drscs_itestc150"
    output_file = "/gpfs/cfel/fsds/labs/processed/kuhnm/currentSource_chunked.h5"

    #column_specs = [15, 26, 37, 48]
    column_specs = [[15, 1], [26, 2], [37, 3], [48, 4]]

    max_part = False

    GatherData(rel_file_path, file_base_name, output_file, column_specs, max_part)
