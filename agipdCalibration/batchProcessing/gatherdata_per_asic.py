from __future__ import print_function

import h5py
import time
import sys
import numpy as np
import glob
import os
from string import Template


class GatherData():
    def __init__(self, asic, rel_file_path, file_base_name, output_file_name, col_spec, max_part):

        base_path = "/gpfs/cfel/fsds/labs/calibration/current/"
        filename_template = Template("${prefix}_col${column}")
        self.data_path_in_file = "/entry/instrument/detector/data"
        self.collection_path = "/entry/instrument/detector/collection"
        self.seq_number_path = "/entry/instrument/detector/sequence_number"
        self.total_lost_frames_path = "{}/total_loss_frames".format(self.collection_path)
        self.error_code_path = "{}/error_code".format(self.collection_path)

        file_path = os.path.join(base_path, rel_file_path, file_base_name)

        self.save_file = output_file_name
        self.check_output_file_exist()

        # [[<column>, <file index>],...]
        self.column_specs = col_spec

        self.mem_cells = 352
        self.module_h = 128  # in pixels
        self.module_l = 512  # in pixels
        self.asic_size = 64  # in pixels

        # how the asics are located on the module
        self.asic_mapping = [[16, 15, 14, 13, 12, 11, 10, 9],
                             [1,   2,  3,  4,  5,  6,  7, 8]]

        #                       [rows,                     columns]
        self.asics_per_module = [len(self.asic_mapping), len(self.asic_mapping[0])]
        self.index_map = range(self.asics_per_module[0] * self.asics_per_module[1])

        self.asic = asic
        print ("asic={}".format(self.asic))
        self.mapped_asic = self.calculate_mapped_asic()
        print ("mapped_asic={}".format(self.mapped_asic))

        self.a_row_start = None
        self.a_row_stop = None
        self.a_col_start = None
        self.a_col_stop = None
        self.determine_asic_border()

        self.file_name_prefix = []
        if type(self.column_specs[0]) == list:
            for c, i in self.column_specs:
                file_prefix = filename_template.substitute(prefix=file_path, column=c)

                # The method zfill() pads string on the left with zeros to fill width.
                self.file_name_prefix.append("{}_{}".format(file_prefix,
                                                            str(i).zfill(5)))
        else:
            for c in self.column_specs:
                self.file_name_prefix.append(
                    filename_template.substitute(prefix=file_path, column=c))

        print("\nUsed prefixes")
        self.files = []
        self.get_files()
        self.check_for_single_index()

        if max_part:
            for i in range(len(self.files)):
                self.files[i] = self.files[i][:max_part]

        self.check_file_number()

        self.charges_per_file = None
        self.charges = None
        self.get_charges()

        self.seq_number = None
        self.expected_nimages_per_file = 0
        self.treated_lost_frames = 0

        self.raw_data_shape = (self.charges_per_file, self.mem_cells, 2,
                               self.module_h, self.module_l)

        # pixel data from raw is written into an intermediate format before
        # it is transposed into the target shape
        self.intermediate_shape = (self.charges, self.mem_cells,
                                   self.asic_size, self.asic_size)

        self.target_shape = (self.asic_size, self.asic_size,
                             self.mem_cells, self.charges)
        self.chunksize = self.target_shape

        # (self.charges, self.mem_cells, self.asic_size, self.asic_size)
        # is transposed to
        # (self.asic_size, self.asic_size, self.mem_cells, self.charges)
        self.transpose_order = (2,3,1,0)

        self.analog = None
        self.digital = None

        print("\nStart gathering")
        print("save_file = {}\n".format(self.save_file))
        t = time.time()
        self.get_data()
        self.write_data()
        print("Total run time: {}".format(time.time() - t))

    def calculate_mapped_asic(self):
        for row_i in xrange(len(self.asic_mapping)):
            try:
                col_i = self.asic_mapping[row_i].index(self.asic)
                return self.index_map[row_i * self.asics_per_module[1] + col_i]
            except:
                pass

    def check_output_file_exist(self):
        print("save_file = {}".format(self.save_file))
        if os.path.exists(self.save_file):
            print("Output file already exists")
            sys.exit(1)
        else:
            print("Output file: ok")


    def get_files(self):
        self.files = []
        for p in self.file_name_prefix:
            self.files.append(glob.glob("{}*".format(p)))
            print(p)

        # to make sure that part_00000 comes before part_00001
        for file_list in self.files:
            file_list.sort()

        self.check_for_single_index()

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

    def get_charges(self):
        try:
            source_file = h5py.File(self.files[0][0], "r", libver="latest")

            # TODO: verify that the shape is always right and not dependant on frame loss
            source_shape = source_file[self.data_path_in_file].shape


            # if there is frame loss this is recognizable by a missing entry in seq_number
            self.seq_number = source_file[self.seq_number_path][()]

            # total_lost_frames are totally missing frames as well as
            # frames where only a package was lost
            total_lost_frames = source_file[self.total_lost_frames_path][0]

            # frames where a package is lost occure in error_code with
            # an nonzero entry whereas complety lost frames do not have
            # an entry in error code at all
            frames_with_pkg_loss = source_file[self.error_code_path][()].nonzero()
        except:
            print("Error when getting shape")
            raise
        finally:
            source_file.close()

        if source_shape[1] != self.module_h and source_shape[2] != self.module_l:
            print("Shape of file {} does not match requirements".format(fname))
            print("source_shape = {}".format(source_shape))
            sys.exit(1)

        self.expected_nimages_per_file = (len(self.seq_number)
                                          # for the very first file
                                          # total_lost_frames was not yet aggregated
                                          + total_lost_frames
                                          - len(np.hstack(frames_with_pkg_loss)))

        self.charges_per_file = int(self.expected_nimages_per_file / 2 / self.mem_cells)
        self.charges = self.charges_per_file * self.number_of_files

    def determine_asic_border(self):
        #       ____ ____ ____ ____ ____ ____ ____ ____
        # 0x64 |    |    |    |    |    |    |    |    |
        #      |  0 |  1 | 2  | 3  |  4 |  5 | 6  | 7  |
        # 1x64 |____|____|____|____|____|____|____|____|
        #      |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
        # 2x64 |____|____|____|____|____|____|____|____|
        #      0*64 1x64 2x64 3x64 4x64 5x64 6x64 7x64 8x64

        row_progress = self.mapped_asic / self.asics_per_module[1]
        col_progress = self.mapped_asic % self.asics_per_module[1]
        print("row_progress: {}".format(row_progress))
        print("col_progress: {}".format(col_progress))

        self.a_row_start = row_progress * self.asic_size
        self.a_row_stop = (row_progress + 1) * self.asic_size
        self.a_col_start = col_progress * self.asic_size
        self.a_col_stop = (col_progress + 1) * self.asic_size

        print("asic_size {}".format(self.asic_size))
        print("a_col_start: {}".format(self.a_row_start))
        print("a_row_stop: {}".format(self.a_row_stop))
        print("a_row_start: {}".format(self.a_col_start))
        print("a_row_stop: {}".format(self.a_col_stop))

    def init_metadata(self):
        collection = {}

        #self.save_file.create_group(self.collection_path)
        try:
            source_file = h5py.File(self.files[0][0], "r", libver="latest")

            for k in source_file[self.collection_path].keys():
                collection[k] = {
                    "path": "{}/{}".format(self.collection_path, k),
                    "shape": dset.shape,
                    "dtype": dset.dtype,
                    "value": None
                }
        except:
            print("Error when initiating metadata")
            raise
        finally:
            source_file.close()

    def get_data(self):
        """
        shape of input data: <total image number (analog and digital)>,
                             <module height>,
                             <module length>

                         ____________
                        |___________ |
                       |___________ ||
                      |            |||    analog and digital images
        module height |            ||
                      |____________|
                       module_length

        serparate data sets for analog and digital
        shape of reshaped data: <number of mem cells>,
                                <images_per_mcell>,
                                <asic_height>,
                                <asic_length>
        for each memory cell
                         ____________                         ____________
                        |___________ |                       |___________ |
                       |___________ ||                      |___________ ||
                      |            |||   images_per_mcell  |            |||
        module height |   analog   ||                      |  digital   ||
                      |____________|                       |____________|
                       module_length

        i.e. totalframes, module_h, module_l
                 -> charges_per_file, mem_cells, asic_h, asic_l
        e.g.totalframes, 128, 512 -> charges_per_file, 352, 64, 64
        """

        print("Initiate analog data")
        t = time.time()
        self.analog = np.zeros(self.intermediate_shape, dtype="int16")
        print("took time: {}".format(time.time() - t))

        print("Initiate digital data")
        t = time.time()
        # Creating the array with np.zero is faster than copying the array from analog
        self.digital = np.zeros(self.intermediate_shape, dtype="int16")
        print("took time: {}".format(time.time() - t))

        n_cols = len(self.files)
        print("n_cols {}".format(n_cols))

        source_file = None
        try:
            for i in np.arange(n_cols):
                #TODO: find a generic solution for this
                if self.a_row_start == 0:
                    col_index_source = np.arange(self.a_col_start + (n_cols - 1) - i,
                                                 self.a_col_stop,
                                                 n_cols)
                    col_index_target = np.arange((n_cols - 1) - i,
                                                 self.asic_size,
                                                 n_cols)
                    #index_upper_row = np.arange(3 - i, 512, 4)
                else:
                    # the asics of the lower rows are upside down
                    col_index_source = np.arange(self.a_col_start + i,
                                                 self.a_col_stop,
                                                 n_cols)
                    col_index_target = np.arange(i, self.asic_size, n_cols)
                    #index_lower_row = np.arange(i, 512, 4)

                print("col_index_source {}".format(col_index_source))
                print("col_index_target {}".format(col_index_target))

                for j in np.arange(self.number_of_files):
                    print("\nStarting file {}".format(self.files[i][j]))
                    source_file = h5py.File(self.files[i][j], "r", libver="latest")

                    print("Start sequence number loading")
                    t = time.time()
                    # if there is frame loss this is recognizable by a missing entry in seq_number
                    # seq_number should be loaded before duing operations on
                    # it due to performance benefits
                    source_seq_number = source_file[self.seq_number_path][()]
                    self.seq_number = (source_seq_number
                                       # seq_number starts counting with 1
                                       - 1
                                       # the seq_number refers to the whole run not one file
                                       - j * self.expected_nimages_per_file)
                    print("took time: {}".format(time.time() - t))
                    print("seq_number: {}".format(self.seq_number))

                    source_shape, expected_shape = self.determine_expected_shape(source_file)
                    print("expected_shape={}".format(expected_shape))

                    print("Start data loading")
                    t = time.time()
                    loaded_raw_data = source_file[self.data_path_in_file][()]
                    print("took time: {}".format(time.time() - t))

                    if source_shape == expected_shape:
                        raw_data = loaded_raw_data
                    else:
                        print("Start initializing raw data with zeros")
                        t = time.time()
                        raw_data = np.zeros(expected_shape)
                        print("took time: {}".format(time.time() - t))

                        print("Start getting data blocks")
                        t = time.time()
                        self.fillup_frame_loss(raw_data, loaded_raw_data)
                        print("took time: {}".format(time.time() - t))

                    print("Start reshaping")
                    reshaping_t_start = time.time()
                    print("raw_data_shape={}".format(raw_data.shape))
                    raw_data.shape = self.raw_data_shape

                    dc_start = j * self.charges_per_file
                    dc_stop = (j + 1) * self.charges_per_file

                    # raw_data:       charges_per_file, mem_cells, 2, module_h, module_l
                    # analod/digital: charges_per_file, mem_cells, asic_h, asic_l
                    tmp = raw_data[:, :, 0, :, :]
                    self.analog[dc_start:dc_stop, :, :,
                                col_index_target] = tmp[..., self.a_row_start:self.a_row_stop,
                                                             col_index_source]
                    tmp = raw_data[:, :, 1, :, :]
                    self.digital[dc_start:dc_stop, :, :,
                                 col_index_target] = tmp[..., self.a_row_start:self.a_row_stop,
                                                              col_index_source]

                    print("Reshaping of data took time: {}".format(time.time() - reshaping_t_start))
                    source_file.close()
                    source_file = None

        finally:
            if source_file is not None:
                source_file.close()
                source_file = None

    def determine_expected_shape(self, source_file):
        # TODO: verify that the shape is always right and not dependant on frame loss
        source_shape = source_file[self.data_path_in_file].shape
        print("source_shape reading from the file {}".format(source_shape))

        if source_shape[1] != self.module_h and source_shape[2] != self.module_l:
            print("Shape does not match requirements")
            print("source_shape = {}".format(source_shape))
            sys.exit(1)

        # total_lost_frames are totally missing frames as well as
        # frames where only a package was lost
        total_lost_frames = source_file[self.total_lost_frames_path][0]

        # total_lost_frames does not refer to the current file but the total run
        new_lost_frames = total_lost_frames - self.treated_lost_frames

        # frames where a package is lost occure in error_code with
        # an nonzero entry whereas complety lost frames do not have
        # an entry in error code at all
        error_code = source_file[self.error_code_path][()]
        frames_with_pkg_loss = error_code.nonzero()
        number_of_frames_with_pkg_loss = len(np.hstack(frames_with_pkg_loss))
        print("Frames with packet loss: {}".format(number_of_frames_with_pkg_loss))

        print("len seq_number: {}".format(len(self.seq_number)))
        print("new_lost_frames: {}".format(new_lost_frames))

        self.expected_nimages_per_file = (len(self.seq_number)
                                          + new_lost_frames
                                          - number_of_frames_with_pkg_loss)

        self.treated_lost_frames = total_lost_frames
        print("treated_lost_frames = {}".format(self.treated_lost_frames))

        expected_shape = (self.expected_nimages_per_file, source_shape[1], source_shape[2])

        return source_shape, expected_shape

    def fillup_frame_loss(self, raw_data, loaded_raw_data):
        # The borders (regarding the expected_shape) of
        # continuous blocks of data written into the target
        # (in between these blocks there will be zeros)
        target_index = [[0, 0]]
        # The borders (regarding the source_shape) of
        # continuous blocks of data read from the source
        # (no elements in between these blocks)
        source_index = []
        stop = 0
        for i in np.arange(len(self.seq_number)):

            # a gap in the numbering occured
            if stop - self.seq_number[i] < -1:
                # the number before the gab gives the end of
                # the continuous block of data
                target_index[-1][1] = stop
                # the next block starts now
                target_index.append([self.seq_number[i], 0])
                # the end of the block in the source
                source_index.append(i)

            stop = self.seq_number[i]

        # the last block ends with the end of the data
        target_index[-1][1] = self.seq_number[-1]
        source_index.append(self.expected_nimages_per_file)
        print("target_index {}".format(target_index))
        print("source_index {}".format(source_index))

        # getting the blocks from source to target
        s_start = 0
        for i in range(len(target_index)):

            # start and stop of the block in the target
            # [t_start, t_stop)
            t_start = target_index[i][0]
            t_stop = target_index[i][1] + 1

            # start and stop of the block in the source
            # s_start was set in the previous loop iteration
            # (or for i=0 is set to 0)
            s_stop = source_index[i]

            raw_data[t_start:t_stop, ...] = loaded_raw_data[s_start:s_stop, ...]
            s_start = source_index[i]

    def extract_metadata(self, source_file):
        pass

    def write_data(self):
        """
        transposes the data dimensions for optimal analysis access
        e.g. charges, mem_cells, asic_h, asic_l -> asic_h, asic_l, mem_cells, charges
        and writes it into a file
        """

        save_file = h5py.File(self.save_file, "w", libver='latest')
        print("Create analog data set")
        dset_analog = save_file.create_dataset("analog",
                                               shape=self.target_shape,
                                               chunks=self.chunksize,
                                               compression=None, dtype='int16')
        print("Create digital data set")
        dset_digital = save_file.create_dataset("digital",
                                               shape=self.target_shape,
                                               chunks=self.chunksize,
                                               compression=None, dtype='int16')

        try:
            t = time.time()
            print("\nStart transposing")
            analog_transposed = self.analog.transpose(self.transpose_order)
            digital_transposed = self.digital.transpose(self.transpose_order)
            print("took time: {}".format(time.time() - t))

            t = time.time()
            print("\nStart saving")
            dset_analog[...] = analog_transposed
            dset_digital[...] = digital_transposed
            save_file.flush()
            print("took time: {}".format(time.time() - t))
        finally:
            save_file.close()


if __name__ == "__main__":
    rel_file_path = "311-312-301-300-310-234/temperature_m20C/drscs/itestc150"
    file_base_name = "M234_m8_drscs_itestc150"
    #file_base_name = "M301_m3_drscs_itestc150"
    output_file = "/gpfs/cfel/fsds/labs/processed/kuhnm/currentSource_chunked.h5"

    #column_specs = [15, 26, 37, 48]
    column_specs = [[15, 1], [26, 2], [37, 3], [48, 4]]

    max_part = False
    asic = 7
    GatherData(asic, rel_file_path, file_base_name, output_file, column_specs, max_part)
