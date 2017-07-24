from __future__ import print_function

import h5py
import time
import sys
import numpy as np
import glob
import os
from string import Template


class GatherData():
    def __init__(self, asic, input_file, output_file_name, meas_type, max_part, col_spec=[0]):
        print("Using numpy version {}".format(np.__version__))
        print("Using h5py version {}".format(h5py.__version__))

        filename_template = Template("${prefix}_col${column}")
        self.data_path = "/entry/instrument/detector/data"
        self.digital_data_path = "/entry/instrument/detector/data_digital"
        self.collection_path = "/entry/instrument/detector/collection"
        self.seq_number_path = "/entry/instrument/detector/sequence_number"
        self.total_lost_frames_path = "{}/total_loss_frames".format(self.collection_path)
        self.error_code_path = "{}/error_code".format(self.collection_path)
        self.frame_number_path = "{}/frame_numbers".format(self.collection_path)

        self.save_file = output_file_name
        self.check_output_file_exist()

        self.measurement_type = meas_type

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
                file_prefix = filename_template.substitute(prefix=input_file, column=c)

                # The method zfill() pads string on the left with zeros to fill width.
                self.file_name_prefix.append("{}_{}".format(file_prefix,
                                                            str(i).zfill(5)))
        else:
            for c in self.column_specs:
                self.file_name_prefix.append(
                    filename_template.substitute(prefix=input_file, column=c))

        print("\nUsed prefixes")
        self.files = []
        self.get_files()
        self.check_for_single_index()

        if max_part:
            for i in range(len(self.files)):
                self.files[i] = self.files[i][:max_part]

        self.check_file_number()

        self.metadata = {}
        self.metadata_tmp = {}
        self.metadata_derived = {}
        # the metadata for which derived data is calculated
        # the sequence number is implied in this
        self.special_keys = ["error_code", "total_loss_frames"]

        self.source_seq_number = None
        self.seq_number = None
        self.expected_total_images = 0
        self.expected_nimages_per_file = 0
        self.treated_lost_frames = 0

        self.target_index = None
        self.source_index = None

        self.charges = None
        self.get_charges()

        self.raw_data_shape = (self.expected_total_images,
                               self.asic_size, self.asic_size)
        # pixel data from raw is written into an intermediate format before
        # it is transposed into the target shape
        self.reshaped_data_shape = (self.charges, self.mem_cells, 2,
                                    self.asic_size, self.asic_size)

        # reshaped data is split into analog and digital data + transposed
        self.target_shape = (self.asic_size, self.asic_size,
                             self.mem_cells, self.charges)
        self.chunksize = self.target_shape

        self.raw_frame_loss_shape = (self.charges, self.mem_cells)

        # (self.charges, self.mem_cells, self.asic_size, self.asic_size)
        # is transposed to
        # (self.asic_size, self.asic_size, self.mem_cells, self.charges)
        self.transpose_order = (2,3,1,0)

        # (self.charges, self.mem_cells)
        # is transposed to
        # (self.mem_cells, self.charges)
        self.frame_loss_transpose_order = (1, 0)

        self.analog = None
        self.digital = None

        print("\nStart gathering")
        print("save_file = {}\n".format(self.save_file))
        t = time.time()
        self.get_data()
        self.write_data()
        print("Total run time: {}".format(time.time() - t))

    def calculate_mapped_asic(self):
        for row_i in np.arange(len(self.asic_mapping)):
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

        source_file = None
        try:
            source_file = h5py.File(self.files[0][0], "r", libver="latest", drivers="core")

            # TODO: verify that the shape is always right and not dependant on frame loss
            source_shape = source_file[self.data_path].shape

            self.expected_total_images = source_file[self.frame_number_path][0]

            # if there is frame loss this is recognizable by a missing entry in seq_number
            self.seq_number = source_file[self.seq_number_path][()]
            print("len seq_number: {}".format(len(self.seq_number)))

            # total_lost_frames are totally missing frames as well as
            # frames where only a package was lost
            total_lost_frames = source_file[self.total_lost_frames_path][0]

            # frames where a package is lost occure in error_code with
            # an nonzero entry whereas complety lost frames do not have
            # an entry in error code at all
            error_code = source_file[self.error_code_path][()]
            frames_with_pkg_loss = error_code.nonzero()
            number_of_frames_with_pkg_loss = len(np.hstack(frames_with_pkg_loss))
            print("Frames with packet loss: {}".format(number_of_frames_with_pkg_loss))
        except:
            print("Error when getting shape")
            raise
        finally:
            if source_file is not None:
                source_file.close()

        if source_shape[1] != self.module_h and source_shape[2] != self.module_l:
            print("Shape of file {} does not match requirements".format(fname))
            print("source_shape = {}".format(source_shape))
            sys.exit(1)

        # due to a "bug" in Tango the total_lost_frames can only be found in the next file
        # TODO once Tango has fixed this, adjust it here as well
        source_file = None
        try:
            source_file = h5py.File(self.files[0][1], "r", libver="latest", driver="core")
            # total_lost_frames are totally missing frames as well as
            # frames where only a package was lost
            total_lost_frames = source_file[self.total_lost_frames_path][0]
        except:
            print("Error when getting shape")
            raise
        finally:
            if source_file is not None:
                source_file.close()

        self.expected_nimages_per_file = (len(self.seq_number)
                                          # for the very first file
                                          # total_lost_frames was not yet aggregated
                                          + total_lost_frames
                                          - number_of_frames_with_pkg_loss)

        self.charges = int(self.expected_total_images / 2 / self.mem_cells)

        print("expected_total_images={}".format(self.expected_total_images))
        print("expected_nimages_per_file={}".format(self.expected_nimages_per_file))
        print("charges={}".format(self.charges))

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
        #self.save_file.create_group(self.collection_path)
        try:
            source_file = h5py.File(self.files[0][0], "r", libver="latest", driver="core")

            ### normal metadata ###
            # metadata which is not handled individually is gathered
            # in a matrix of the shape
            # (<number of patterns>, <number of file parts>, <original metadata shape>)
            for k in source_file[self.collection_path].keys():
                dset_path = "{}/{}".format(self.collection_path, k)

                if k not in self.special_keys:
                    dset = source_file[dset_path]
                    shape = (len(self.files), self.number_of_files) + dset.shape

                    self.metadata[k] = {
                        "path": dset_path,
                        "value": np.zeros(shape, dset.dtype)
                    }

            ### special metadata ###
            # for special metadata it makes no sense to just gather it
            # for this data derived values are stored

            dset_path = "{}/gathered_files".format(self.collection_path)
            self.metadata_derived["gathered_files"] = {
                "path": dset_path,
                "value": self.files
            }

            # aggregated over all parts but distinguished between file types
            dset_path = "{}/total_loss_frames".format(self.collection_path)
            dset_dtype = source_file[dset_path].dtype
            self.metadata_derived["total_loss_frames"] = {
                "path": dset_path,
                "value": np.zeros(len(self.column_specs), dset_dtype)
            }

            # lists can have different length
            dset_path = "{}/error_code".format(self.collection_path)
            dset_dtype = source_file[dset_path].dtype
            shape = (len(self.files), self.number_of_files, self.expected_nimages_per_file)
            self.metadata_derived["error_code"] = {
                "path": dset_path,
                "value": np.zeros(shape, dset_dtype)
            }

            # lists can have different length
            dset_dtype = source_file[self.seq_number_path].dtype
            shape = (len(self.files), self.number_of_files, self.expected_nimages_per_file)
            self.metadata_derived["sequence_number"] = {
                "path": self.seq_number_path,
                "value": np.zeros(shape, dset_dtype)
#                "value": [[[] for _ in t] for t in self.files]
            }

            dset_path = "{}/frame_loss_analog".format(self.collection_path)
            self.metadata_derived["frame_loss_analog"] = {
                "path": dset_path,
                "value": [[] for _ in self.files]
            }

            dset_path = "{}/frame_loss_digital".format(self.collection_path)
            self.metadata_derived["frame_loss_digital"] = {
                "path": dset_path,
                "value": [[] for _ in self.files]
            }

            # meaning:
            # -1: frame loss
            #  0: frame ok
            #  1: frame damaged
            shape = (
                # one entry for every file type
                (len(self.column_specs),
                # merged file parts without frame loss
                self.expected_total_images))
            print("setting frame_loss_details to shape: {}".format(shape))
            # initiate with -1
            self.metadata_tmp["frame_loss_details"] = -1 * np.ones(shape, "int32")

            self.metadata_tmp["source_seq_number"] = [0]


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
                 -> charges, mem_cells, asic_h, asic_l
        e.g.totalframes, 128, 512 -> charges, 352, 64, 64
        """

        print("Initiate tmp data")
        t = time.time()
        # Creating the array with np.zero is faster than copying the array from analog
        self.tmp_data_real_position = np.zeros(self.raw_data_shape, dtype="int16")
        print("took time: {}".format(time.time() - t))

        self.init_metadata()

        n_cols = len(self.files)
        print("n_cols {}".format(n_cols))

        source_file = None
        try:
            for i in np.arange(n_cols):

                print("Initiate tmp data")
                t = time.time()
                # Creating the array with np.zero is faster than copying the array from analog
                self.tmp_data = np.zeros(self.raw_data_shape, dtype="int16")
                print("took time: {}".format(time.time() - t))

                #TODO: find a generic solution for this
                if self.a_row_start == 0:
                    col_index_module = np.arange(self.a_col_start + (n_cols - 1) - i,
                                                 self.a_col_stop,
                                                 n_cols)
                    col_index_asic = np.arange((n_cols - 1) - i,
                                               self.asic_size,
                                               n_cols)
                    #index_upper_row = np.arange(3 - i, 512, 4)
                else:
                    # the asics of the lower rows are upside down
                    col_index_module = np.arange(self.a_col_start + i,
                                                 self.a_col_stop,
                                                 n_cols)
                    col_index_asic = np.arange(i, self.asic_size, n_cols)
                    #index_lower_row = np.arange(i, 512, 4)

                print("col_index_module {}".format(col_index_module))
                print("col_index_asic {}".format(col_index_asic))

                self.source_seq_number = [0]
                for j in np.arange(self.number_of_files):
                    file_processing_time = time.time()

                    print("\nStarting file {}".format(self.files[i][j]))
                    t = time.time()
                    source_file = h5py.File(self.files[i][j], "r", libver="latest", driver="core")
                    print("Opening file took {}".format(time.time() - t))

                    print("Getting metadata")
                    t = time.time()
                    self.get_metadata(source_file, i, j)
                    print("took time: {}".format(time.time() - t))

                    # if there is frame loss this is recognizable by a missing entry in seq_number
                    # seq_number should be loaded before duing operations on
                    # it due to performance benefits
                    seq_number_last_entry_previous_file = self.source_seq_number[-1]
                    self.source_seq_number = self.metadata_tmp["source_seq_number"]
                    print("seq_number_last_entry_previous_file={}"
                          .format(seq_number_last_entry_previous_file))
                    print("seq_number before modifying: {}".format(self.source_seq_number))
                    self.seq_number = (self.source_seq_number
                                       # seq_number starts counting with 1
                                       - 1
                                       # the seq_number refers to the whole run not one file
                                       - seq_number_last_entry_previous_file)
                                       #- j * self.expected_nimages_per_file)
                    print("seq_number: {}".format(self.seq_number))

                    self.get_frame_loss_indices()
                    self.get_tmp_metadata(source_file, i, j)

                    print("Start data loading")
                    t = time.time()
                    loaded_raw_data = source_file[self.data_path][:, self.a_row_start:self.a_row_stop,
                                                                     self.a_col_start:self.a_col_stop]
                    print("took time: {}".format(time.time() - t))

                    source_file.close()
                    source_file = None

                    print("Start getting data blocks")
                    t = time.time()
                    self.fillup_frame_loss(self.tmp_data, loaded_raw_data, self.target_index_full_size)
                    print("took time: {}".format(time.time() - t))

                    print("Processing the file took: {}".format(time.time() - file_processing_time))

                # this is done on the end of the file type handling due to performance reasons
                print("Start getting columns")
                t = time.time()
                self.tmp_data_real_position[..., col_index_asic] = self.tmp_data[..., col_index_asic]
                print("took time: {}".format(time.time() - t))

        finally:
            if source_file is not None:
                source_file.close()
                source_file = None

        print("Start reshaping")
        reshaping_t_start = time.time()
        print("tmp_data_shape={}".format(self.tmp_data_real_position.shape))
        print("reshaped_data_shape={}".format(self.reshaped_data_shape))
        self.tmp_data_real_position.shape = self.reshaped_data_shape

        # raw_data:       charges, mem_cells, 2, module_h, module_l
        # analod/digital: charges, mem_cells, asic_h, asic_l
        self.analog = self.tmp_data_real_position[:, :, 0, :, :]
        self.digital = self.tmp_data_real_position[:, :, 1, :, :]

        print("Reshaping of data took time: {}".format(time.time() - reshaping_t_start))

    def get_frame_loss_indices(self):
        # The borders (regarding the expected shape) of
        # continuous blocks of data written into the target
        # (in between these blocks there will be zeros)
        self.target_index = [[0, 0]]
        # original sequence number starts with 1
        self.target_index_full_size = [[self.source_seq_number[0] - 1, 0]]
        # The borders (regarding the source_shape) of
        # continuous blocks of data read from the source
        # (no elements in between these blocks)
        self.source_index = [[0, 0]]
        stop = 0
        for i in np.arange(len(self.seq_number)):

            # a gap in the numbering occured
            if stop - self.seq_number[i] < -1:

                # the number before the gab gives the end of
                # the continuous block of data
                self.target_index[-1][1] = stop
                # the next block starts now
                self.target_index.append([self.seq_number[i], 0])

                # the number before the gab gives the end of
                # the continuous block of data in the fully sized array
                self.target_index_full_size[-1][1] = stop_full_size
                # the next block starts now
                # original sequence number started with 1
                self.target_index_full_size.append([self.source_seq_number[i] - 1, 0])

                self.source_index[-1][1] = stop_source
                # the end of the block in the source
                self.source_index.append([i, 0])

                print("seq_number[{}]={}, seq_number[{}]={}"
                      .format(i-1, self.seq_number[i-1], i, self.seq_number[i]))
                print("source_seq_number[{}]={}, source_seq_number[{}]={}"
                      .format(i-1, self.source_seq_number[i-1], i, self.source_seq_number[i]))

            stop_source = i
            stop = self.seq_number[i]
            # original sequence number started with 1
            stop_full_size = self.source_seq_number[i] - 1

        # the last block ends with the end of the data
        self.target_index[-1][1] = self.seq_number[-1]
        self.target_index_full_size[-1][1] = self.source_seq_number[-1] - 1
        self.source_index[-1][1] = len(self.seq_number) - 1
        print("target_index {}".format(self.target_index))
        print("target_index_full_size {}".format(self.target_index_full_size))
        print("source_index {}".format(self.source_index))

        if len(self.target_index_full_size) > 1:
            print("sequence_number: {}".format(
                self.source_seq_number[(self.target_index[0][1] - 1):(self.target_index[1][0] + 1)]))

    def fillup_frame_loss(self, raw_data, loaded_raw_data, target_index):
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
            s_start = self.source_index[i][0]
            s_stop = self.source_index[i][1] + 1

            print("t_start = {}".format(t_start))
            print("t_stop = {}".format(t_stop))
            print("s_start = {}".format(s_start))
            print("s_stop = {}".format(s_stop))

            #print("loaded_raw_data = {}".format(loaded_raw_data[s_start:s_stop, ...]))
            raw_data[t_start:t_stop, ...] = loaded_raw_data[s_start:s_stop, ...]
            #print("after raw_data = {}".format(raw_data[t_start:t_stop, ...]))

            # debug output
            if raw_data.shape == self.raw_data_shape and t_start != 0:
                print_start = t_start - 2
                print_stop = t_start + 2
                print("raw_data in frame_loss region={}"
                      .format(raw_data[(print_start):(print_stop), 0, 0]))

    def get_metadata(self, source_file, column_index, part_index):

        ### normal metadata ###
        for k in self.metadata:
            dset_path = self.metadata[k]["path"]
            dset = source_file[dset_path][()]
            self.metadata[k]["value"][column_index, part_index, ...] = dset

        ### special metadata ###

        # get total_loss_frames
        dset_path = self.metadata_derived["total_loss_frames"]["path"]
        self.metadata_derived["total_loss_frames"]["value"][column_index] = source_file[dset_path][()]

        # merge lists of error codes
        dset_path = self.metadata_derived["error_code"]["path"]
        dset = source_file[dset_path][()]
        self.metadata_derived["error_code"]["value"][column_index][part_index][:dset.shape[0]] = dset

        # merge lists of error codes
        dset_path = self.metadata_derived["sequence_number"]["path"]
        dset = source_file[dset_path][()]
        self.metadata_derived["sequence_number"]["value"][column_index][part_index][:dset.shape[0]] = dset

        self.metadata_tmp["source_seq_number"] = dset

    def get_tmp_metadata(self, source_file, column_index, part_index):

        dset = self.metadata_derived["error_code"]["value"][column_index][part_index]

        # get aggregated information about frame and package loss
        self.fillup_frame_loss(self.metadata_tmp["frame_loss_details"][column_index],
                               dset,
                               self.target_index_full_size)

    def write_data(self):
        """
        transposes the data dimensions for optimal analysis access
        e.g. charges, mem_cells, asic_h, asic_l -> asic_h, asic_l, mem_cells, charges
        and writes it into a file
        """

        save_file = h5py.File(self.save_file, "w", libver="latest")
        print("Create analog data set")
        dset_analog = save_file.create_dataset(self.data_path,
                                               shape=self.target_shape,
                                               chunks=self.chunksize,
                                               compression=None, dtype='int16')
        print("Create digital data set")
        dset_digital = save_file.create_dataset(self.digital_data_path,
                                               shape=self.target_shape,
                                               chunks=self.chunksize,
                                               compression=None, dtype='int16')

        try:
            print("\nStart saving metadata")
            t = time.time()
            self.write_metadata(save_file)
            print("took time: {}".format(time.time() - t))

            print("\nStart transposing")
            t = time.time()
            analog_transposed = self.analog.transpose(self.transpose_order)
            digital_transposed = self.digital.transpose(self.transpose_order)
            print("took time: {}".format(time.time() - t))

            print("\nStart saving data")
            t = time.time()
            dset_analog[...] = analog_transposed
            dset_digital[...] = digital_transposed
            save_file.flush()
            print("took time: {}".format(time.time() - t))
        finally:
            save_file.close()

    def extend_metadata(self):

        for i in np.arange(len(self.files)):
            dset = self.metadata_tmp["frame_loss_details"][i]
            print("{}: lost frames: {}".format(i, np.where(dset == -1)))

            dset_analog = dset[::2]
            dset_analog.shape = self.raw_frame_loss_shape
            dset_analog = dset_analog.transpose(self.frame_loss_transpose_order)

            dset_digital = dset[1::2]
            dset_digital.shape = self.raw_frame_loss_shape
            dset_digital = dset_digital.transpose(self.frame_loss_transpose_order)

            self.metadata_derived["frame_loss_analog"]["value"][i] = dset_analog
            self.metadata_derived["frame_loss_digital"]["value"][i] = dset_digital

    def write_metadata(self, target):

        self.extend_metadata()

        ### normal metadata ###
        for k in self.metadata:
            dset_path = self.metadata[k]["path"]
            dset = self.metadata[k]["value"]

            target.create_dataset(dset_path, data=dset)

        ### special metadata ###
        for k in self.metadata_derived:
            dset_path = self.metadata_derived[k]["path"]
            dset = self.metadata_derived[k]["value"]

            target.create_dataset(dset_path, data=dset)

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
