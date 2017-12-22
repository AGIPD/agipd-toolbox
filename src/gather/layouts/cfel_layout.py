import h5py
import numpy as np
import os
import sys
import time
import glob
import configparser

try:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
except:
    CURRENT_DIR = os.path.dirname(os.path.realpath('__file__'))

BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))
SRC_PATH = os.path.join(BASE_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import utils  # noqa E402
import cfel_optarg  # noqa E402


class CfelLayout():
    def __init__(self,
                 in_fname,
                 runs,
                 use_interleaved,
                 preproc_fname=None,
                 max_part=False,
                 asic=None):

        self.in_fname = in_fname
        self.runs = runs
        self.use_interleaved
        self.preprocessing_fname = preproc_fname
        self.max_part = max_part
        self.asic = asic

    def initiate(self):
        self.data_path = "entry/instrument/detector/data"
        self.digital_data_path = "entry/instrument/detector/data_digital"

        self.collection_path = "entry/instrument/detector/collection"
        self.seq_number_path = "entry/instrument/detector/sequence_number"
        self.total_lost_frames_path = ("{}/total_loss_frames"
                                       .format(self.collection_path))
        self.error_code_path = "{}/error_code".format(self.collection_path)
        self.frame_number_path = ("{}/frame_numbers"
                                  .format(self.collection_path))

        run_number = self.runs[0]
        fname = self.in_fname.format(run_number=run_number, part=0)

        f = None
        try:
            f = h5py.File(fname, "r")
            raw_data_shape = f[self.data_path].shape
        finally:
            if f is not None:
                f.close()

        # dark
        self.max_pulses = 704
        self.n_memcells = 352
        # xray
#        self.max_pulses = 2
#        self.n_memcells = 1

        self.get_number_of_frames()
        print("n_frames_total {}".format(self.n_frames_total))

        self.raw_shape = (self.n_memcells, 2, self.n_rows, self.n_cols)

    def get_number_of_frames(self):
        f = None

        # take first run and asume that the others have as many frames
        # TODO check this
        run_number = self.runs[0]
        self.max_pulses = 0
        n_trains = 0

        try:
            fname = self.in_fname.format(run_number=run_number, part=0)
            f = h5py.File(fname, "r", libver="latest", drivers="core")

            # TODO: verify that the shape is always right and not
            #       dependant on frame loss
            source_shape = f[self.data_path].shape
            exp_total_frames = f[self.frame_number_path][0]

        except:
            print("Error when getting shape")
            raise
        finally:
            if f is not None:
                f.close()

        if (source_shape[1] != self.n_rows_total and
                source_shape[2] != self.n_cols_total):
            msg = "Shape of file {} ".format(fname)
            msg += "does not match requirements\n"
            msg += "source_shape = {}".format(source_shape)
            raise RuntimeError(msg)

        self.n_frames_total = int(exp_total_frames)

    def load(self,
             fname,
             seq,
             load_idx_rows,
             load_idx_cols,
             file_content,
             tmp_data):
        # load data
        f = None
        try:
            f = h5py.File(fname, "r")
            idx = (Ellipsis, load_idx_rows, load_idx_cols)
            raw_data = f[self.data_path][idx]
        finally:
            if f is not None:
                f.close()

        print("raw_data.shape", raw_data.shape)
        print("self.raw_shape", self.raw_shape)
        self.get_seq_number(file_content[self.seq_number_path])
        self.get_frame_loss_indices()
        self.fillup_frame_loss(tmp_data,
                               raw_data,
                               self.target_index_full_size)

    def get_seq_number(self, source_seq_number):
        # if there is frame loss this is recognizable by a missing
        # entry in seq_number. seq_number should be loaded before
        # doing operations on it due to performance benefits
        seq_number_last_entry_previous_file = self.source_seq_number[-1]
        self.source_seq_number = source_seq_number

        print("seq_number_last_entry_previous_file={}"
              .format(seq_number_last_entry_previous_file))
        print("seq_number before modifying: {}"
              .format(self.source_seq_number))
        self.seq_number = (self.source_seq_number
                           # seq_number starts counting with 1
                           - 1
                           # the seq_number refers to the whole run
                           # not one file
                           - seq_number_last_entry_previous_file)
        print("seq_number: {}".format(self.seq_number))

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
        stop_full_size = 0
        stop_source = 0
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
                seqlst = [self.source_seq_number[i] - 1, 0]
                self.target_index_full_size.append(seqlst)

                self.source_index[-1][1] = stop_source
                # the end of the block in the source
                self.source_index.append([i, 0])

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

        # check to see the values of the sequence number in the first frame
        # loss gap
        if len(self.target_index_full_size) > 1:
            start = self.source_index[0][1] - 1
            stop = self.source_index[1][0] + 2
            seq_num = self.source_seq_number[start:stop]
            print("seq number in first frame loss region: {}"
                  .format(seq_num))

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

            for index_set in self.pos_idxs:
                pos_idx_rows = index_set[0]
                pos_idx_cols = index_set[1]

                raw_idx = (slice(t_start, t_stop),
                           Ellipsis,
                           pos_idx_rows,
                           pos_idx_cols)
                loaded_idx = (slice(s_start, s_stop),
                              Ellipsis,
                              pos_idx_rows,
                              pos_idx_cols)

                raw_data[raw_idx] = loaded_raw_data[loaded_idx]
