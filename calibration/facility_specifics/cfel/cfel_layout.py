# (c) Copyright 2017-2018 DESY, FS-DS
#
# This file is part of the FS-DS AGIPD toolbox.
#
# This software is free: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software.  If not, see <http://www.gnu.org/licenses/>.

"""
@author: Manuela Kuhn <manuela.kuhn@desy.de>
         Jennifer Poehlsen <jennifer.poehlsen@desy.de>
"""

import h5py
import numpy as np
import os
import sys
import glob

from __init__ import SRC_DIR

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import utils


class Layout(object):
    def __init__(self,
                 in_fname,
                 runs,
                 run_names,
                 use_interleaved,
                 properties,
                 preproc_fname=None,
                 max_part=False,
                 asic=None):

        self._in_fname = in_fname
        self._runs = runs
        self._run_names = run_names
        self._use_interleaved = use_interleaved
        self._preprocessing_fname = preproc_fname
        self._max_part = max_part
        self._asic = asic

        self._path = {
            'data': "entry/instrument/detector/data",
            'data_digital': "entry/instrument/detector/data_digital",
            'collection': "entry/instrument/detector/collection",
            'seq_number': "entry/instrument/detector/sequence_number"
        }
        self._path['error_code'] = ("{}/error_code"
                                    .format(self._path['collection']))
        self._path['frame_number'] = ("{}/frame_numbers"
                                      .format(self._path['collection']))
        self._path['total_lost_frames'] = ("{}/total_loss_frames"
                                           .format(self._path['collection']))

        self._channel = None

        self._n_rows_total = properties["n_rows_total"]
        self._n_cols_total = properties["n_cols_total"]
        self._max_pulses = properties["max_pulses"]
        self._n_memcells = properties["n_memcells"]

        self._seq_number = None
        self._source_seq_number = [0]

        self._raw_shape = None

        self._target_index = None
        self._target_index_full_size = None
        self._source_index = None

    def initiate(self, n_rows, n_cols):
        """Initiates all layout dependent attributes.

        Args:
            n_rows: Number of rows in the module.
            n_cols: Number of columns in the module.

        Return:
            A tuple containing layout specific dimensions:

            - Input file name: Some layouts require the input file name to
                               be modified (e.g. insert module position)
            - Number of memory cells
            - Total frames contained in the data
            - Raw shape
            - Path where to find the data inside the file.

        Raises:
            RuntimeError: The dimensions of the data do not match the
                          requirements.
        """

        # take first run and asume that the others have as many frames
        # TODO check this
        run_number = self._runs[0]
        if self._run_names:
            run_name = self._run_names[0]
        else:
            run_name = None

        fname_with_wildcard = self._in_fname.format(run_name=run_name,
                                                    run_number=run_number,
                                                    part=0)
        # cfel file names have ...<module name>_<module position>...
        # -> the wildcard for the module position has to be filled
        print(fname_with_wildcard)
        found_files = glob.glob(fname_with_wildcard)
        try:
            fname = found_files[0]
        except:
            if len(found_files) == 0:
                print("No files found for fname {}".format(fname_with_wildcard))

            raise

        module, self._channel = self._get_module_and_channel(
            fname,
            fname_with_wildcard
        )
        print("module {}".format(module))
        print("channel {}".format(self._channel))

        new_fname = self._in_fname.replace("*", "_" + self._channel)

        source_shape = None
        # get number of frames
        with h5py.File(fname, "r", libver="latest", driver="core") as f:
            try:
                # TODO: verify that the shape is always right and not
                #       dependent on frame loss
                source_shape = f[self._path['data']].shape
                exp_total_frames = f[self._path['frame_number']][0]
            except:
                print("Error when getting shape")
                raise

        if (source_shape[1] != self._n_rows_total and
                source_shape[2] != self._n_cols_total):
            msg = "Shape of file {} ".format(fname)
            msg += "does not match requirements\n"
            msg += "source_shape = {}".format(source_shape)
            raise RuntimeError(msg)

        n_frames_total = int(exp_total_frames)
        print("n_frames_total {}".format(n_frames_total))

        self._raw_shape = (self._n_memcells, 2, n_rows, n_cols)

        results = {
            'in_fname': new_fname,
            'module': module,
            'channel': self._channel,
            'n_memcells': self._n_memcells,
            'n_frames_total': n_frames_total,
            'raw_shape': self._raw_shape,
            'data_path': self._path['data'],
            'asic_order': utils.get_asic_order()
        }

        return results

    def _get_module_and_channel(self, fname, fname_with_wildcard):
        """Determines module and module position.

        Args:
            fname: File name without any wildcard but containing the actual
                   module position
            fname_with_wildcard: File name with wildcard where the module
                                 position should be filled in

        Return:
            The module workin on and on which channel it is plugged in the
            detector: [module, channel].


        """

        # determine the cut off parts to receive the module position
        pre_and_postfix = fname_with_wildcard.split("*")
        # cut off the part before the wildcard + underscore
        channel = fname[len(pre_and_postfix[0]) + 1:]
        # cut off the part after the wildcard
        channel = channel[:-len(pre_and_postfix[1])]

        # determine module out of the first part of the file name
        module = os.path.basename(pre_and_postfix[0])

        return module, channel

    def load(self,
             fname,
             run_idx,
             seq,
             load_idx_rows,
             load_idx_cols,
             file_content,
             tmp_data,
             pos_idxs):
        """Load the data.

        Args:
        fname: The name of the file containing the data to be loaded.
        run_idx: The run currently looked at (not the actual run number but
                 the index in the overall run list). This is needed to get
                 the corresponding preprocessing information.
        seq: The sequence number to be loaded.
        load_idx_rows: The data of which rows should be loaded only.
        load_idx_cols: The data of which columns should be loaded only.
        file_content: All metadata in corresponding to the data.
        tmp_data: Array where the data is stored into.
        pos_idxs: Which data parts should be loaded (shich columns and rows),
                  load and store positions are the same.
        """

        self.pos_idxs = pos_idxs

        # fill in the wildcard for the module position
        fname = glob.glob(fname)[0]

        # load data
        with h5py.File(fname, "r") as f:
            idx = (Ellipsis, load_idx_rows, load_idx_cols)
            raw_data = f[self._path['data']][idx]

        utils.check_data_type(raw_data)

        # cfel stores the data as int16 whereas xfel stores it at uint16
        # -> have one common type uint16 because ADC values also should be
        # positive
        utils.convert_dtype(raw_data, np.uint16)

        print("raw_data.shape", raw_data.shape)
        print("self._raw_shape", self._raw_shape)
        self.get_seq_number(file_content[self._path['seq_number']])
        self.get_frame_loss_indices()
        self.fillup_frame_loss(tmp_data, raw_data)

    def get_seq_number(self, source_seq_number):
        """Clean up sequence number (remove independence on previous parts)

        Args:
            source_seq_number: Sequence number read in from the current part.
        """

        # if there is frame loss this is recognizable by a missing
        # entry in seq_number. seq_number should be loaded before
        # doing operations on it due to performance benefits
        seq_number_last_entry_previous_file = self._source_seq_number[-1]

        print("seq_number_last_entry_previous_file={}"
              .format(seq_number_last_entry_previous_file))
        print("seq_number before modifying: {}"
              .format(source_seq_number))
        self._seq_number = (source_seq_number
                            # seq_number starts counting with 1
                            - 1
                            # the seq_number refers to the whole run
                            # not one file
                            - seq_number_last_entry_previous_file)
        print("seq_number: {}".format(self._seq_number))

        self._source_seq_number = source_seq_number

    def get_frame_loss_indices(self):
        """
        Calculates blocks of data without frame loss.

        Calculates indices in respect to the part (source index, target index)
        and the the whole run (target_index_full_size)
        """

        # The borders (regarding the expected shape) of
        # continuous blocks of data written into the target
        # (in between these blocks there will be zeros)
        self._target_index = [[0, 0]]
        # original sequence number starts with 1
        self._target_index_full_size = [[self._source_seq_number[0] - 1, 0]]
        # The borders (regarding the source_shape) of
        # continuous blocks of data read from the source
        # (no elements in between these blocks)
        self._source_index = [[0, 0]]
        stop = 0
        stop_full_size = 0
        stop_source = 0
        for i in np.arange(len(self._seq_number)):

            # a gap in the numbering occured
            if stop - self._seq_number[i] < -1:

                # the number before the gab gives the end of
                # the continuous block of data
                self._target_index[-1][1] = stop
                # the next block starts now
                self._target_index.append([self._seq_number[i], 0])

                # the number before the gab gives the end of
                # the continuous block of data in the fully sized array
                self._target_index_full_size[-1][1] = stop_full_size
                # the next block starts now
                # original sequence number started with 1
                seqlst = [self._source_seq_number[i] - 1, 0]
                self._target_index_full_size.append(seqlst)

                self._source_index[-1][1] = stop_source
                # the end of the block in the source
                self._source_index.append([i, 0])

            stop_source = i
            stop = self._seq_number[i]
            # original sequence number started with 1
            stop_full_size = self._source_seq_number[i] - 1

        # the last block ends with the end of the data
        self._target_index[-1][1] = self._seq_number[-1]
        self._target_index_full_size[-1][1] = self._source_seq_number[-1] - 1
        self._source_index[-1][1] = len(self._seq_number) - 1
        print("_target_index {}".format(self._target_index))
        print("_target_index_full_size {}".format(self._target_index_full_size))
        print("_source_index {}".format(self._source_index))

        # check to see the values of the sequence number in the first frame
        # loss gap
        if len(self._target_index_full_size) > 1:
            start = self._source_index[0][1] - 1
            stop = self._source_index[1][0] + 2
            seq_num = self._source_seq_number[start:stop]
            print("seq number in first frame loss region: {}".format(seq_num))

    def fillup_frame_loss(self, raw_data, loaded_raw_data):
        """Fills the loaded data into an array while considering frame loss.

        Args:
        raw_data: An array allocated with the full size of the run
                  (total frames) where the loaded data in filled in.
        loaded_data: Data loaded from this part.
        """

        target_index = self._target_index_full_size

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
            s_start = self._source_index[i][0]
            s_stop = self._source_index[i][1] + 1

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
