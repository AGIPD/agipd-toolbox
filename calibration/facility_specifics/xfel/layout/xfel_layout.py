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
import configparser

from calibration.facility_specifics.xfel.layout.__init__ import SRC_DIR
import calibration.facility_specifics.xfel.layout.cfel_optarg

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import utils  # noqa E402


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

        self._path_temp = {
            'image_first': ("INDEX/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/"
                            "image/first"),
            'train_count': ("INDEX/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/count"),

            'data': "INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/image/data",
            'cellid': ("INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf/"
                       "image/cellId"),

        }

        self._path = {}

        self._channel = None

        self._train_pos_per_run = []
        self._trainid_outlier = []

        self._n_memcells = None
        self._raw_shape = None

    def get_preproc_res(self, run):
        """Get and return the preprocessing results of a run.

        Args:
            run: Run number to get preprocessing results for.
        """

        if self._preprocessing_fname is None:
            return {}
        else:
            fname = self._preprocessing_fname.format(run=run)

            if not os.path.exists(fname):
                print("fname={}".format(fname))
                print("ERROR: preprocessing file does not exist")
                sys.exit(1)

            config = configparser.RawConfigParser()
            config.read(fname)

            return cfel_optarg.parse_parameters(config=config)

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
        """
        # TODO raise RuntimeError if dimensions of the data do not match the
        # requirements

        print("in_fname", self._in_fname)
        module, self._channel = self._get_module_and_channel()
        print("channel", self._channel)

        n_memcells_per_run = []
        n_trains_per_run = []
        for i, run in enumerate(self._runs):
            preproc = self.get_preproc_res(run=run)

            n_memcells_per_run.append(preproc['general']['n_memcells'])
            n_trains_per_run.append(preproc['general']['n_trains_total'])

            ch = "channel{:02}".format(self._channel)
            self._train_pos_per_run.append(preproc[ch]['train_pos'])
            self._trainid_outlier.append(preproc[ch]['trainid_outliers'])

        self._n_memcells = max(n_memcells_per_run)

        if self._use_interleaved:
            self._n_memcells_to_iterate = self._n_memcells * 2

            # xfel format has swapped rows and cols
            self._raw_shape = (self._n_memcells, 2, 2,
                               n_cols, n_rows)

            n_frames_total = max(n_trains_per_run) * self._n_memcells * 2
        else:
            self._n_memcells_to_iterate = self._n_memcells

            self._raw_shape = (self._n_memcells, 2,
                               n_cols, n_rows)
            n_frames_total = max(n_trains_per_run) * self._n_memcells

        print("Number of memory cells found", self._n_memcells)
        print("n_frames_total", n_frames_total)
        print("Runs with no data for this channel",
              [run for i, run in enumerate(self._runs)
               if self._train_pos_per_run[i] == None])

        for key in self._path_temp:
            self._path[key] = self._path_temp[key].format(self._channel)

        results = {
            'in_fname': self._in_fname,  # no modification
            'module': module,
            'channel': self._channel,
            'n_memcells': self._n_memcells,
            'n_frames_total': n_frames_total,
            'raw_shape': self._raw_shape,
            'data_path': self._path['data'],
            'asic_order': utils.get_asic_order_xfel(self._channel)
        }

        return results

    def _get_module_and_channel(self):
        """Determines module and module position.

        Return:
            The module workin on and on which channel it is plugged in:
            [module, channel].

        """

        # Get module
        module = None

        # Get channel
        split_tmp = self._in_fname.split("-")
        channel = int(split_tmp[-2].split("AGIPD")[1])

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

        train_pos = self._train_pos_per_run[run_idx]
        outliers = self._trainid_outlier[run_idx]

        with h5py.File(fname, "r") as f:
            raw_data = f[self._path['data']][..., load_idx_rows, load_idx_cols]
            #raw_data = f[self._path['data']][()]

#            status = utils.as_nparray(f[self._path['status']][()], np.int)
            first = utils.as_nparray(f[self._path['image_first']][()], np.int)
            count = utils.as_nparray(f[self._path['train_count']][()], np.int)
#            last = utils.as_nparray(f[self._path['image_last']][()], np.int)
            cellid = utils.as_nparray(f[self._path['cellid']][()], np.int)

        print("raw_data.shape", raw_data.shape)
        print("self._raw_shape", self._raw_shape)

        if np.all(count == 0):
            print("Sequence does not contain data for this channel")
            return

        utils.check_data_type(raw_data)

        if self._use_interleaved:
            # for the first experiments the splitting in digital and analog
            # did not work for XFEL
            # -> all data is in the first entry of the analog/digital
            #    dimension
            raw_data = raw_data[:, 0, ...]

        raw_data = utils.convert_to_agipd_format(module=self._channel,
                                                 data=raw_data,
                                                 # because checking works by
                                                 # comparing dim of rows and
                                                 # columns to 128 and 512 this
                                                 # does not work on asic level
                                                 check=False)

        count_not_zero = np.where(count != 0)[0]

        first_index = count_not_zero[0]
        if len(count_not_zero) == 1:
            # if the file only contains one train
            last_index = count_not_zero[0] + 1
        else:
            last_index = count_not_zero[-1] + 1

        for i, source_fidx in enumerate(first[first_index:last_index]):
            if i in outliers[seq]:
                print("outlier detected at position", i)
                continue

            source_lidx = source_fidx + count[first_index:last_index][i]
            if source_lidx == source_fidx:
#                print("count_not_zero", count_not_zero)
#                print("first_index", first_index)
#                print("last_index", last_index)
                print("WARNING: train loss detected at index {}".format(source_fidx))
                continue

            # Get train position (taken care of train loss)
            target_fidx = train_pos[seq][i] * self._n_memcells_to_iterate

            # Detect pulse loss
            diff = np.diff(cellid[source_fidx:source_lidx])
            cell_loss = np.where(diff != 1)[0]
            #if cell_loss.size != 0:
            #    print("cell_loss", cell_loss)

            # Fill up pulse loss
            source_interval = [source_fidx, source_fidx]
            for cidx in np.concatenate((cell_loss, [source_lidx-source_fidx-1])):
                source_interval[1] = source_fidx + cidx + 1

                t_start = target_fidx + cellid[source_interval[0]]
                t_stop = target_fidx + cellid[source_interval[1] - 1] + 1

                target_interval = [t_start, t_stop]

#                print(train_pos[seq][i], "interval", source_interval,
#                      "-", target_interval)

                for index_set in pos_idxs:
                    pos_idx_rows = index_set[0]
                    pos_idx_cols = index_set[1]

                    source_idx = (slice(*source_interval),
                                  Ellipsis,
                                  pos_idx_rows,
                                  pos_idx_cols)
                    target_idx = (slice(*target_interval),
                                  Ellipsis,
                                  pos_idx_rows,
                                  pos_idx_cols)

                    try:
                        tmp_data[target_idx] = raw_data[source_idx]
                    except:
                        print("tmp_data.shape", tmp_data.shape)
                        #print(tmp_data[target_idx].shape)
                        print("raw_data.shape", raw_data.shape)
                        #print(raw_data[source_idx].shape)
                        print("source_idx", source_idx)
                        print("target_idx", target_idx)
                        raise

                source_interval[0] = source_interval[1]
