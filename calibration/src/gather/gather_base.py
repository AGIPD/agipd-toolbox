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
import time
import glob

from __init__ import FACILITY_DIR

import utils
from _version import __version__


class GatherBase(object):
    def __init__(self,
                 in_fname,
                 out_fname,
                 runs,
                 run_names,
                 properties,
                 use_interleaved,
                 detector_string,
                 preproc_fname=None,
                 max_part=False,
                 asic=None,
                 layout="xfel_layout",
                 facility="xfel",
                 backing_store=True):
        global FACILITY_DIR

        self._in_fname = in_fname
        self._out_fname = out_fname
        self._properties = properties
        self._use_interleaved = use_interleaved
        self._detector_string = detector_string

        self.runs = [int(r) for r in runs]
        self.run_names = run_names

        self._max_part = max_part
        self._asic = asic
        self._backing_store = backing_store

        self._facility = facility

        # load facility
        LAYOUT_DIR = os.path.join(FACILITY_DIR, self._facility, "layout")
        if LAYOUT_DIR not in sys.path:
            sys.path.insert(0, LAYOUT_DIR)

        # load layout
        Layout = __import__(layout).Layout
        if layout.startswith("cfel") and not self._use_interleaved:
                print("ERROR: CFEL only supports interleaved mode.")

        self._n_rows_total = self._properties["n_rows_total"]
        self._n_cols_total = self._properties["n_cols_total"]

        self._analog = None
        self._digital = None

        self.raw_shape = None
        self._raw_tmp_shape = None
        self._tmp_shape = None
        self._target_shape = None

        self._data_path = None

        self._module = None
        self._channel = None

        # public to be used in inherited classes
        self.n_rows = None
        self.n_cols = None

        self.asic_size = 64

        self.a_row_start = None
        self.a_row_stop = None
        self.a_col_start = None
        self.a_col_stop = None

        self._set_n_parts()

        if self._n_parts == 0:
            print("No parts to gather found, check if data exists without parts")
            prefix = self._in_fname.rsplit("_", 1)[0]
            
            # use the first run number to determine number of parts
            run_number = self.runs[0]

            try:
                run_name = self.run_names[0]
            # TypeError happens when self.run_names is None (e.g. in XFEL dark)
            except (IndexError, TypeError):
                run_name = None
                
            prefix = prefix.format(run_name=run_name, run_number = run_number)
            files = glob.glob("{}*".format(prefix))
            
            print(len(files), " file(s) found.")
            self._n_parts = 1
            self._in_fname = prefix + ".nxs"
            print("prefix={}".format(prefix))
        
        if self._n_parts == 0:
            msg = "Still no files to gather found"
            msg += "in_fname={}".format(self._in_fname)
            raise Exception(msg)

        # (frames, mem_cells, rows, columns)
        # is transposed to
        # (rows, columns, mem_cells, frames)
        self.transpose_order = (2, 3, 1, 0)

        if self._asic is None:
            self.n_rows = self._n_rows_total
            self.n_cols = self._n_cols_total
        else:
            print("asic {}".format(self._asic))
            self.n_rows = self.asic_size
            self.n_cols = self.asic_size

        self.layout = Layout(
            in_fname=self._in_fname,
            runs=self.runs,
            run_names=self.run_names,
            use_interleaved=self._use_interleaved,
            properties=self._properties,
            preproc_fname=preproc_fname,
            max_part=self._max_part,
            asic=self._asic,
            detector_string=self._detector_string
        )

        self._initiate()

        print("\n\n\n"
              "start gather\n"
              "in_fname = {}\n"
              "out_fname ={}\n"
              "data_path = {}\n"
              .format(self._in_fname,
                      self._out_fname,
                      self._data_path))

    def _set_n_parts(self):
        # remove extension
        prefix = self._in_fname.rsplit(".", 1)[0]

        part_string = "{part:05}"
        if prefix.endswith(part_string):
            # remove the part section
            prefix = prefix[:-len(part_string)]

            # use the first run number to determine number of parts
            run_number = self.runs[0]

            try:
                run_name = self.run_names[0]
            # TypeError happens when self.run_names is None (e.g. in XFEL dark)
            except (IndexError, TypeError):
                run_name = None

            prefix = prefix.format(run_name=run_name, run_number=run_number)
            print("prefix={}".format(prefix))

            part_files = glob.glob("{}*".format(prefix))
            print(part_files)

            self._n_parts = self._max_part or len(part_files)
            print("n_parts {}".format(self._n_parts))
        else:
            # data is not split in parts
            self._n_parts = 1

    def _initiate(self):
        print("n_rows", self.n_rows)
        print("n_cols", self.n_cols)
        init_results = self.layout.initiate(n_rows=self.n_rows,
                                            n_cols=self.n_cols)

        self._in_fname = init_results['in_fname']
        self._module = init_results['module']
        self._channel = init_results['channel']
        n_memcells = init_results['n_memcells']
        n_frames_total = init_results['n_frames_total']
        self.raw_shape = init_results['raw_shape']
        self._data_path = init_results['data_path']

        self.define_needed_data_paths()

        if self._asic is not None:
            asic_order = init_results['asic_order']
            mapped_asic = utils.calculate_mapped_asic(asic=self._asic,
                                                      asic_order=asic_order)
            print("mapped_asic={}".format(mapped_asic))

            (self.a_row_start,
             self.a_row_stop,
             self.a_col_start,
             self.a_col_stop) = utils.determine_asic_border(
                mapped_asic=mapped_asic,
                asic_size=self.asic_size,
                asic_order=asic_order
            )

        # tmp data is already converted into agipd format
        if self._use_interleaved:
            self._raw_tmp_shape = (n_frames_total,
                                   self.n_rows, self.n_cols)
        else:
            self._raw_tmp_shape = (n_frames_total, 2,
                                   self.n_rows, self.n_cols)

        self._tmp_shape = (-1, n_memcells, 2, self.n_rows, self.n_cols)

#        self._target_shape = (-1, n_memcells, self.n_rows, self.n_cols)
#        print("target shape:", self._target_shape)

    # to give classes which inherite from this class the possibility to define
    # file internal paths they need
    def define_needed_data_paths(self):
        pass

    def run(self):
        totalTime = time.time()

        self._load_data()

        print("Start saving")
        self._write_data()
        print("Saving done")

        print("gather took time:", time.time() - totalTime, "\n\n")

    def _load_data(self):

        print("raw_tmp_shape", self._raw_tmp_shape)
        tmp_data = np.zeros(self._raw_tmp_shape, dtype=np.uint16)

        self.metadata = {}

        for run_idx, run_number in enumerate(self.runs):
            if self.run_names and self.run_names != [None]:
                run_name = self.run_names[run_idx]
                print("\n\nrun {} ({})".format(run_number, run_name))
            else:
                run_name = None
                print("\n\nrun {}".format(run_number))


            self.pos_idxs = self.set_pos_indices(run_idx, self._asic)
            print("pos_idxs", self.pos_idxs)

            load_idx_rows = slice(self.a_row_start, self.a_row_stop)
            load_idx_cols = slice(self.a_col_start, self.a_col_stop)
            print("load idx: {}, {}".format(load_idx_rows, load_idx_cols))

            for i in range(self._n_parts):
                fname = self._in_fname.format(run_name=run_name,
                                              run_number=run_number,
                                              part=i)
                print("loading file {}".format(fname))

                excluded = [self._data_path]
                file_content = utils.load_file_content(fname, excluded)

                self.layout.load(fname=fname,
                                 run_idx=run_idx,
                                 seq=i,
                                 load_idx_rows=load_idx_rows,
                                 load_idx_cols=load_idx_cols,
                                 file_content=file_content,
                                 tmp_data=tmp_data,
                                 pos_idxs=self.pos_idxs)

                self.metadata[fname] = file_content

        print("self._tmp_shape", self._tmp_shape)
        print("tmp_data.shape", tmp_data.shape)

        tmp_data.shape = self._tmp_shape
        print("tmp_data.shape", tmp_data.shape)

        analog = tmp_data[:, :, 0, ...]
        digital = tmp_data[:, :, 1, ...]

        # reorder data
        print("\nStart transposing")
        t = time.time()
        self._analog = analog.transpose(self.transpose_order)
        self._digital = digital.transpose(self.transpose_order)
        print("took time: {}".format(time.time() - t))

    def _write_data(self):

        collection = {
            "module": str(self._module),
            "channel": str(self._channel),
            "version": str(__version__)
        }

        print("out_fname = {}".format(self._out_fname))
        with h5py.File(self._out_fname, "w", libver='latest') as f:
            f.create_dataset("analog", data=self._analog, dtype=np.uint16)
            f.create_dataset("digital", data=self._digital, dtype=np.uint16)

            prefix = "collection"
            for key, value in collection.items():
                name = "{}/{}".format(prefix, key)
                f.create_dataset(name, data=value)

            # sort metadata entries before writing them into a file
            keys = [key for key in self.metadata]
            sorted_keys = sorted(keys)

            # save metadata from original files
            idx = 0
            # for set_name, set_value in iter(self.metadata.items()):
            for set_name in sorted_keys:
                set_value = self.metadata[set_name]

                gname = "metadata_{}".format(idx)

                name = "{}/source".format(gname)
                f.create_dataset(name, data=set_name)

                for key, value in iter(set_value.items()):
                    try:
                        name = "{}/{}".format(gname, key)
                        f.create_dataset(name, data=value)
                    except:
                        print("Error in", name, value.dtype)
                        raise
                idx += 1

            f.flush()

    def set_pos_indices(self, run_idx, asic):
        pos_idx_rows = slice(None)
        pos_idx_cols = slice(None)

        # retuns a list of row/col indixes to give the possibility to
        # define subgroups
        # e.g. top half should use these cols and bottom half those ones
        return [[pos_idx_rows, pos_idx_cols]]

    def asic_in_upper_half(self):
        return utils.located_in_upper_half(self._asic)

if __name__ == "__main__":

    # use_xfel_format = True
    use_xfel_format = False

    if use_xfel_format:
        # base_path = "/gpfs/exfel/exp/SPB/201730/p900009"
        # run_list = [["0428"], ["0429"], ["0430"]]

        base_path = "/gpfs/exfel/exp/SPB/201730/p900009"
        run_list = [["0819"]]
        run_type = "dark"
        # run_list = [["0488"]]

        # base_path = "/gpfs/exfel/exp/SPB/201701/p002012"
        # run_list = [["0007"]]

        subdir = "scratch/user/kuhnm/tmp"

        channel = 1
        in_file_name = ("RAW-R{run_number:04}-" +
                        "AGIPD{:02}".format(channel) +
                        "-S{part:05}.h5")
        in_fname = os.path.join(base_path,
                                "raw",
                                "r{run_number:04}",
                                in_file_name)
        print("in_fname", in_fname)

        for runs in run_list:
            run_subdir = "r" + "-r".join(runs)
            out_dir = os.path.join(base_path,
                                   subdir,
                                   run_type,
                                   run_subdir,
                                   "gather")
            utils.create_dir(out_dir)

            preproc_fname = os.path.join(base_path,
                                         subdir,
                                         run_type,
                                         "r{run:04}",
                                         "R{run:04}-preprocessing.result")
            print("preproc_fname", preproc_fname)

            out_file_name = ("{}-AGIPD{:02}-gathered.h5"
                             .format(run_subdir.upper(), channel))
            out_fname = os.path.join(out_dir,
                                     out_file_name)
            print("out_fname", out_fname)

            obj = GatherBase(in_fname=in_fname,
                             out_fname=out_fname,
                             runs=runs,
                             preproc_fname=preproc_fname,
                             max_part=False,
                             asic=None,
                             use_xfel_format=use_xfel_format)
            obj.run()

    else:

        in_base_path = "/gpfs/cfel/fsds/labs/agipd/calibration"
        # with frame loss
#        in_subdir = "raw/317-308-215-318-313/temperature_m15C/dark"
#        module = "M317_m2"
#        runs = ["00001"]

        # no frame loss
        in_subdir = "raw/315-304-309-314-316-306-307/temperature_m25C/dark"
        module = "M304"
        runs = [12]

        asic = None  # asic (None means full module)
#        asic = 1

        max_part = False
        out_base_path = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm"
        out_subdir = "tmp/cfel"
        measurement = "dark"
        meas_spec = {
            "dark": "tint150ns",
        }

        in_file_name = ("{}*_{}_{}_"
                        .format(module,
                                measurement,
                                meas_spec[measurement]) +
                        "{run_number:05}_part{part:05}.nxs")
        in_fname = os.path.join(in_base_path,
                                in_subdir,
                                in_file_name)

        out_dir = os.path.join(out_base_path,
                               out_subdir,
                               "gather")
        utils.create_dir(out_dir)

        if asic is None:
            out_file_name = ("{}_{}_{}.h5"
                             .format(module.split("_")[0],
                                     measurement,
                                     meas_spec[measurement]))
        else:
            out_file_name = ("{}_{}_{}_asic{:02}.h5"
                             .format(module.split("_")[0],
                                     measurement,
                                     meas_spec[measurement],
                                     asic))
        out_fname = os.path.join(out_dir, out_file_name)

        print("Used parameters:")
        print("in_fname=", in_fname)
        print("out_fname=", out_fname)
        print("runs=", runs)
        print("max_part=", max_part)
        print("asic=", asic)
        print("use_xfel_format=", use_xfel_format)
        print()

        obj = GatherBase(in_fname=in_fname,
                         out_fname=out_fname,
                         runs=runs,
                         max_part=max_part,
                         asic=asic,
                         use_xfel_format=use_xfel_format)
        obj.run()
