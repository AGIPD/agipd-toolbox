import h5py
import numpy as np
import os
import sys
import time
import glob


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


class AgipdGatherBase(object):
    def __init__(self,
                 in_fname,
                 out_fname,
                 runs,
                 preproc_fname=None,
                 max_part=False,
                 asic=None,
                 use_xfel_format=False,
                 backing_store=True):

        self._in_fname = in_fname
        self._out_fname = out_fname

        self.runs = [int(r) for r in runs]

        self._max_part = max_part
        self._asic = asic
        self._backing_store = backing_store

        if use_xfel_format:
            from layouts.xfel_layout import XfelLayout as layout

            # to use the interleaved or not interleaved format
            # self._use_interleaved = True
            self._use_interleaved = False

        else:
            from layouts.cfel_layout import CfelLayout as layout
            self._use_interleaved = True

        self.layout = layout(
            in_fname=self._in_fname,
            runs=self.runs,
            use_interleaved=self._use_interleaved,
            preproc_fname=preproc_fname,
            max_part=self._max_part,
            asic=self._asic
        )

        self._analog = None
        self._digital = None

        self.raw_shape = None
        self._raw_tmp_shape = None
        self._tmp_shape = None
        self._target_shape = None

        self._data_path = None

        self._n_rows_total = 128
        self._n_cols_total = 512

        self._asic_size = 64

        self._a_row_start = None
        self._a_row_stop = None
        self._a_col_start = None
        self._a_col_stop = None

        # public to be used in inherited classes
        self.n_rows = None
        self.n_cols = None

        self.get_parts()

        if self._n_parts == 0:
            msg = "No parts to gather found\n"
            msg += "in_fname={}".format(self._in_fname)
            raise Exception(msg)

        if self._asic is None:
            self.n_rows = self._n_rows_total
            self.n_cols = self._n_cols_total
        else:
            print("asic {}".format(self._asic))
            self.n_rows = self._asic_size
            self.n_cols = self._asic_size

            asic_order = utils.get_asic_order()
            mapped_asic = utils.calculate_mapped_asic(asic_order)
            print("mapped_asic={}".format(mapped_asic))

            (self._a_row_start,
             self._a_row_stop,
             self._a_col_start,
             self._a_col_stop) = utils.determine_asic_border(mapped_asic,
                                                             self._asic_size)

        self.intiate()

        print("\n\n\n"
              "start gather\n"
              "in_fname = {}\n"
              "out_fname ={}\n"
              "data_path = {}\n"
              .format(self._in_fname,
                      self._out_fname,
                      self._data_path))

    def get_parts(self):
        # remove extension
        prefix = self._in_fname.rsplit(".", 1)[0]
        # remove the part section
        prefix = prefix[:-9]
        # use the first run number to determine number of parts
        run_number = self.runs[0]
        prefix = prefix.format(run_number=run_number)
        print("prefix={}".format(prefix))

        part_files = glob.glob("{}*".format(prefix))

        self._n_parts = self._max_part or len(part_files)
        print("n_parts {}".format(self._n_parts))

    def intiate(self):
        (self._in_fname,
         n_memcells,
         n_frames_total,
         self.raw_shape,
         self._data_path) = self.layout.initiate(n_rows=self.n_rows,
                                                 n_cols=self.n_cols)

        self.define_needed_data_paths()

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

        self.load_data()

        print("Start saving")
        print("out_fname = {}".format(self._out_fname))
        with h5py.File(self._out_fname, "w", libver='latest') as f:
            f.create_dataset("analog", data=self._analog, dtype=np.int16)
            f.create_dataset("digital", data=self._digital, dtype=np.int16)

            # save metadata from original files
            idx = 0
            for set_name, set_value in iter(self.metadata.items()):
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
            print("Saving done")

            f.flush()

        print("gather took time:", time.time() - totalTime, "\n\n")

    def load_data(self):

        print("raw_tmp_shape", self._raw_tmp_shape)
        tmp_data = np.zeros(self._raw_tmp_shape, dtype=np.int16)

        self.metadata = {}

        for run_idx, run_number in enumerate(self.runs):
            print("\n\nrun {}".format(run_number))

            self.pos_idxs = self.set_pos_indices(run_idx)
            print("pos_idxs", self.pos_idxs)

            load_idx_rows = slice(self._a_row_start, self._a_row_stop)
            load_idx_cols = slice(self._a_col_start, self._a_col_stop)
            print("load idx: {}, {}".format(load_idx_rows, load_idx_cols))

            for i in range(self._n_parts):
                fname = self._in_fname.format(run_number=run_number, part=i)
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

        self._analog = tmp_data[:, :, 0, ...]
        self._digital = tmp_data[:, :, 1, ...]

    def set_pos_indices(self, run_idx):
        pos_idx_rows = slice(None)
        pos_idx_cols = slice(None)

        # retuns a list of row/col indixes to give the possibility to
        # define subgroups
        # e.g. top half should use these cols and bottom half those ones
        return [[pos_idx_rows, pos_idx_cols]]


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

            obj = AgipdGatherBase(in_fname=in_fname,
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
        meas_type = "dark"
        meas_spec = {
            "dark": "tint150ns",
        }

        in_file_name = ("{}*_{}_{}_"
                        .format(module,
                                meas_type,
                                meas_spec[meas_type]) +
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
                                     meas_type,
                                     meas_spec[meas_type]))
        else:
            out_file_name = ("{}_{}_{}_asic{:02}.h5"
                             .format(module.split("_")[0],
                                     meas_type,
                                     meas_spec[meas_type],
                                     asic))
        out_fname = os.path.join(out_dir, out_file_name)

        obj = AgipdGatherBase(in_fname=in_fname,
                              out_fname=out_fname,
                              runs=runs,
                              max_part=max_part,
                              asic=asic,
                              use_xfel_format=use_xfel_format)
        obj.run()
