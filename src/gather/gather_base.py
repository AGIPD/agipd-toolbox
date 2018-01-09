import h5py
import numpy as np
import os
import sys
import time
import glob
import configparser

from layouts.xfel_layout import XfelLayout
from layouts.cfel_layout import CfelLayout

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


class AgipdGatherBase():
    def __init__(self,
                 in_fname,
                 out_fname,
                 runs,
                 preproc_fname=None,
                 max_part=False,
                 asic=None,
                 use_xfel_format=False,
                 backing_store=True):

        self.in_fname = in_fname
        self.out_fname = out_fname
        self.preprocessing_fname = preproc_fname

        self.runs = [int(r) for r in runs]

        self.max_part = max_part
        self.asic = asic
        self.backing_store = backing_store

        # to use the interleaved or not interleaved format
        # self.use_interleaved = True
        self.use_interleaved = False

        if use_xfel_format:
            layout = XfelLayout
        else:
            layout = CfelLayout
        self.layout = layout(self.in_fname,
                             self.runs,
                             self.use_interleaved,
                             self.preprocessing_fname,
                             self.max_part,
                             self.asic)

        self.analog = None
        self.digital = None

        self.raw_shape = None
        self.tmp_shape = None
        self.target_shape = None

        self.target_index = None
        self.target_index_full_size = None
        self.source_index = None
        self.source_seq_number = None
        self.seq_number = None
        self.max_pulses = None

        self.n_rows_total = 128
        self.n_cols_total = 512

        self.asic_size = 64

        self.a_row_start = None
        self.a_row_stop = None
        self.a_col_start = None
        self.a_col_stop = None

        self.get_parts()

        if self.n_parts == 0:
            msg = "No parts to gather found\n"
            msg += "in_fname={}".format(self.in_fname)
            raise Exception(msg)

        if self.asic is None:
            self.n_rows = self.n_rows_total
            self.n_cols = self.n_cols_total
        else:
            print("asic {}".format(self.asic))
            self.n_rows = self.asic_size
            self.n_cols = self.asic_size

            asic_order = utils.get_asic_order()
            mapped_asic = utils.calculate_mapped_asic(asic_order)
            print("mapped_asic={}".format(mapped_asic))

            (self.a_row_start,
             self.a_row_stop,
             self.a_col_start,
             self.a_col_stop) = utils.determine_asic_border(mapped_asic,
                                                            self.asic_size)

        self.intiate()

        print("\n\n\n"
              "start gather\n"
              "in_fname = {}\n"
              "out_fname ={}\n"
              "data_path = {}\n"
              .format(self.in_fname,
                      self.out_fname,
                      self.data_path))

    def get_parts(self):
        # remove extension
        prefix = self.in_fname.rsplit(".", 1)[0]
        # removet the part section
        prefix = prefix[:-10]
        # use the first run number to determine number of parts
        run_number = self.runs[0]
        prefix = prefix.format(run_number=run_number)
        print("prefix={}".format(prefix))

        part_files = glob.glob("{}*".format(prefix))

        self.n_parts = self.max_part or len(part_files)
        print("n_parts {}".format(self.n_parts))

    def intiate(self):
        (self.n_memcells,
         self.n_frames_total,
         self.raw_shape,
         self.data_path) = self.layout.initiate(self.n_rows, self.n_cols)

        self.define_needed_data_paths()

        # tmp data is already converted into agipd format
        if self.use_interleaved:
            self.raw_tmp_shape = (self.n_frames_total,
                                  self.n_rows, self.n_cols)
        else:
            self.raw_tmp_shape = (self.n_frames_total, 2,
                                  self.n_rows, self.n_cols)

        self.tmp_shape = (-1, self.n_memcells, 2, self.n_rows, self.n_cols)

        self.target_shape = (-1, self.n_memcells, self.n_rows, self.n_cols)
        print("target shape:", self.target_shape)

    def get_preproc_res(self):
        if self.preprocessing_fname is None:
            return {}
        else:
            config = configparser.RawConfigParser()
            config.read(self.preprocessing_fname)

            return cfel_optarg.parse_parameters(config)

    # to give classes which inherite from this class the possibility to define
    # file internal paths they need
    def define_needed_data_paths(self):
        pass

    def run(self):

        totalTime = time.time()

        self.load_data()

        print("Start saving")
        print("out_fname = {}".format(self.out_fname))
        f = None
        try:
            f = h5py.File(self.out_fname, "w", libver='latest')
            f.create_dataset("analog", data=self.analog, dtype=np.int16)
            f.create_dataset("digital", data=self.digital, dtype=np.int16)

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
        finally:
            if f is not None:
                f.close()

        print("gather took time:", time.time() - totalTime, "\n\n")

    def load_data(self):

        print("raw_tmp_shape", self.raw_tmp_shape)
        tmp_data = np.zeros(self.raw_tmp_shape, dtype=np.int16)

        self.metadata = {}
        self.seq_number = None

        for run_idx, run_number in enumerate(self.runs):
            print("\n\nrun {}".format(run_number))

            self.pos_idxs = self.set_pos_indices(run_idx)
            print("pos_idxs", self.pos_idxs)

            load_idx_rows = slice(self.a_row_start, self.a_row_stop)
            load_idx_cols = slice(self.a_col_start, self.a_col_stop)
            print("load idx: {}, {}".format(load_idx_rows, load_idx_cols))

            self.source_seq_number = [0]
            for i in range(self.n_parts):
                fname = self.in_fname.format(run_number=run_number, part=i)
                print("loading file {}".format(fname))

                excluded = [self.data_path]
                file_content = utils.load_file_content(fname, excluded)

                self.layout.load(fname,
                                 i,
                                 load_idx_rows,
                                 load_idx_cols,
                                 file_content,
                                 tmp_data,
                                 self.pos_idxs)

                self.metadata[fname] = file_content

        print("self.tmp_shape", self.tmp_shape)
        print("tmp_data.shape", tmp_data.shape)

        tmp_data.shape = self.tmp_shape
        print("tmp_data.shape", tmp_data.shape)

        self.analog = tmp_data[:, :, 0, ...]
        self.digital = tmp_data[:, :, 1, ...]

    def set_pos_indices(self, run_idx):
        pos_idx_rows = slice(None)
        pos_idx_cols = slice(None)

        # retuns a list of row/col indixes to give the possibility to
        # define subgroups
        # e.g. top half should use these cols and bottom half those ones
        return [[pos_idx_rows, pos_idx_cols]]

if __name__ == "__main__":

    use_xfel_format = True
    # use_xfel_format = False

    if use_xfel_format:
        # base_path = "/gpfs/exfel/exp/SPB/201730/p900009"
        # run_list = [["0428"], ["0429"], ["0430"]]

        base_path = "/gpfs/exfel/exp/SPB/201730/p900009"
        run_list = [["0709"]]
        # run_list = [["0488"]]

        # base_path = "/gpfs/exfel/exp/SPB/201701/p002012"
        # run_list = [["0007"]]

        subdir = "scratch/user/kuhnm/tmp"

        channel = 1
        in_file_name = ("RAW-R{run_number:04}-" +
                        "AGIPD{:02}".format(channel) +
                        "-S{part:05d}.h5")
        in_fname = os.path.join(base_path,
                                "raw",
                                "r{run_number:04}",
                                in_file_name)

        for runs in run_list:
            run_subdir = "r" + "-r".join(runs)
            out_dir = os.path.join(base_path,
                                   subdir,
                                   run_subdir,
                                   "gather")
            utils.create_dir(out_dir)

            preproc_fname = os.path.join(base_path,
                                         subdir,
                                         run_subdir,
                                         "{}-preprocessing.result"
                                         .format(run_subdir.upper()))

            out_file_name = ("{}-AGIPD{:02}-gathered.h5"
                             .format(run_subdir.upper(), channel))
            out_fname = os.path.join(out_dir,
                                     out_file_name)

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
        module = "M304_m2"
        runs = ["00012"]
#        asic = None  # asic (None means full module)
        asic = 1

        max_part = False
        out_base_path = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm"
        out_subdir = "tmp"
        meas_type = "dark"
        meas_spec = {
            "dark": "tint150ns",
        }

        in_file_name = ("{}_{}_{}_"
                        .format(module,
                                meas_type,
                                meas_spec[meas_type]) +
                        "{run_number}_part{part:05d}.nxs")
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
            out_file_name = ("{}_{}_{}_asic{:02d}.h5"
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
