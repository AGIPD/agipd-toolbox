#!/usr/bin/python3

import os
import sys
import datetime
import time
# import numpy as np
# from string import Template
import glob
import string

import utils
# from merge_drscs import ParallelMerge
from correct import Correct
from convert_format import ConvertFormat
from join_constants import JoinConstants

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print("BASE_PATH", BASE_PATH)
SRC_PATH = os.path.join(BASE_PATH, "src")
GATHER_PATH = os.path.join(SRC_PATH, "gather")
PROCESS_PATH = os.path.join(SRC_PATH, "process")

if GATHER_PATH not in sys.path:
    sys.path.insert(0, GATHER_PATH)

if PROCESS_PATH not in sys.path:
    sys.path.insert(0, PROCESS_PATH)


class Analyse():
    def __init__(self,
                 run_type,
                 meas_type,
                 in_base_dir,
                 out_base_dir,
                 n_processes,
                 module,
                 temperature,
                 meas_spec,
                 asic,
                 asic_list,
                 safety_factor,
                 runs,
                 max_part,
                 current_list=None,
                 use_xfel_in_format=True,
                 use_xfel_out_format=False):

        print("started Analyse")

        self.run_type = run_type
        self.meas_type = meas_type
        self.in_base_dir = in_base_dir
        self.out_base_dir = out_base_dir
        self.n_processes = n_processes
        self.module = module
        self.temperature = temperature
        self.meas_spec = meas_spec
        self.asic = asic
        self.asic_list = asic_list
        self.safety_factor = safety_factor
        self.runs = runs
        self.current_list = current_list
        self.channel = module

        self.max_part = max_part

        self.use_xfel_in_format = use_xfel_in_format
        self.use_xfel_out_format = use_xfel_out_format

        print("====== Configured parameter in class Analyse ======")
        print("type {}:".format(self.run_type))
        print("module/channel: ", self.module)
        print("temperature: ", self.temperature)
        print("type: ", self.meas_type)
        print("meas_spec: ", self.meas_spec)
        print("asic: ", self.asic)
        print("asic_list: ", self.asic_list)
        print("in_dir: ", self.in_base_dir)
        print("out_dir: ", self.out_base_dir)
        print("runs: ", self.runs)
        print("max_part: ", self.max_part)
        print("current_list: ", self.current_list)
        print("use_xfel_in_format: ", self.use_xfel_in_format)
        print("use_xfel_out_format: ", self.use_xfel_out_format)
        print("===================================================")

        # Usually the in directory and file names correspond to the
        # meas_type
        self.meas_in = {self.meas_type: self.meas_type}
        # but there are exceptions
        self.meas_in["drscs_dark"] = "drscs"

        self.run()

    def run(self):
        print("\nStarted at", str(datetime.datetime.now()))
        t = time.time()

        if self.run_type == "preprocess":
            self.run_preprocess()
        elif self.run_type == "gather":
            self.run_gather()
        elif self.run_type == "process":
            self.run_process()
        elif self.run_type == "merge":
            self.run_merge_drscs()
        elif self.run_type == "join":
            self.run_join()
        else:
            print("Unsupported argument: run_type {}".format(self.run_type))

        print("\nFinished at", str(datetime.datetime.now()))
        print("took time: ", time.time() - t)

    ########################################
    ##           generate paths           ##
    ########################################

    def generate_raw_path(self, base_dir, as_template=False):
        if self.use_xfel_in_format:
            if as_template:
                channel = "{:02}"
            else:
                channel = str(self.channel).zfill(2)

            # define in files
            fdir = os.path.join(base_dir,
                                "raw",
                                "r{run_number:04}")

            fname = ("RAW-R{run_number:04}-" +
                     "AGIPD{}".format(channel) +
                     "-S{part:05}.h5")

        else:
            # define in files
            fdir = os.path.join(base_dir,
                                self.temperature,
                                self.meas_in[self.meas_type])

            if self.meas_type not in ["dark", "xray"]:
                fdir = os.path.join(base_dir,
                                    self.meas_spec)

            # fname = ("{}*_{}_{}_" # only module without location, e.g. M304
            fname = ("{}_{}_{}_"
                     .format(self.module,
                             self.meas_type,
                             self.meas_spec)
                     + "{run_number:04}_part{part:05}.nxs")

        return fdir, fname

    def generate_preproc_path(self, base_dir, as_template=False):
        if self.use_xfel_in_format:
            if self.meas_type == "pcdrs" or len(self.runs) == 1:
                run_subdir = "r" + "-r".join(str(r).zfill(4)
                                             for r in self.runs)

            else:
                print("WARNING: generate_preproc_path is running in 'else'. Why?")
                print("self.runs={}".format(self.runs))
                run_subdir = "r{:04}".format(self.runs[0])

            if as_template:
                fdir = os.path.join(base_dir,
                                    self.meas_type,
                                    "r{run:04}")

                fname = "R{run:04}-preprocessing.result"
            else:
                fdir = os.path.join(base_dir,
                                    self.meas_type,
                                    run_subdir)

                fname = "R{:04}-preprocessing.result".format(self.runs[0])

        else:
            print("Preprocessing not implemented for CFEL layout")
            return None, None

        return fdir, fname

    def generate_gather_path(self, base_dir):
        if self.use_xfel_in_format:
            # TODO: concider additing this into out_base_dir (joined) and
            #       create subdirs for gathered files
            if self.meas_type == "pcdrs" or len(self.runs) == 1:
                run_subdir = "r" + "-r".join(str(r).zfill(4)
                                             for r in self.runs)

                fname = ("{}-AGIPD{:02}-gathered.h5"
                         .format(run_subdir.upper(), self.channel))

            # TODO fill run_number + for what is this 'else' (what cases)?
            else:
                run_subdir = "r{run_number:04}"

                fname = ("R{run_number:04}-" +
                         "AGIPD{:02}-gathered.h5".format(self.channel))

            fdir = os.path.join(base_dir,
                                self.meas_type,
                                run_subdir,
                                "gather")

        else:
            # define out files
            fdir = os.path.join(base_dir,
                                self.module,
                                self.temperature,
                                self.meas_type,
                                self.meas_spec,
                                self.run_type)
            # for testing
#            out_base_dir = ("/gpfs/exfel/exp/SPB/201701/p002012/" +
#                               "scratch/user/kuhnm")
#            out_subdir = "tmp"
#            out_dir = os.path.join(out_base_dir,
#                                      out_subdir,
#                                      "gather")
            if self.asic is None:
                fname = ("{}_{}_{}.h5"
                         .format(self.module,
                                 self.meas_type,
                                 self.meas_spec))
            else:
                fname = ("{}_{}_{}_asic{:02}.h5"
                         .format(self.module,
                                 self.meas_type,
                                 self.meas_spec,
                                 self.asic))

        return fdir, fname

    def generate_process_path(self, base_dir, use_xfel_out_format,
                              as_template=False):
        run_subdir = "r" + "-r".join(str(r).zfill(4) for r in self.runs)

        fdir = os.path.join(base_dir,
                            self.meas_type,
                            run_subdir,
                            "process")

        if use_xfel_out_format:
            fname = self.meas_type + "_AGIPD{:02}_xfel.h5"
        else:
            fname = self.meas_type + "_AGIPD{:02}_agipd.h5"

        if not as_template:
            fname = fname.format(self.channel)

        self.use_cfel_gpfs = False
        if self.use_cfel_gpfs:
            fname = ("{}_{}_{}_asic{:02}_processed.h5"
                     .format(self.module,
                             self.meas_type,
                             self.meas_spec,
                             self.asic))

            print("process fname", fname)

            fdir = os.path.join(self.out_base_dir,
                                self.module,
                                self.temperature,
                                self.meas_type,
                                self.meas_spec,
                                self.run_type)

        return fdir, fname

    def generate_join_path(self, base_dir, use_xfel_out_format):
        run_subdir = "r" + "-r".join(str(r).zfill(4) for r in self.runs)

        fdir = os.path.join(base_dir,
                            self.meas_type,
                            run_subdir)

        if use_xfel_out_format:
            fname = "{}_joined_constants_xfel.h5".format(self.meas_type)
        else:
            fname = "{}_joined_constants_agipd.h5".format(self.meas_type)

        self.use_cfel_gpfs = False
        if self.use_cfel_gpfs:
            raise Exception("CFEL gpfs not supported for join at the moment")
#            fname = ("{}_{}_{}_asic{:02d}_processed.h5"
#                     .format(self.module,
#                             self.meas_type,
#                             self.meas_spec,
#                             self.asic))

#            print("process fname", fname)

#            fdir = os.path.join(self.out_base_dir,
#                                self.module,
#                                self.temperature,
#                                self.meas_type,
#                                self.meas_spec,
#                                self.run_type)

        return fdir, fname

    ########################################
    ##                run                 ##
    ########################################

    def run_preprocess(self):
        from gather.preprocess import PreprocessXfel as Preprocess

        if len(self.runs) != 1:
            raise Exception("Preprocessing can only be done per run")

        # define in files
        in_dir, in_file_name = self.generate_raw_path(self.in_base_dir,
                                                      as_template=True)
        in_fname = os.path.join(in_dir, in_file_name)
        # partially substitute the string
        split = in_fname.rsplit("-AGIPD", 1)
        in_fname = split[0].format(run_number=self.runs[0]) + "-AGIPD" + split[1]

        # define out files
        out_dir, out_file_name = self.generate_preproc_path(self.out_base_dir)
        out_fname = os.path.join(out_dir, out_file_name)

        if os.path.exists(out_fname):
            print("output filename = {}".format(out_fname))
            print("WARNING: output file already exist. "
                  "Skipping preprocessing.")
        else:
            utils.create_dir(out_dir)

            print("in_fname=", in_fname)
            print("out_fname=", out_fname)
            print()
            obj = Preprocess(in_fname, out_fname)
            obj.run()

    def run_gather(self):
        if self.meas_type == "pcdrs":
            from gather.gather_pcdrs import AgipdGatherPcdrs as Gather
        else:
            from gather.gather_base import AgipdGatherBase as Gather

#        if self.use_xfel_in_format:
#            self.in_base_dir = "/gpfs/exfel/exp/SPB/201730/p900009"
#            self.out_base_dir = ("/gpfs/exfel/exp/SPB/201730/p900009/" +
#                                 "scratch/user/kuhnm")

        # define in files
        in_dir, in_file_name = self.generate_raw_path(self.in_base_dir)
        in_fname = os.path.join(in_dir, in_file_name)

        # define preprocess files
        preproc_dir, preproc_file_name = (
            self.generate_preproc_path(self.out_base_dir, as_template=True)
        )
        if preproc_dir is not None:
            preproc_fname = os.path.join(preproc_dir, preproc_file_name)
        else:
            preproc_fname = None

        # define out files
        out_dir, out_file_name = self.generate_gather_path(self.out_base_dir)
        out_fname = os.path.join(out_dir, out_file_name)

        if os.path.exists(out_fname):
            print("output filename = {}".format(out_fname))
            print("WARNING: output file already exist. Skipping gather.")
        else:
            utils.create_dir(out_dir)

            print("in_fname=", in_fname)
            print("out_fname=", out_fname)
            print("runs=", self.runs)
            print("preproc_fname", preproc_fname)
            print("max_part=", self.max_part)
            print("asic=", self.asic)
            print("use_xfel_in_format=", self.use_xfel_in_format)
            print()
            obj = Gather(in_fname=in_fname,
                         out_fname=out_fname,
                         runs=self.runs,
                         preproc_fname=preproc_fname,
                         max_part=self.max_part,
                         asic=self.asic,
                         use_xfel_format=self.use_xfel_in_format)
            obj.run()

    def run_process(self):

        if self.meas_type == "dark":
            from process_dark import AgipdProcessDark as Process
        elif self.meas_type == "pcdrs":
            from process_pcdrs import AgipdProcessPcdrs as Process
        else:
            msg = "Process is not supported for type {}".format(self.meas_type)
            raise Exception(msg)

        if self.meas_type == "pcdrs":
            # adjust list of runs
            run_list = ["r" + "-r".join(str(r).zfill(4) for r in self.runs)]

            # define out files
            # the in files for processing are the out ones from gather
            in_dir, in_file_name = self.generate_gather_path(self.in_base_dir)
            in_fname = os.path.join(in_dir, in_file_name)

        else:
            run_list = self.runs

            # define out files
            # the in files for processing are the out ones from gather
            in_dir, in_file_name = self.generate_gather_path(self.in_base_dir)
            in_fname = os.path.join(in_dir, in_file_name)

        # define out files
        out_dir, out_file_name = (
            self.generate_process_path(self.out_base_dir,
                                       self.use_xfel_out_format)
        )
        out_fname = os.path.join(out_dir, out_file_name)

        if os.path.exists(out_fname):
            print("output filename = {}".format(out_fname))
            print("WARNING: output file already exist. Skipping process.")
        else:
            utils.create_dir(out_dir)

            # generate output
            print("channel", self.channel)
            print("in_fname=", in_fname)
            print("out_fname", out_fname)
            print("runs", run_list)
            print("use_xfel_out_format=", self.use_xfel_out_format)
            Process(in_fname,
                    out_fname,
                    run_list,
                    self.use_xfel_out_format)

    #            ParallelProcess(self.asic,
    #                            in_fname,
    #                            np.arange(64),
    #                            np.arange(64),
    #                            np.arange(352),
    #                            self.n_processes,
    #                            self.safety_factor,
    #                            out_fname)

        c_out_dir, c_out_file_name = (
            self.generate_process_path(self.out_base_dir,
                                       not self.use_xfel_out_format)
        )
        c_out_fname = os.path.join(c_out_dir, c_out_file_name)

        print("convert format")
        print("output filename = {}".format(c_out_fname))
        if os.path.exists(c_out_fname):
            print("WARNING: output file already exist. Skipping convert.")
        else:
            if self.use_xfel_out_format:
                c_obj = ConvertFormat(out_fname,
                                      c_out_fname,
                                      "agipd",
                                      self.channel)
            else:
                c_obj = ConvertFormat(out_fname,
                                      c_out_fname,
                                      "xfel",
                                      self.channel)

            c_obj.run()

    def run_join(self):
        # join constants in agipd format as well as the xfel format

        in_dir, in_file_name = (
            self.generate_process_path(self.in_base_dir,
                                       self.use_xfel_out_format,
                                       as_template=True)
        )
        in_fname = os.path.join(in_dir, in_file_name)

        out_dir, out_file_name = (
            self.generate_join_path(self.out_base_dir,
                                    self.use_xfel_out_format)
        )
        out_fname = os.path.join(out_dir, out_file_name)

        obj = JoinConstants(in_fname, out_fname)
        obj.run()

        # now do the other format
        in_dir, in_file_name = (
            self.generate_process_path(self.in_base_dir,
                                       not self.use_xfel_out_format,
                                       as_template=True)
        )
        in_fname = os.path.join(in_dir, in_file_name)

        out_dir, out_file_name = (
            self.generate_join_path(self.out_base_dir,
                                    not self.use_xfel_out_format)
        )
        out_fname = os.path.join(out_dir, out_file_name)

        obj = JoinConstants(in_fname, out_fname)
        obj.run()

#    def run_merge_drscs(self):
#
#        base_path = self.in_base_dir
#        asic_list = self.asic_list
#
#        in_path = os.path.join(base_path,
#                               self.module,
#                               self.temperature,
#                               "drscs")
#        # substitute all except current and asic
#        in_template = (
#            string.Template("${p}/${c}/process/${m}_drscs_${c}_asic${a}_processed.h5")
#            .safe_substitute(p=in_path, m=self.module)
#        )
#        # make a template out of this string to let Merge set current and asic
#        in_template = string.Template(in_template)
#
#        out_dir = os.path.join(base_path,
#                               self.module,
#                               self.temperature,
#                               "drscs",
#                               "merged")
#        out_template = (
#            string.Template("${p}/${m}_drscs_asic${a}_merged.h5")
#            .safe_substitute(p=out_dir,
#                             m=self.module,
#                             t=self.temperature)
#        )
#        out_template = string.Template(out_template)
#
#        if os.path.exists(out_fname):
#            print("output filename = {}".format(out_fname))
#            print("WARNING: output file already exist. Skipping gather.")
#        else:
#
#            utils.create_dir(out_dir)
#
#            ParallelMerge(in_template,
#                          out_template,
#                          asic_list,
#                          self.n_processes,
#                          self.current_list)

    def run_correct(self, run_number):
        data_fname_prefix = ("RAW-{}-AGIPD{}*"
                             .format(run_number.upper(), self.module))
        data_fname = os.path.join(self.in_dir,
                                  run_number,
                                  data_fname_prefix)

        data_parts = glob.glob(data_fname)
        print("data_parts", data_parts)

        for data_fname in data_parts:
            part = int(data_fname[-8:-3])
            print("part", part)

            if self.use_xfel_in_format:
                fname_prefix = "dark_AGIPD{}_xfel".format(self.module)
            else:
                fname_prefix = "dark_AGIPD{}_agipd_".format(self.module)
            dark_fname_prefix = os.path.join(self.dark_dir, fname_prefix)
            dark_fname = glob.glob("{}*".format(dark_fname_prefix))
            if dark_fname:
                dark_fname = dark_fname[0]
            else:
                print("No dark constants found. Quitting.")
                sys.exit(1)
            print(dark_fname)

            if self.use_xfel_in_format:
                fname_prefix = "gain_AGIPD{}_xfel".format(self.module)
            else:
                fname_prefix = "gain_AGIPD{}_agipd_".format(self.module)

            gain_fname_prefix = os.path.join(self.gain_dir, fname_prefix)
            gain_fname = glob.glob("{}*".format(gain_fname_prefix))
            if gain_fname:
                gain_fname = gain_fname[0]
            else:
                print("No gain constants found.")
#                print("No gain constants found. Quitting.")
#                sys.exit(1)

            out_dir = os.path.join(self.out_dir, run_number)
            utils.create_dir(out_dir)

            fname = "corrected_AGIPD{}-S{:05d}.h5".format(self.module, part)
            out_fname = os.path.join(out_dir, fname)

            Correct(data_fname,
                    dark_fname,
                    None,
                    # gain_fname,
                    out_fname,
                    self.energy,
                    self.use_xfel_out_format)

    def cleanup(self):
        # remove gather dir
        pass
