import os
import sys
import datetime
import time
# import numpy as np
from datetime import date
from string import Template
import glob

import utils
#from merge_drscs import ParallelMerge
from correct import Correct

class Analyse():
    def __init__(self, run_type, meas_type, in_base_dir, out_base_dir,
                 n_processes, module, temperature, meas_spec, asic, asic_list,
                 safety_factor, runs, max_part,
                 current_list=None):
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
        self.channel = 0

        self.max_part = max_part

        print("Configured parameter for type {}: ".format(self.run_type))
        print("module: ", self.module)
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

        # Usually the in directory and file names correspond to the
        # meas_type
        self.meas_in = {self.meas_type: self.meas_type}
        # but there are exceptions
        self.meas_in["drscs_dark"] = "drscs"

        self.use_xfel_in_format = True
        self.use_xfel_out_format = False

        self.run()

    def run(self):
        print("\nStarted at", str(datetime.datetime.now()))
        t = time.time()

        if self.run_type == "gather":
            self.run_gather()
        elif self.run_type == "process":
            self.run_process()
        elif self.run_type == "merge":
            self.run_merge_drscs()
        else:
            print("Unsupported argument: run_type {}".format(self.run_type))

        print("\nFinished at", str(datetime.datetime.now()))
        print("took time: ", time.time() - t)

    def generate_raw_path(self, base_dir):
        if self.use_xfel_in_format:
            # define in files
            fdir = os.path.join(base_dir,
                                "raw",
                                "r{run_number:04d}")

            fname = ("RAW-R{run_number:04d}-" +
                     "AGIPD{:02d}".format(self.channel) +
                     "-S{part:05d}.h5")

        else:
            print("meas_in", self.meas_in)
            print("meas_type", self.meas_type)
            print("meas_in[meas_type]", self.meas_in[self.meas_type])
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
                     + "{run_number:04d}_part{part:05d}.nxs")

        return fdir, fname

    def generate_gather_path(self, base_dir):
        if self.use_xfel_in_format:
            # TODO: concider additing this into out_base_dir (joined) and create
            #       subdirs for gathered files
            run_subdir = "r" + "-r".join(str(r).zfill(4) for r in self.runs)

            fdir = os.path.join(base_dir,
                                run_subdir,
                                "gather")

            utils.create_dir(fdir)

            fname = ("{}-AGIPD{:02d}-gathered.h5"
                     .format(run_subdir.upper(), self.channel))

        else:
            # define out files
            fdir = os.path.join(base_dir,
                                self.module,
                                self.temperature,
                                self.meas_type,
                                self.meas_spec,
                                # "gather")
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
                fname = ("{}_{}_{}_asic{:02d}.h5"
                         .format(self.module,
                                 self.meas_type,
                                 self.meas_spec,
                                 self.asic))

        return fdir, fname

    def generate_process_path(self, base_dir):
        today = str(date.today())

        fdir = os.path.join(base_dir,
                            self.meas_type)

        if self.use_xfel_out_format:
            fname = "{}_AGIPD{:02d}_xfel_{}.h5".format(self.meas_type, self.channel, today)
        else:
            fname = "{}_AGIPD{:02d}_agipd_{}.h5".format(self.meas_type, self.channel, today)

        self.use_cfel_gpfs = False
        if self.use_cfel_gpfs:
            fname = ("{}_{}_{}_asic{}_processed.h5"
                     .format(self.module,
                             self.meas_type,
                             self.meas_spec,
                             str(self.asic).zfill(2)))

            print("process fname", fname)

            fdir = os.path.join(self.out_base_dir,
                                self.module,
                                self.temperature,
                                self.meas_type,
                                self.meas_spec,
                                self.run_type)

        return fdir, fname

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

        # define out files
        out_dir, out_file_name = self.generate_gather_path(self.out_base_dir)
        utils.create_dir(out_dir)
        out_fname = os.path.join(out_dir, out_file_name)

        print("in_fname=", in_fname)
        print("out_fname=", out_fname)
        print("runs=", self.runs)
        print("max_part=", self.max_part)
        print("asic=", self.asic)
        print("use_xfel_in_format=", self.use_xfel_in_format)
        print()
        Gather(in_fname,
               out_fname,
               self.runs,
               self.max_part,
               self.asic,
               self.use_xfel_in_format)

    def run_process(self):

        if self.meas_type == "dark":
            from process_dark import AgipdProcessDark as Process
        elif self.meas_type == "pcdrs":
            from process_pcdrs import AgipdProcessPcdrs as Process
        else:
            msg = "Process is not supported for type {}".format(self.meas_type)
            raise Exception(msg)

        # define out files
        # the in files for processing are the out ones from gather
        in_dir, in_file_name = self.generate_gather_path(self.in_base_dir)
        in_fname = os.path.join(in_dir, in_file_name)

        # define out files
        out_dir, out_file_name = self.generate_process_path(self.out_base_dir)
        utils.create_dir(out_dir)
        out_fname = os.path.join(out_dir, out_file_name)

        # generate output
        run_list = ["r" + "-r".join(str(r).zfill(4) for r in self.runs)]
        #run_list = ["r" + "-r".join(str(r).zfill(4) for r in runs) for runs in self.runs]
        #run_list = ["0428", "0429", "0430"]

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

    def run_merge_drscs(self):

        base_path = self.in_base_dir
        asic_list = self.asic_list

        in_path = os.path.join(base_path,
                               self.module,
                               self.temperature,
                               "drscs")
        # substitute all except current and asic
        in_template = (
            Template("${p}/${c}/process/${m}_drscs_${c}_asic${a}_processed.h5")
            .safe_substitute(p=in_path, m=self.module)
        )
        # make a template out of this string to let Merge set current and asic
        in_template = Template(in_template)

        out_dir = os.path.join(base_path,
                               self.module,
                               self.temperature,
                               "drscs",
                               "merged")
        out_template = (
            Template("${p}/${m}_drscs_asic${a}_merged.h5")
            .safe_substitute(p=out_dir,
                             m=self.module,
                             t=self.temperature)
        )
        out_template = Template(out_template)

        utils.create_dir(out_dir)

        ParallelMerge(in_template,
                      out_template,
                      asic_list,
                      self.n_processes,
                      self.current_list)

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
