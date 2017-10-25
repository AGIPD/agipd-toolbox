import os
import sys
import argparse
import datetime
import time
import numpy as np
from string import Template

from utils import create_dir
from gather import GatherData
from parallel_process import ParallelProcess
from merge_drscs import ParallelMerge
from dark_and_xray.gatherDarkData import GatherDarkData
from dark_and_xray.batchProcessDarkData import BatchProcessDarkData
from dark_and_xray.batchProcessXRayTubeData import BatchProcessXRayTubeData
from dark_and_xray.gatherXRayTubeData import GatherXRayTubeData


class Analyse():
    def __init__(self, run_type, meas_type, input_base_dir, output_base_dir,
                 n_processes, module, temperature, meas_spec, asic, asic_list,
                 safety_factor, column_spec, reduced_columns, max_part,
                 current_list=None):
        print("started Analyse")

        self.run_type = run_type
        self.meas_type = meas_type
        self.input_base_dir = input_base_dir
        self.output_base_dir = output_base_dir
        self.n_processes = n_processes
        self.module = module
        self.temperature = temperature
        self.meas_spec = meas_spec
        self.asic = asic
        self.asic_list = asic_list
        self.safety_factor = safety_factor
        self.reduced_columns = reduced_columns
        self.current_list = current_list

        if column_spec and len(column_spec) == 4:
            # [[<column>, <file index>],...]
            # e.g. for a file name and the corresponding entry
            # M234_m8_drscs_itestc150_col15_00001_part00000.nxs
            #                           [15,   1]
            # column_specs = [[15, 9], [26, 10], [37, 11], [48, 12]]
            self.column_specs = [[15, column_spec[0]],
                                 [26, column_spec[1]],
                                 [37, column_spec[2]],
                                 [48, column_spec[3]]]
        else:
            self.column_specs = [15, 26, 37, 48]

        if self.reduced_columns:
            self.column_specs = self.reduced_columns

        # the columns for drscs dark are a permutation of the columns of drscs
        # columns 1, 5 injected -> 3, 7 dark
        # columns 2, 6 injected -> 4, 8 dark
        # columns 3, 7 injected -> 1, 5 dark
        # columns 4, 8 injected -> 2, 6 dark
        if self.meas_type == "drscs_dark":
            c = self.column_specs
            self.column_specs = [c[2], c[3], c[0], c[1]]

        self.max_part = max_part

        print("Configured parameter for type {}: ".format(self.run_type))
        print("module: ", self.module)
        print("temperature: ", self.temperature)
        print("type: ", self.meas_type)
        print("meas_spec: ", self.meas_spec)
        print("asic: ", self.asic)
        print("asic_list: ", self.asic_list)
        print("input_dir: ", self.input_base_dir)
        print("output_dir: ", self.output_base_dir)
        print("column_specs: ", self.column_specs)
        print("max_part: ", self.max_part)
        print("current_list: ", self.current_list)

        # Usually the input directory and file names correspond to the
        # meas_type
        self.meas_input = {meas_type:  meas_type}
        # but there are exceptions
        self.meas_input["drscs_dark"] = "drscs"

        self.use_xfel_input_format = False
        self.use_xfel_output_format = False

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

    def generate_raw_path(self, base_path):
        if self.use_xfel_input_format:
            # define input files
            fdir = os.path.join(base_dir,
                                "raw",
                                "r{run_number}")

            fname = ("RAW-R{run_number}-"
                     "AGIPD{:02d}".format(self.channel)
                     "-S{part:05d}.h5")

        else:
            # define input files
            fdir = os.path.join(base_dir,
                                self.temperature,
                                self.meas_input[self.meas_type])

            if self.meas_type not in ["dark", "xray"]:
                fdir = os.path.join(base_dir,
                                    self.meas_spec)

            #fname = ("{}*_{}_{}_" # only module without location, e.g. M304
            fname = ("{}_{}_{}_"
                     .format(self.module,
                             self.meas_type,
                             self.meas_spec)
                     + "{run_number}_part{part:05d}.nxs")

        return fdir, fname

    def generate_gather_path(self, base_dir):
        if self.use_xfel_input_format:
            run_subdir = "r" + "-r".join(self.runs)

            fdir = os.path.join(base_dir,
                                run_subdir,
                                "gather")

            utils.create_dir(output_dir)

            fname = ("{}-AGIPD{}-gathered.h5"
                     .format(run_subdir.upper(), self.channel))

        else:
            # define output files
            fdir = os.path.join(base_dir,
                                self.module,
                                self.temperature,
                                self.meas_type,
                                self.meas_spec,
                                self.run_type)
                                #"gather")
            #for testing
            #output_base_dir = "/gpfs/exfel/exp/SPB/201701/p002012/scratch/user/kuhnm"
            #output_subdir = "tmp"
            #output_dir = os.path.join(output_base_dir,
            #                          output_subdir,
            #                          "gather")
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

        if self.use_xfel_output_format:
            fname = "dark_AGIPD{}_xfel_{}.h5".format(self.channel, today)
        else:
            fname = "dark_AGIPD{}_agipd_{}.h5".format(self.channel, today)

        self.use_cfel_gpfs = False
        if self.use_cfel_gpfs:
            fname = ("{}_{}_{}_asic{}_processed.h5"
                     .format(self.module,
                             self.meas_type,
                             self.meas_spec,
                             str(self.asic).zfill(2)))

            print("process fname", fname)

            fdir = os.path.join(self.output_base_dir,
                                self.module,
                                self.temperature,
                                self.meas_type,
                                self.meas_spec,
                                self.run_type)

        return fdir, fname


    def run_gather(self):
        # TODO: concider addition this into output_base_dir (joined) and create subdirs for gathered files
        self.runs = ["0428", "0429", "0430"]
        self.runs = ["00012"]

        if self.meas_type == "pcdrs":
            from gather_base import AgipdGatherPcdrs as Gather
        else:
            from gather_base import AgipdGatherBase as Gather

        if self.use_xfel_input_format:
            self.input_base_dir = "/gpfs/exfel/exp/SPB/201730/p900009"
            self.output_base_dir = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm"

            self.channel = 0

        # define input files
        input_dir, input_file_name = self.generate_raw_path(self.input_base_dir)
        input_fname = os.path.join(input_dir, input_file_name)

        # define output files
        output_dir, output_file_name = self.generate_gather_path(self.output_base_dir)
        utils.create_dir(output_dir)
        output_fname = os.path.join(output_dir, output_file_name)

        obj = Gather(input_fname,
                     output_fname,
                     self.runs,
                     self.max_part,
                     True,  # split_asics
                     self.use_xfel_format)

    def run_process(self):

        if self.meas_type == "dark":
            from process_dark import AgipdProcessDark as Process
        elif self.meas_type == "pcdrs":
            from gather_pcdrs import AgipdProcessPcdrs as Process
        else:
            raise Exception("Process is not supported for type {}".format(self.meas_type))

        self.input_base_dir = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm"
        self.output_base_dir = input_base_dir

        run_list = ["0428", "0429", "0430"]
        self.channel = 0
        print("channel", self.channel)

        # define output files
        # the input files for processing are the output ones from gather
        input_dir, input_file_name = self.generate_gather_path(self.input_base_dir)
        input_fname = os.path.join(input_dir, input_file_name)

        # define output files
        output_dir, output_file_name = self.generate_process_path(self.output_base_dir)
        utils.create_dir(output_dir)
        output_fname = os.path.join(output_dir, output_file_name)

        obj = Process(input_fname,
                      output_fname,
                      run_list,
                      use_xfel_format)

#            ParallelProcess(self.asic,
#                            input_fname,
#                            np.arange(64),
#                            np.arange(64),
#                            np.arange(352),
#                            self.n_processes,
#                            self.safety_factor,
#                            output_fname)

    def run_merge_drscs(self):

        base_path = self.input_base_dir
        asic_list = self.asic_list

        input_path = os.path.join(base_path,
                                  self.module,
                                  self.temperature,
                                  "drscs")
        # substitute all except current and asic
        input_template = (
            Template("${p}/${c}/process/${m}_drscs_${c}_asic${a}_processed.h5")
            .safe_substitute(p=input_path, m=self.module)
        )
        # make a template out of this string to let Merge set current and asic
        input_template = Template(input_template)

        output_dir = os.path.join(base_path,
                                  self.module,
                                  self.temperature,
                                  "drscs",
                                  "merged")
        output_template = (
            Template("${p}/${m}_drscs_asic${a}_merged.h5")
            .safe_substitute(p=output_dir,
                             m=self.module,
                             t=self.temperature)
        )
        output_template = Template(output_template)

        create_dir(output_dir)

        ParallelMerge(input_template,
                      output_template,
                      asic_list,
                      self.n_processes,
                      self.current_list)

    def run_correct(self, run_number):
        data_fname_prefix = "RAW-{}-AGIPD{}*".format(run_number.upper(), module)
        data_fname = os.path.join(self.input_dir,
                                  run_number,
                                  data_fname_prefix)

        data_parts = glob.glob(data_fname)
        print("data_parts", data_parts)

        for data_fname in data_parts:
            part = int(data_fname[-8:-3])
            print("part", part)

            if use_xfel_format:
                fname_prefix = "dark_AGIPD{}_xfel".format(module)
            else:
                fname_prefix = "dark_AGIPD{}_agipd_".format(module)
            dark_fname_prefix = os.path.join(self.dark_dir, fname_prefix)
            dark_fname = glob.glob("{}*".format(dark_fname_prefix))
            if dark_fname:
                dark_fname = dark_fname[0]
            else:
                print("No dark constants found. Quitting.")
                sys.exit(1)
            print(dark_fname)

            if use_xfel_format:
                fname_prefix = "gain_AGIPD{}_xfel".format(module)
            else:
                fname_prefix = "gain_AGIPD{}_agipd_".format(module)

            gain_fname_prefix = os.path.join(self.gain_dir, fname_prefix)
            gain_fname = glob.glob("{}*".format(gain_fname_prefix))
            if gain_fname:
                gain_fname = gain_fname[0]
            else:
                print("No gain constants found.")
                #print("No gain constants found. Quitting.")
                #sys.exit(1)

            output_dir = os.path.join(self.output_dir, run_number)
            create_dir(output_dir)

            fname = "corrected_AGIPD{}-S{:05d}.h5".format(module, part)
            output_fname = os.path.join(output_dir, fname)

            obj = Correct(data_fname,
                          dark_fname,
                          None,
                          #gain_fname,
                          output_fname,
                          self.energy,
                          use_xfel_format))

    def cleanup(self):
        # remove gather dir
        #self.output_dir_gather
        pass

