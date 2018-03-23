#!/usr/bin/python3

import os
import sys
import datetime
import time
# import glob

import utils
# from merge_drscs import ParallelMerge
# from correct import Correct
from convert_format import ConvertFormat
from join_constants import JoinConstants

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print("BASE_DIR", BASE_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")
PREPROCESS_DIR = os.path.join(SRC_DIR, "preprocess")
GATHER_DIR = os.path.join(SRC_DIR, "gather")
PROCESS_DIR = os.path.join(SRC_DIR, "process")

if PREPROCESS_DIR not in sys.path:
    sys.path.insert(0, PREPROCESS_DIR)

if GATHER_DIR not in sys.path:
    sys.path.insert(0, GATHER_DIR)

if PROCESS_DIR not in sys.path:
    sys.path.insert(0, PROCESS_DIR)


class Analyse(object):
    def __init__(self,
                 run_type,
                 meas_type,
                 in_base_dir,
                 out_base_dir,
                 n_processes,
                 module,
                 channel,
                 temperature,
                 meas_spec,
                 asic,
                 asic_list,
                 safety_factor,
                 runs,
                 run_name,
                 max_part,
                 use_interleaved,
                 current_list=None,
                 use_xfel_in_format=True,
                 use_xfel_out_format=False,
                 overwrite=False):

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
        self.run_name = run_name
        self.current_list = current_list
        self.channel = channel

        self.max_part = max_part
        self.use_interleaved = use_interleaved

        self.use_xfel_in_format = use_xfel_in_format
        self.use_xfel_out_format = use_xfel_out_format
        self.overwrite = overwrite

        print("====== Configured parameter in class Analyse ======")
        print("run_type: {}".format(self.run_type))
        print("module: ", self.module)
        print("channel: ", self.channel)
        print("temperature: ", self.temperature)
        print("measurement: ", self.meas_type)
        print("meas_spec: ", self.meas_spec)
        print("asic: ", self.asic)
        print("asic_list: ", self.asic_list)
        print("in_dir: ", self.in_base_dir)
        print("out_dir: ", self.out_base_dir)
        print("runs: ", self.runs)
        print("run_name: ", self.run_name)
        print("max_part: ", self.max_part)
        print("use_interleaved", self.use_interleaved)
        print("current_list: ", self.current_list)
        print("use_xfel_in_format: ", self.use_xfel_in_format)
        print("use_xfel_out_format: ", self.use_xfel_out_format)
        print("overwrite: ", self.overwrite)
        print("===================================================")

        # Usually the in directory and file names correspond to the
        # meas_type
        self.meas_in = {self.meas_type: self.meas_type}
        # but there are exceptions
        self.meas_in["drscs_dark"] = "drscs"

        self.properties = {
            "measurement": self.meas_type,
            "n_rows_total": 128,
            "n_cols_total": 512,
        }

        if self.meas_type == "xray":
            self.properties["max_pulses"] = 2
            self.properties["n_memcells"] = 1
        else:
            self.properties["max_pulses"] = 704
            self.properties["n_memcells"] = 352

        if self.use_xfel_in_format:
            from generate_paths import GeneratePathsXfel as GeneratePaths
        else:
            from generate_paths import GeneratePathsCfel as GeneratePaths

        generate_paths = GeneratePaths(
            run_type=self.run_type,
            meas_type=self.meas_type,
            out_base_dir=self.out_base_dir,
            module=self.module,
            channel=self.channel,
            temperature=self.temperature,
            meas_spec=self.meas_spec,
            meas_in=self.meas_in,
            asic=self.asic,
            runs=self.runs,
            run_name=self.run_name,
            use_xfel_out_format=self.use_xfel_out_format
        )

        if self.run_type in ["preprocess", "gather"]:
            self.preproc_module, self.layout_module = (
                generate_paths.get_layout_versions(self.in_base_dir)
            )

        self.generate_raw_path = generate_paths.raw
        self.generate_preproc_path = generate_paths.preproc
        self.generate_gather_path = generate_paths.gather
        self.generate_process_path = generate_paths.process
        self.generate_join_path = generate_paths.join

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

    def run_preprocess(self):
        Preprocess = __import__(self.preproc_module).Preprocess

        if len(self.runs) != 1:
            raise Exception("Preprocessing can only be done per run")

        # define in files
        in_dir, in_file_name = self.generate_raw_path(self.in_base_dir,
                                                      as_template=True)
        in_fname = os.path.join(in_dir, in_file_name)
        # partially substitute the string
        split = in_fname.rsplit("-AGIPD", 1)
        in_fname = (
            split[0].format(run_number=self.runs[0]) +
            "-AGIPD" +
            split[1]
        )

        # define out files
        out_dir, out_file_name = self.generate_preproc_path(self.out_base_dir)
        out_fname = os.path.join(out_dir, out_file_name)

        if not self.overwrite and os.path.exists(out_fname):
            print("output filename = {}".format(out_fname))
            print("WARNING: output file already exist. "
                  "Skipping preprocessing.")
        else:
            utils.create_dir(out_dir)

            print("in_fname=", in_fname)
            print("out_fname=", out_fname)
            print()
            obj = Preprocess(in_fname, out_fname, self.use_interleaved)
            obj.run()

    def run_gather(self):
        if self.meas_type == "pcdrs":
            from gather.gather_pcdrs import GatherPcdrs as Gather
        else:
            from gather.gather_base import GatherBase as Gather

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

        if not self.overwrite and os.path.exists(out_fname):
            print("output filename = {}".format(out_fname))
            print("WARNING: output file already exist. Skipping gather.")
        else:
            utils.create_dir(out_dir)

            print("in_fname=", in_fname)
            print("out_fname=", out_fname)
            print("runs=", self.runs)
            print("properties", self.properties)
            print("use_interleaved", self.use_interleaved)
            print("preproc_fname", preproc_fname)
            print("max_part=", self.max_part)
            print("asic=", self.asic)
            print("layout=", self.layout_module)
            print()
            obj = Gather(in_fname=in_fname,
                         out_fname=out_fname,
                         runs=self.runs,
                         properties=self.properties,
                         use_interleaved=self.use_interleaved,
                         preproc_fname=preproc_fname,
                         max_part=self.max_part,
                         asic=self.asic,
                         layout=self.layout_module)
            obj.run()

    def run_process(self):

        if self.meas_type == "dark":
            from process_dark import ProcessDark as Process

            if self.use_xfel_in_format or self.run_name is None:
                run_list = self.runs
            else:
                run_list = self.run_name

        elif self.meas_type == "pcdrs":
            from process_pcdrs import ProcessPcdrs as Process

            # adjust list of runs
            run_list = ["r" + "-r".join(str(r).zfill(4) for r in self.runs)]

        elif self.meas_type == "drscs":
            from process_pcdrs import AgipdProcessDrscs as Process

            if not self.use_xfel_in_format:
                run_list = self.runs

        else:
            msg = "Process is not supported for type {}".format(self.meas_type)
            raise Exception(msg)

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

        if not self.overwrite and os.path.exists(out_fname):
            print("output filename = {}".format(out_fname))
            print("WARNING: output file already exist. Skipping process.")
        else:
            utils.create_dir(out_dir)

            # generate out_put
            print("Used parameter for process:")
            print("in_fname=", in_fname)
            print("out_fname", out_fname)
            print("runs", run_list)
            print("use_xfel_out_format=", self.use_xfel_out_format)
            Process(in_fname=in_fname,
                    out_fname=out_fname,
                    runs=run_list,
                    use_xfel_format=self.use_xfel_out_format)

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

        # do not convert cfel data
        if out_fname != c_out_fname:
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
#        if not self.overwrite and os.path.exists(out_fname):
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

#    def run_correct(self, run_number):
#        data_fname_prefix = ("RAW-{}-AGIPD{}*"
#                             .format(run_number.upper(), self.module))
#        data_fname = os.path.join(self.in_dir,
#                                  run_number,
#                                  data_fname_prefix)
#
#        data_parts = glob.glob(data_fname)
#        print("data_parts", data_parts)
#
#        for data_fname in data_parts:
#            part = int(data_fname[-8:-3])
#            print("part", part)
#
#            if self.use_xfel_in_format:
#                fname_prefix = "dark_AGIPD{}_xfel".format(self.module)
#            else:
#                fname_prefix = "dark_AGIPD{}_agipd_".format(self.module)
#            dark_fname_prefix = os.path.join(self.dark_dir, fname_prefix)
#            dark_fname = glob.glob("{}*".format(dark_fname_prefix))
#            if dark_fname:
#                dark_fname = dark_fname[0]
#            else:
#                print("No dark constants found. Quitting.")
#                sys.exit(1)
#            print(dark_fname)
#
#            if self.use_xfel_in_format:
#                fname_prefix = "gain_AGIPD{}_xfel".format(self.module)
#            else:
#                fname_prefix = "gain_AGIPD{}_agipd_".format(self.module)
#
#            gain_fname_prefix = os.path.join(self.gain_dir, fname_prefix)
#            gain_fname = glob.glob("{}*".format(gain_fname_prefix))
#            if gain_fname:
#                gain_fname = gain_fname[0]
#            else:
#                print("No gain constants found.")
#                #print("No gain constants found. Quitting.")
#                #sys.exit(1)
#
#            out_dir = os.path.join(self.out_dir, run_number)
#            utils.create_dir(out_dir)
#
#            fname = "corrected_AGIPD{}-S{:05d}.h5".format(self.module, part)
#            out_fname = os.path.join(out_dir, fname)
#
#            Correct(data_fname,
#                    dark_fname,
#                    None,
#                    # gain_fname,
#                    out_fname,
#                    self.energy,
#                    self.use_xfel_out_format)
#
#    def cleanup(self):
#        # remove gather dir
#        pass
