#!/usr/bin/python3

import datetime
import json
import os
import sys
import time

import utils
# from merge_drscs import ParallelMerge
# from correct import Correct
from join_constants import JoinConstants

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print("BASE_DIR", BASE_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")
GATHER_DIR = os.path.join(SRC_DIR, "gather")
PROCESS_DIR = os.path.join(SRC_DIR, "process")
FACILITY_DIR = os.path.join(BASE_DIR, "facility_specifics")

if GATHER_DIR not in sys.path:
    sys.path.insert(0, GATHER_DIR)

if PROCESS_DIR not in sys.path:
    sys.path.insert(0, PROCESS_DIR)


class Analyse(object):
    def __init__(self, config):
        global FACILITY_DIR

        print("started Analyse")

        # add all entries of config into the class namespace
        for k, v in config.items():
            setattr(self, k, v)

        print("====== Configured parameter in class Analyse ======")
        print(json.dumps(vars(self), sort_keys=True, indent=4))
        print("===================================================")

        # Usually the in directory and file names correspond to the
        # measurement
        self.meas_in = {self.measurement: self.measurement}
        # but there are exceptions
        self.meas_in["drscs_dark"] = "drscs"

        self.properties = {
            "measurement": self.measurement,
            "n_rows_total": 128,
            "n_cols_total": 512,
        }

        if self.measurement == "xray":
            self.properties["max_pulses"] = 2
            self.properties["n_memcells"] = 1
        else:
            self.properties["max_pulses"] = 704
            self.properties["n_memcells"] = 352


        # load facility specifics
        if self.use_xfel_layout:
            self._facility = "xfel"
        else:
            self._facility = "cfel"

        fac_dir = os.path.join(FACILITY_DIR, self._facility)
        if fac_dir not in sys.path:
            sys.path.insert(0, fac_dir)
        # this is located in the facility dir
        from generate_paths import GeneratePaths

        generate_paths = GeneratePaths(
            run_type=self.run_type,
            measurement=self.measurement,
            out_base_dir=self.output_dir,
            module=self.module,
            channel=self.channel,
            temperature=self.temperature,
            meas_spec=self.meas_spec,
            subdir=self.subdir,
            meas_in=self.meas_in,
            asic=self.asic,
            runs=self.runs,
            run_name=self.run_name
        )

        if self.run_type in ["preprocess", "gather"]:
            self.preproc_module, self.layout_module = (
                generate_paths.get_layout_versions(self.input_dir)
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

        # add module location to python path
        preprocess_dir = os.path.join(FACILITY_DIR, self._facility, "preprocess")
        if preprocess_dir not in sys.path:
            sys.path.insert(0, preprocess_dir)

        # load module
        Preprocess = __import__(self.preproc_module).Preprocess

        if len(self.runs) != 1:
            print("runs", self.runs)
            raise Exception("Preprocessing can only be done per run")

        # define in files
        in_dir, in_file_name = self.generate_raw_path(self.input_dir,
                                                      as_template=True)
        in_fname = os.path.join(in_dir, in_file_name)
        # partially substitute the string
        split = in_fname.rsplit("-AGIPD", 1)
        if self.run_name:
            split_tmp = split[0].format(run_name=self.run_name[0],
                                        run_number=self.runs[0])
        else:
            split_tmp = split[0].format(run_number=self.runs[0])

        in_fname = (
            split_tmp +
            "-AGIPD" +
            split[1]
        )

        # define out files
        out_dir, out_file_name = self.generate_preproc_path(self.output_dir)
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
        if self.measurement == "drspc":
            from gather.gather_drspc import GatherDrspc as Gather
        else:
            from gather.gather_base import GatherBase as Gather

        # define in files
        in_dir, in_file_name = self.generate_raw_path(self.input_dir)
        in_fname = os.path.join(in_dir, in_file_name)

        # define preprocess files
        preproc_dir, preproc_file_name = (
            self.generate_preproc_path(base_dir=self.output_dir,
                                       as_template=True)
        )
        if preproc_dir is not None:
            preproc_fname = os.path.join(preproc_dir, preproc_file_name)
        else:
            preproc_fname = None

        # define out files
        out_dir, out_file_name = self.generate_gather_path(
            base_dir=self.output_dir
        )
        out_fname = os.path.join(out_dir, out_file_name)

        if not self.overwrite and os.path.exists(out_fname):
            print("output filename = {}".format(out_fname))
            print("WARNING: output file already exist. Skipping gather.")
        else:
            utils.create_dir(out_dir)

            print("in_fname=", in_fname)
            print("out_fname=", out_fname)
            print("runs=", self.runs)
            print("run_names=", self.run_name)
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
                         run_names=self.run_name,
                         properties=self.properties,
                         use_interleaved=self.use_interleaved,
                         preproc_fname=preproc_fname,
                         max_part=self.max_part,
                         asic=self.asic,
                         layout=self.layout_module,
                         facility=self._facility)
            obj.run()

    def run_process(self):

        run_list = self.runs
        run_name = self.run_name

        if self.measurement == "dark":
            from process_dark import ProcessDark as Process

        elif self.measurement == "drspc":
            from process_drspc import ProcessDrspc as Process

            # adjust list of runs
            run_list = ["r" + "-r".join(str(r).zfill(4) for r in self.runs)]
            if self.run_name != [None]:
                run_name = ["-".join(self.run_name)]

        elif self.measurement == "drscs":
            from process_drspc import ProcessDrscs as Process

        elif self.measurement == "xray":
            from process_xray import ProcessXray as Process
            
        else:
            msg = "Process is not supported for type {}".format(self.measurement)
            raise Exception(msg)

        # define out files
        # the in files for processing are the out ones from gather
        in_dir, in_file_name = self.generate_gather_path(
            base_dir=self.input_dir,
            as_template=True
        )
        in_fname = os.path.join(in_dir, in_file_name)

        # define out files
        out_dir, out_file_name = self.generate_process_path(
            base_dir=self.output_dir,
            use_xfel_format=False
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
            Process(in_fname=in_fname,
                    out_fname=out_fname,
                    runs=run_list,
                    run_name=run_name)

    #            ParallelProcess(self.asic,
    #                            in_fname,
    #                            np.arange(64),
    #                            np.arange(64),
    #                            np.arange(352),
    #                            self.n_processes,
    #                            self.safety_factor,
    #                            out_fname)

        c_out_dir, c_out_file_name = (
            self.generate_process_path(base_dir=self.output_dir,
                                       use_xfel_format=True)
        )
        c_out_fname = os.path.join(c_out_dir, c_out_file_name)

        # do not convert cfel data
        if out_fname != c_out_fname:
            print("convert format")
            print("output filename = {}".format(c_out_fname))
            if os.path.exists(c_out_fname):
                print("WARNING: output file already exist. Skipping convert.")
            else:
                fac_dir = os.path.join(FACILITY_DIR, self._facility)
                if fac_dir not in sys.path:
                    sys.path.insert(0, fac_dir)
                # this is located in the facility dir
                from convert_format import ConvertFormat

                c_obj = ConvertFormat(out_fname,
                                      c_out_fname,
                                      "xfel",
                                      self.channel)

                c_obj.run()

    def run_join(self):
        # join constants in agipd format as well as the xfel format

        in_dir, in_file_name = (
            self.generate_process_path(base_dir=self.input_dir,
                                       use_xfel_format=False,
                                       as_template=True)
        )
        in_fname = os.path.join(in_dir, in_file_name)

        out_dir, out_file_name = (
            self.generate_join_path(base_dir=self.output_dir,
                                    use_xfel_format=False)
        )
        out_fname = os.path.join(out_dir, out_file_name)

        obj = JoinConstants(in_fname, out_fname)
        obj.run()

        # now do the other format
        in_dir, in_file_name = (
            self.generate_process_path(base_dir=self.input_dir,
                                       use_xfel_format=True,
                                       as_template=True)
        )
        in_fname = os.path.join(in_dir, in_file_name)

        out_dir, out_file_name = (
            self.generate_join_path(base_dir=self.output_dir,
                                    use_xfel_format=True)
        )
        out_fname = os.path.join(out_dir, out_file_name)

        obj = JoinConstants(in_fname, out_fname)
        obj.run()

    def run_merge_drscs(self):
        pass
#
#        base_path = self.input_dir
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
#            if self.use_xfel_layout:
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
#            if self.use_xfel_layout:
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
#                    self.energy)
#
#    def cleanup(self):
#        # remove gather dir
#        pass
