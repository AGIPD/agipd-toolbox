#!/usr/bin/python3

from __future__ import print_function

import os
import sys
import datetime
import configparser
import subprocess
import argparse
import glob

BATCH_JOB_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_BASE_DIR = os.path.dirname(BATCH_JOB_DIR)
CONF_DIR = os.path.join(SCRIPT_BASE_DIR, "conf")

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SRC_PATH = os.path.join(BASE_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from generate_paths import GeneratePathsCfel as GeneratePaths


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir",
                        type=str,
#                        required=True,
                        help="Directory to get data from")
    parser.add_argument("--output_dir",
                        type=str,
#                        required=True,
                        help="Base directory to write results to")
    parser.add_argument("--type",
                        type=str,
#                        required=True,
                        choices=["dark", "pcdrs"],
                        help="Which type to run:\n"
                             "dark: generating the dark constants\n"
                             "pcdrs: generating the pulse capacitor constants")
    parser.add_argument("--run_list",
                        type=int,
                        nargs="+",
                        help="Run numbers to extract data from. "
                             "Requirements:\n"
                             "dark: 3 runs for the gain stages "
                             "high, medium, low (in this order)\n"
                             "pcdrs: 8 runs")

    parser.add_argument("--config_file",
                        type=str,
                        help="Config file name to get config parameters from")

    parser.add_argument("--run_type",
                        type=str,
                        choices=["preprocess",
                                 "gather",
                                 "process",
                                 "merge",
                                 "join",
                                 "all"],
                        help="Run type of the analysis")

    parser.add_argument("--cfel",
                        action="store_true",
                        help="Activate cfel mode (default is xfel mode)")
    parser.add_argument("--module",
                        type=str,
                        help="Module to be analysed (e.g M215).\n"
                             "This is only used in cfel mode")
    parser.add_argument("--temperature",
                        type=str,
                        help="The temperature for which the data was taken "
                             "(e.g. temperature_m25C).\n"
                             "This is only used in cfel mode")

    parser.add_argument("--no_slurm",
                        action="store_true",
                        help="The job(s) are not submitted to slurm but run "
                             "interactively")

    args = parser.parse_args()

    if not args.cfel:
        if (not args.input_dir
                or not args.output_dir
                or not args.type
                or not args.run_list):
            msg = "XFEL mode requires a run list to be specified."
            parser.error(msg)

    return args


class SubmitJobs(object):
    def __init__(self):
        global CONF_DIR

        # get command line arguments
        args = get_arguments()

        # consider user running in cfel mode more advance thus being able to
        # add argument to command line; default should always be XFEL case
        self.use_xfel = not args.cfel
        self.no_slurm = args.no_slurm

        if self.use_xfel:
            self.config_file = "xfel"
        else:
            self.config_file = "cfel"

        # load base config
        ini_file = os.path.join(CONF_DIR, "base.ini")
        self.config = dict()
        self.load_config(ini_file)

        # override base config with values of user config file
        config_name = args.config_file or self.config_file
        ini_file = os.path.join(CONF_DIR, "{}.ini".format(config_name))
        print("Using ini_file: {}".format(ini_file))
        self.load_config(ini_file)

        # override user config file with command line arguments
        self.insert_args_in_config(args)

        print("\nsbatch uses config:")
        for key in self.config:
            print(key, self.config[key])
        print()

        try:
            self.mail_address = self.config["general"]["mail_address"]
        except KeyError:
            self.mail_address = None

        self.run_type = self.config["general"]["run_type"]
        self.measurement = self.config["general"]["measurement"]
        self.partition = self.config["general"]["partition"]

        self.safety_factor = None
        self.meas_spec = None
        self.module = None

        if self.measurement == "drscs":
            self.run_type_list = ["gather", "process", "merge"]
        else:
            self.run_type_list = ["gather", "process", "join"]

        try:
            # convert str into list
            run_name = self.config[self.measurement]["run_name"]
            if run_name == "None":
                self.run_name = None
            else:
                self.run_name = self.config[self.measurement]["run_name"].split(", ")
        except KeyError:
            self.run_name = None

        if self.use_xfel:
            self.run_list = self.config["general"]["run_list"]

            if self.run_type == "preprocess":
                # preprocess only should run once and not for every channel
                # -> because the channel is not used in prepocess at all use
                # channel 0 as placeholder
                self.module_list = ["0"]
            else:
                # convert str into list
                self.module_list = self.config["general"]["channel"].split(" ")
            self.temperature = None

            self.run_type_list = ["preprocess"] + self.run_type_list

        else:
            self.module_list = self.config["general"]["module"].split(" ")
            self.temperature = self.config["general"]["temperature"]

            if self.measurement == "drscs":
                self.safety_factor = self.config["drscs"]["safety_factor"]
                self.meas_spec = self.config[self.measurement]["current"]

            elif self.measurement == "dark":
                self.meas_spec = self.config[self.measurement]["tint"]

            elif self.measurement == "xray":
                self.meas_spec = self.config[self.measurement]["element"]
                self.run_type_list = ["gather", "process"]

        self.n_jobs = {}
        self.n_processes = {}
        self.time_limit = {}
        self.input_dir = {}
        self.output_dir = {}
        if self.run_type == 'all':
            for run_type in self.run_type_list:
                self.n_jobs[run_type] = int(self.config[run_type]['n_jobs'])
                self.n_processes[run_type] = self.config[run_type]['n_processes']
                self.time_limit[run_type] = self.config[run_type]['time_limit']

            runs_using_all_conf = [self.run_type_list[0]]
            if self.use_xfel:
                #if self.run_type_list[0] == "preprocess":
                #
                runs_using_all_conf.append(self.run_type_list[1])
                run_type_list = self.run_type_list[2:]
            else:
                run_type_list = self.run_type_list[1:]

            # the runs which have to use the general input directory
            for run_type in runs_using_all_conf:
                self.input_dir[run_type] = self.config['all']['input_dir']
                self.output_dir[run_type] = self.config['all']['output_dir']

            # the runs which are following in the chain and work on the ourput
            # of the bevious ones
            for run_type in run_type_list:
                self.input_dir[run_type] = self.output_dir[self.run_type_list[0]]
                self.output_dir[run_type] = self.output_dir[self.run_type_list[0]]

        else:
            c_run_type = self.config[self.run_type]

            self.n_jobs[self.run_type] = int(c_run_type["n_jobs"])
            self.n_processes[self.run_type] = c_run_type["n_processes"]
            self.time_limit[self.run_type] = c_run_type["time_limit"]

            self.input_dir[self.run_type] = c_run_type["input_dir"]
            self.output_dir[self.run_type] = c_run_type["output_dir"]

        # This has to be done after input_dir is defined thus can not be put
        # in the if above
        if not self.use_xfel:
            if self.run_type in ["gather", "all"]:
                self.run_list = self.get_cfel_run_list()
            else:
                self.run_list = []

        # Needed for gather
        try:
            self.max_part = self.config["gather"]["max_part"]
        except KeyError:
            self.max_part = None

        if self.config["general"]["asic_set"] == "None":
            self.asic_set = None
        else:
            conf_asic_set = self.config["general"]["asic_set"]
            # convert str into list
            self.asic_set = conf_asic_set[1:-1].split(", ")
            # convert list entries into ints
            self.asic_set = list(map(int, self.asic_set))

    def load_config(self, ini_file):

        config = configparser.ConfigParser()
        config.read(ini_file)

        if not config.sections():
            print("ERROR: No ini file found (tried to find {})".format(ini_file))
            sys.exit(1)

        for section, sec_value in config.items():
            if section not in self.config:
                self.config[section] = {}
            for key, key_value in sec_value.items():
                self.config[section][key] = key_value

        return config

    def insert_args_in_config(self, args):
        c_general = self.config['general']

        try:
            c_general['run_type'] = args.run_type or c_general['run_type']
        except:
            raise Exception("No run_type specified. Abort.")
            sys.exit(1)

        try:
            c_general['measurement'] = args.type or c_general['measurement']
        except:
            raise Exception("No measurement type specified. Abort.")
            sys.exit(1)

        # cfel specific
        try:
            c_general['module'] = args.module or c_general['module']
        except:
            if not self.use_xfel:
                raise Exception("No module specified. Abort.")
                sys.exit(1)

        try:
            c_general['temperature'] = args.temperature or c_general['temperature']
        except:
            if not self.use_xfel:
                raise Exception("No temperature specified. Abort.")
                sys.exit(1)


        # xfel specific
        try:
            c_general['run_list'] = args.run_list or c_general['run_list']
        except KeyError:
            if self.use_xfel:
                raise Exception("No run_list specified. Abort.")
                sys.exit(1)
            else:
                c_general['run_list'] = None

        run_type = c_general['run_type']

        #if run_type == "all" and "all" not in self.config:
        if "all" not in self.config:
            self.config['all'] = {}

        c_run_type = self.config[run_type]

        try:
            # 'all' takes higher priority than the run type specific config
            if 'input_dir' in self.config['all']:
                c_run_type['input_dir'] = (args.input_dir
                                           or self.config['all']['input_dir'])
            else:
                c_run_type['input_dir'] = (args.input_dir
                                           or c_run_type['input_dir'])
        except KeyError:
            raise Exception("No input_dir specified. Abort.".format(run_type))
            sys.exit(1)

        try:
            # 'all' takes higher priority than the run type specific config
            if 'output_dir' in self.config['all']:
                c_run_type['output_dir'] = (args.output_dir
                                           or self.config['all']['output_dir'])
            else:
                c_run_type['output_dir'] = (args.output_dir
                                           or c_run_type['output_dir'])
        except KeyError:
            raise Exception("No output_dir specified. Abort.")
            sys.exit(1)

    def get_cfel_run_list(self):

        generate_paths = GeneratePaths(
            run_type=None,
            meas_type=self.measurement,
            out_base_dir=None,
            module=self.module_list[0],
            channel=None,
            temperature=self.temperature,
            meas_spec=self.meas_spec,
            meas_in={self.measurement: self.measurement},
            asic=None,
            runs=None,
            run_name=None,
            use_xfel_out_format=None
        )

        raw_dir, raw_fname = generate_paths.raw(self.input_dir['gather'])
        raw_path = os.path.join(raw_dir, raw_fname)

        run_number_templ = "{run_number:05}"
        # we are trying to determine the run_number, thus we cannot fill it in
        # and have to replace it with a wildcard
        raw = raw_path.replace(run_number_templ, "*")

        raw=raw.format(part=0)
#        print("raw", raw)

        found_files = glob.glob(raw)
#        print("found_files", found_files)

        raw_splitted = raw_path.split(run_number_templ)

        postfix = raw_splitted[-1]
        postfix = postfix.format(part=0)

        middle_part = f[len(raw_splitted[0]) + 2:]

        run_numbers = []
        for f in found_files:
            # also remove underscore
            # only take the runs for which run_names are defined
            if middle_part.startswith(tuple(self.run_name)):
                # cut off the part after the run_number
                rn = f[:-len(postfix)]
                print("rn", rn)
                # the run number now is the tail of the string till the underscore
                # (for drscs it is not the last but the second to last underscore)
                rn = rn.rsplit("_", split_number)[1:]
                # for drscs the splitted elements have to be join again
                rn = "_".join(rn)
                print("rn", rn)

                run_numbers.append(int(rn))

#               if self.use_xfel:
#                    run_numbers.append(int(rn))
#               else:
#                    run_numbers.append(rn)

        return run_numbers

    def generate_asic_lists(self, asic_set, n_jobs):

        if asic_set is None:
            self.asic_lists = [None]
            return

        if len(asic_set) <= n_jobs:
            # if there are less tasks than jobs, start a job for every task
            self.asic_lists = [[i] for i in asic_set]
        else:
            size = int(len(asic_set) / n_jobs)
            rest = len(asic_set) % n_jobs

            # distribute the workload
            stop = len(asic_set) - size * (rest + 1)
            self.asic_lists = [asic_set[i:i + size]
                               for i in range(0, stop, size)]

            # if list to split is not a multiple of size, the rest is equaly
            # distributed over the remaining jobs
            start = len(self.asic_lists) * size
            stop = len(asic_set)
            self.asic_lists += [asic_set[i:i + size + 1]
                                for i in range(start, stop, size + 1)]

    def run(self):
        if self.run_type == "preprocess":
            self.run_type_list_module_dep_before = [self.run_type]
            self.run_type_list_per_module = []
            self.run_type_list_module_dep_after = []
        elif self.run_type == "join":
            self.run_type_list_module_dep_before = []
            self.run_type_list_per_module = []
            self.run_type_list_module_dep_after = [self.run_type]
        # everything which is not preprocess or join
        elif self.run_type != "all":
            self.run_type_list_module_dep_before = []
            self.run_type_list_per_module = [self.run_type]
            self.run_type_list_module_dep_after = []
        # for "all"
        else:
            if self.use_xfel:
                self.run_type_list_module_dep_before = ["preprocess"]
                self.run_type_list_per_module = [t for t in self.run_type_list
                                                 if t not in ["preprocess",
                                                              "join"]]
                self.run_type_list_module_dep_after = ["join"]
            else:
                self.run_type_list_module_dep_before = []
                self.run_type_list_per_module = [t for t in self.run_type_list
                                                 if t != "join"]
                self.run_type_list_module_dep_after = ["join"]

        dep_overview = {}

        # jobs concering all modules (before single module jobs are done)
        dep_overview["all_modules_before"] = {}
        jobnums_indp = []
        for run_type in self.run_type_list_module_dep_before:
            # as a placeholder because a module has to be defined
            self.module = self.module_list[0]

            dep_jobs = ":".join(jobnums_indp)
            jn = self.create_job(run_type=run_type,
                                 runs=self.run_list,
                                 run_name=None,
                                 #run_name=self.run_name,
                                 dep_jobs=dep_jobs)

            if jn is not None:
                jobnums_indp.append(jn)

            # collect information for later overview
            if type(self.run_list) == list:
                runs_string = "-".join(list(map(str, self.run_list)))
            else:
                runs_string = str(self.run_list)

            d_o = dep_overview["all_modules_before"]
            d_o[run_type] = {}
            d_o[run_type][runs_string] = {}
            d_o[run_type][runs_string]["jobnum"] = jn
            d_o[run_type][runs_string]["deb_jobs"] = dep_jobs

        # jobs concering single modules
        jobnums_mod = jobnums_indp
        for module in self.module_list:
            self.module = module

            dep_overview[module] = {}

            jobnums_type = []
            for run_type in self.run_type_list_per_module:
                if run_type == "gather" and self.measurement == "dark":
                    run_list = self.run_list
                    run_name = self.run_name
                else:
                    run_list = [self.run_list]
                    run_name = [self.run_name]
                print("run_list", run_list)
                print("run_name", run_name)

                dep_overview[module][run_type] = {}

                print("run_type", run_type)
                print("jobnums_type", jobnums_type)
                dep_jobs = ":".join(jobnums_type)
                for i, runs in enumerate(run_list):
                    if self.run_name is not None:
                        rname = run_name[i]
                    else:
                        rname = None

                    jn = self.create_job(run_type=run_type,
                                         runs=runs,
                                         run_name=rname,
                                         dep_jobs=dep_jobs)
                    if jn is not None:
                        jobnums_type.append(jn)

                    # collect information for later overview
                    if type(runs) == list:
                        runs_string = "-".join(list(map(str, runs)))
                    else:
                        runs_string = str(runs)

                    d_o = dep_overview[module][run_type]
                    d_o[runs_string] = {}
                    d_o[runs_string]["jobnum"] = jn
                    d_o[runs_string]["deb_jobs"] = dep_jobs

            jobnums_mod += jobnums_type

        # jobs concering all modules (after single module jobs were done)
        dep_overview["all_modules_after"] = {}
        jobnums_indp = jobnums_mod
        for run_type in self.run_type_list_module_dep_after:
            dep_jobs = ":".join(jobnums_indp)

            jn = self.create_job(run_type=run_type,
                                 runs=self.run_list,
                                 run_name=None,
                                 dep_jobs=dep_jobs)
            if jn is not None:
                jobnums_indp.append(jn)

            # collect information for later overview
            if type(self.run_list) == list:
                runs_string = "-".join(list(map(str, self.run_list)))
            else:
                runs_string = str(self.run_list)

            d_o = dep_overview["all_modules_after"]
            d_o[run_type] = {}
            d_o[run_type][runs_string] = {}
            d_o[run_type][runs_string]["jobnum"] = jn
            d_o[run_type][runs_string]["deb_jobs"] = dep_jobs

        # print overview of dependencies
        print("\nDependencies Overview")
        for module in dep_overview:
            for run_type in dep_overview[module]:
                for runs in dep_overview[module][run_type]:
                    d_o = dep_overview[module][run_type][runs]

                    if d_o["deb_jobs"] == "":
                        print("{}\t{}\t{}\t{}\tno dependencies"
                              .format(module,
                                      run_type,
                                      runs,
                                      d_o["jobnum"]))
                    else:
                        print("{}\t{}\t{}\t{}\tdepending on\t{}"
                              .format(module,
                                      run_type,
                                      runs,
                                      d_o["jobnum"],
                                      d_o["deb_jobs"]))

        print("\nCurrent status:\n")
        os.system("squeue --user $USER")

    def create_job(self, run_type, runs, run_name, dep_jobs):
        print("runs", runs, type(runs))
        self.asic_lists = None
        self.generate_asic_lists(self.asic_set, self.n_jobs[run_type])

        # work_dir is the directory where the sbatch log files are stored
        if self.use_xfel:
            work_dir = os.path.join(self.output_dir[run_type],
                                    "sbatch_out")
        else:
            work_dir = os.path.join(self.output_dir[run_type],
                                    self.module,
                                    self.temperature,
                                    "sbatch_out")

        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
            print("Creating sbatch working dir: {}\n".format(work_dir))

        self.sbatch_params = [
            "--partition", self.partition,
            "--time", self.time_limit[run_type],
            "--nodes", "1",
            "--workdir", work_dir,
        ]

        if self.mail_address is not None:
            self.sbatch_params += [
                "--mail-type", "END",
                "--mail-user", self.mail_address
            ]

        self.script_params = [
            "--run_type", run_type,
            "--type", self.measurement,
            "--input_dir", self.input_dir[run_type],
            "--output_dir", self.output_dir[run_type],
            "--n_processes", self.n_processes[run_type],
        ]
        if type(runs) == list:
            self.script_params += ["--run_list"] + [str(r) for r in runs]
        else:
            self.script_params += ["--run_list", str(runs)]

        if run_name is not None:
            if type(run_name) == list:
                self.script_params += ["--run_name"] + [str(r) for r in run_name]
            else:
                self.script_params += ["--run_name", str(run_name)]

        if self.use_xfel:
            self.script_params += [
                "--channel", self.module,
                "--use_xfel_in_format"
            ]
        else:
            self.script_params += ["--module", self.module]

        # parameter which are None raise an error in subprocess
        # -> do not add them
        if self.temperature is not None:
            self.script_params += ["--temperature", self.temperature]

        if self.safety_factor is not None:
            self.script_params += ["--safety_factor", self.safety_factor]

        if self.max_part is not None:
            self.script_params += ["--max_part", self.max_part]

        if self.meas_spec is not None:
            self.script_params += ["--meas_spec", self.meas_spec]

        print("self.sbatch_params")
        print(self.sbatch_params)
        print()
        print("self.script_params")
        print(self.script_params)

        if self.run_type == "merge" and self.measurement == "drscs":
            current_list = self.meas_spec.replace(",", "")
            self.script_params += ["--current_list", current_list]

            # missuse current to set merge as the job name
            self.meas_spec = self.run_type

            print("start_job ({})".format(run_type))
            jobnum = self.start_job(run_type, dep_jobs)

        elif self.measurement == "drscs":
            # comma seperated string into into list
            current_list = [c.split()[0] for c in self.meas_spec.split(",")]

            for current in current_list:
                self.meas_spec = current
                self.script_params += ["--self.meas_spec", self.meas_spec]

                print("start_job ({}): {}".format(run_type, current))
                jobnum = self.start_job(run_type, dep_jobs)

        else:
            print("start_job ({}): {}".format(run_type, self.measurement))
            jobnum = self.start_job(run_type, dep_jobs)

        return jobnum

    def submit_job(self, cmd, jobname):
        p = subprocess.Popen(cmd,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        output, err = p.communicate()
        rc = p.returncode

        # remove newline and "'"
        jobnum = str(output.rstrip())[:-1]
        jobnum = jobnum.split("batch job ")[-1]

        if rc == 0:
            print("{} is {}".format(jobname, jobnum))
        else:
            print("Error submitting {}".format(jobname))
            print("Error:", err)

        return jobnum

    def start_job(self, run_type, dep_jobs):
        global BATCH_JOB_DIR

        # getting date and time
        now = datetime.datetime.now()
        dt = now.strftime("%Y-%m-%d_%H:%M:%S")

        print("run asic_lists", self.asic_lists)
        for asic_set in self.asic_lists:
            job_name = ("{}_{}_{}"
                        .format(run_type,
                                self.measurement,
                                self.module))
            output_name = ("{}_{}_{}"
                           .format(run_type,
                                   self.measurement,
                                   self.module))
            error_name = ("{}_{}_{}"
                          .format(run_type,
                                  self.measurement,
                                  self.module))
            if not self.use_xfel:
                short_temperature = self.temperature[len("temperature_"):]
                job_name = ("{}_{}_{}"
                            .format(job_name,
                                    short_temperature,
                                    self.meas_spec))
                output_name = ("{}_{}_{}"
                               .format(output_name,
                                       short_temperature,
                                       self.meas_spec))
                error_name = ("{}_{}_{}"
                              .format(error_name,
                                      short_temperature,
                                      self.meas_spec))

            if asic_set is not None:
                # map to string to be able to call shell script
                asic_set = " ".join(map(str, asic_set))
                print("Starting job for asics {}\n".format(asic_set))

                job_name = "{}_{}".format(job_name, asic_set)
                output_name = "{}_{}".format(output_name, asic_set)
                error_name = "{}_{}".format(error_name, asic_set)

            self.sbatch_params += [
                "--job-name", job_name,
                "--output", "{}_{}_%j.out".format(output_name, dt),
                "--error", "{}_{}_%j.err".format(error_name, dt)
            ]

#            #shell_script = os.path.join(BATCH_JOB_DIR, "analyse.sh")

            # split of the cmd is unneccessary but easier for debugging
            # (e.g. no job should be launched)
#            cmd = [shell_script] + self.script_params + \
#                  [asic_set]

            shell_script = os.path.join(BATCH_JOB_DIR, "start_analyse.sh")
            cmd = [shell_script, BATCH_JOB_DIR] + self.script_params

            if asic_set is not None:
                cmd += ["--asic_list", asic_set]

            jobnum = None
            if not self.no_slurm:
                if dep_jobs != "":
                    self.sbatch_params += [
                        "--depend=afterok:{}".format(dep_jobs),
                        "--kill-on-invalid-dep=yes"
                    ]

                cmd = ["sbatch"] + self.sbatch_params + cmd
                # print("submitting job with command:", cmd)

                jobnum = self.submit_job(cmd, "{} job".format(run_type))
            else:
                print("cmd {}".format(cmd))
                try:
                    subprocess.call(cmd)
                except:
                    raise

            return jobnum


if __name__ == "__main__":
    obj = SubmitJobs()
    obj.run()
