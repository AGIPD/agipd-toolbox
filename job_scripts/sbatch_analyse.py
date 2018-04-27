#!/usr/bin/python3

import argparse
import copy
import datetime
import json
import os
import subprocess
import sys
import time

import measurement_specifics
import run_specifics

BATCH_JOB_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_BASE_DIR = os.path.dirname(BATCH_JOB_DIR)
CONF_DIR = os.path.join(SCRIPT_BASE_DIR, "conf")

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SRC_PATH = os.path.join(BASE_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import utils  # noqa E402


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir",
                        type=str,
                        help="Directory to get data from")
    parser.add_argument("--output_dir",
                        type=str,
                        help="Base directory to write results to")
    parser.add_argument("--type",
                        type=str,
                        choices=["dark", "pcdrs", "drscs"],
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

    parser.add_argument("--overwrite",
                        action="store_true",
                        help="Overwrite existing output file(s)")

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
        self.overwrite = args.overwrite

        if self.use_xfel:
            self.config_file = "xfel"
        else:
            self.config_file = "cfel"

        # load base config
        ini_file = os.path.join(CONF_DIR, "base.ini")
        self.config = dict()
        utils.load_config(self.config, ini_file)

        # override base config with values of user config file
        config_name = args.config_file or self.config_file
        ini_file = os.path.join(CONF_DIR, "{}.ini".format(config_name))
        print("Using ini_file: {}".format(ini_file))
        utils.load_config(self.config, ini_file)

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

        if self.run_type == "preprocess":
            self.run_conf = run_specifics.Preprocess(self.use_xfel)
        elif self.run_type == "gather":
            self.run_conf = run_specifics.Gather(self.use_xfel)
        elif self.run_type == "process":
            self.run_conf = run_specifics.Process(self.use_xfel)
        elif self.run_type == "merge":
            self.run_conf = run_specifics.Merge(self.use_xfel)
        elif self.run_type == "join":
            self.run_conf = run_specifics.Join(self.use_xfel)
        elif self.run_type == "all":
            self.run_conf = run_specifics.All(self.use_xfel)
        else:
            print("run_type:", self.run_type)
            raise Exception("Run type not supported")

        if self.measurement == "dark":
            self.meas_conf = measurement_specifics.Dark(self.use_xfel)
        elif self.measurement == "pcdrs":
            self.meas_conf = measurement_specifics.Pcdrs(self.use_xfel)
        elif self.measurement == "drscs":
            self.meas_conf = measurement_specifics.Drscs(self.use_xfel)
        elif self.measurement == "xray":
            self.meas_conf = measurement_specifics.Xray(self.use_xfel)
        else:
            print("measurement:", self.measurement)
            raise Exception("Measurement not supported")

        self.run_type_list = self.meas_conf.get_run_type_list()

        try:
            # convert str into list
            run_name = self.config[self.measurement]["run_name"]
            if run_name == "None":
                self.run_name = None
            else:
                self.run_name = (self.config[self.measurement]["run_name"]
                                 .split(", "))
        except KeyError:
            self.run_name = None

        self.safety_factor = self.meas_conf.get_safety_factor(self.config)
        self.meas_spec = self.meas_conf.get_meas_spec(self.config)

        c_general = self.config["general"]
        rconf = self.run_conf

        self.module_list = rconf.get_module_list(c_general["module"])
        self.channel_list = rconf.get_channel_list(c_general["channel"])
        self.temperature = rconf.get_temperature(c_general["temperature"])
        self.max_part = rconf.get_max_part(self.config)

        self.use_interleaved = self.config["general"]["use_interleaved"]
        # convert string to bool
        self.use_interleaved = (True
                                if self.use_interleaved == "True"
                                else False)

        self.panel_list = self.module_list + self.channel_list

        self.n_jobs = {}
        self.n_processes = {}
        self.time_limit = {}
        self.input_dir = {}
        self.output_dir = {}
        if self.run_type == 'all':
            for run_type in self.run_type_list:
                c_run_type = self.config[run_type]

                self.n_jobs[run_type] = int(c_run_type['n_jobs'])
                self.n_processes[run_type] = int(c_run_type['n_processes'])
                self.time_limit[run_type] = c_run_type['time_limit']

            runs_using_all_conf = [self.run_type_list[0]]
            if self.use_xfel:
                runs_using_all_conf.append(self.run_type_list[1])
                run_type_list = self.run_type_list[2:]
            else:
                run_type_list = self.run_type_list[1:]

            # the runs which have to use the general input directory
            for run_type in runs_using_all_conf:
                self.input_dir[run_type] = self.config['all']['input_dir']
                self.output_dir[run_type] = self.config['all']['output_dir']

            # the runs which are following in the chain and work on the output
            # of the bevious ones
            for run_type in run_type_list:
                first_rtl = self.run_type_list[0]
                self.input_dir[run_type] = self.output_dir[first_rtl]
                self.output_dir[run_type] = self.output_dir[first_rtl]

        else:
            c_run_type = self.config[self.run_type]

            self.n_jobs[self.run_type] = int(c_run_type["n_jobs"])
            self.n_processes[self.run_type] = int(c_run_type["n_processes"])
            self.time_limit[self.run_type] = c_run_type["time_limit"]

            self.input_dir[self.run_type] = c_run_type["input_dir"]
            self.output_dir[self.run_type] = c_run_type["output_dir"]

        self.run_list = self.run_conf.get_run_list(
            c_run_list=self.config["general"]["run_list"],
            measurement=self.measurement,
            module_list=self.module_list,
            channel_list=self.channel_list,
            temperature=self.temperature,
            meas_spec=self.meas_spec,
            input_dir=self.input_dir,
            meas_conf=self.meas_conf,
            run_name=self.run_name
        )

        if self.config["general"]["asic_set"] == "None":
            self.asic_set = None
        else:
            conf_asic_set = self.config["general"]["asic_set"]
            # convert str into list
            self.asic_set = conf_asic_set[1:-1].split(", ")
            # convert list entries into ints
            self.asic_set = list(map(int, self.asic_set))

        self.dep_overview = {}

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
            c_general['temperature'] = (args.temperature
                                        or c_general['temperature'])
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

        if "all" not in self.config:
            self.config['all'] = {}

        c_run_type = self.config[run_type]
        c_all = self.config['all']

        try:
            # 'all' takes higher priority than the run type specific config
            if 'input_dir' in c_all:
                c_run_type['input_dir'] = (args.input_dir
                                           or c_all['input_dir'])
            else:
                c_run_type['input_dir'] = (args.input_dir
                                           or c_run_type['input_dir'])
        except KeyError:
            raise Exception("No input_dir specified. Abort.")
            sys.exit(1)

        try:
            # 'all' takes higher priority than the run type specific config
            if 'output_dir' in c_all:
                c_run_type['output_dir'] = (args.output_dir
                                            or c_all['output_dir'])
            else:
                c_run_type['output_dir'] = (args.output_dir
                                            or c_run_type['output_dir'])
        except KeyError:
            raise Exception("No output_dir specified. Abort.")
            sys.exit(1)

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

        rtl = self.run_conf.get_run_type_lists_split(self.run_type_list)

        print("run type lists:")
        print(rtl)

        self.init_overview(rtl)

        # jobs concering all panels (before single panel jobs are done)
        jobnums_indp = []
        for run_type in rtl.panel_dep_before:

            # as a placeholder because a panel has to be defined
            self.panel = self.panel_list[0]

            dep_jobs = ":".join(jobnums_indp)
            jn = self.create_job(run_type=run_type,
                                 runs=self.run_list,
                                 run_name=None,
                                 dep_jobs=dep_jobs)

            if jn is not None:
                jobnums_indp.append(jn)

            # collect information for later overview
            self.fill_overview(group_name="all_before",
                               runs=self.run_list,
                               run_type=run_type,
                               jn=jn,
                               jobs=dep_jobs)

        # jobs concering single panel
        jobnums_mod = jobnums_indp
        jobnums_panel = copy.deepcopy(jobnums_indp)
        for panel in self.panel_list:
            self.panel = panel

            jobnums_type = []
            for run_type in rtl.per_panel:
                l_and_n = self.run_conf.get_list_and_name(
                    measurement=self.measurement,
                    run_list=self.run_list,
                    run_name=self.run_name,
                    run_type=run_type
                )

                run_list = l_and_n[0]
                run_name = l_and_n[1]
                print("run_list", run_list)
                print("run_name", run_name)

                print("run_type", run_type)
                print("jobnums_type", jobnums_type)
                dep_jobs = ":".join(jobnums_panel + jobnums_type)
                for i, runs in enumerate(run_list):
                    if self.run_name is not None:
                        rname = run_name[i]
                        overview_name = rname
                    else:
                        rname = None
                        overview_name = runs

                    jn = self.create_job(run_type=run_type,
                                         runs=runs,
                                         run_name=rname,
                                         dep_jobs=dep_jobs)
                    if jn is not None:
                        jobnums_type.append(jn)

                    # collect information for later overview
                    self.fill_overview(group_name=panel,
                                       runs=overview_name,
                                       run_type=run_type,
                                       jn=jn,
                                       jobs=dep_jobs)

            jobnums_mod += jobnums_type

        # jobs concering all panels (after single panel jobs were done)
        jobnums_indp = jobnums_mod
        for run_type in rtl.panel_dep_after:
            dep_jobs = ":".join(jobnums_indp)

            jn = self.create_job(run_type=run_type,
                                 runs=self.run_list,
                                 run_name=None,
                                 dep_jobs=dep_jobs)
            if jn is not None:
                jobnums_indp.append(jn)

            # collect information for later overview
            self.fill_overview(group_name="all_after",
                               runs=self.run_list,
                               run_type=run_type,
                               jn=jn,
                               jobs=dep_jobs)

        self.print_overview()
#        self.generate_mail_body()

    def init_overview(self, rtl):
        # jobs concering all panels (before single panel jobs are done)
        self.dep_overview["all_before"] = {}
        for run_type in rtl.panel_dep_before:
            self.dep_overview["all_before"][run_type] = {}

        # jobs concering single panel
        for panel in self.panel_list:
            panel = str(panel)

            self.dep_overview[panel] = {}

            for run_type in rtl.per_panel:
                self.dep_overview[panel][run_type] = {}

        # jobs concering all panels (after single panel jobs were done)
        self.dep_overview["all_after"] = {}
        for run_type in rtl.panel_dep_after:
            self.dep_overview["all_after"][run_type] = {}

    def fill_overview(self, group_name, runs, run_type, jn, jobs):
        group_name = str(group_name)

        if type(runs) == list:
            runs_string = "-".join(list(map(str, runs)))
        else:
            runs_string = str(runs)

        ov = self.dep_overview[group_name][run_type]
        ov[runs_string] = {}
        ov[runs_string]["jobnum"] = jn
        ov[runs_string]["deb_jobs"] = jobs

    def print_overview(self):
        # print overview of dependencies

        header = ["Panel", "Run type", "Runs", "JobID", "Dependencies"]
        sep = " "

        max_key_len, header_str, sorted_keys = self.prepare_overview(header,
                                                                     sep)

        print("\nDependencies Overview")
        print(header_str)

        # print overview
        #for panel in self.dep_overview:
        for panel in sorted_keys:
            for run_type in self.dep_overview[panel]:
                for runs in self.dep_overview[panel][run_type]:
                    d_o = self.dep_overview[panel][run_type][runs]

                    row = []
                    row.append(panel.ljust(max_key_len[header[0]]))
                    row.append(run_type.ljust(max_key_len[header[1]]))
                    row.append(runs.ljust(max_key_len[header[2]]))
                    row.append(str(d_o["jobnum"]).ljust(max_key_len[header[3]]))

                    if d_o["deb_jobs"] == "":
                        row.append("no dependencies")
                    else:
                        row.append(str(d_o["deb_jobs"]))

                    print(sep.join(row))

    def prepare_overview(self, header, sep):

        # determine how long the space for the keys should be
        max_key_len = {}
        for key in header:
            max_key_len[key] = len(key)

        for panel in self.dep_overview:
            max_key_len["Panel"] = max(max_key_len["Panel"], len(panel))

            for run_type in self.dep_overview[panel]:
                max_key_len["Run type"] = max(max_key_len["Run type"],
                                              len(run_type))

                for runs in self.dep_overview[panel][run_type]:
                    max_key_len["Runs"] = max(max_key_len["Runs"], len(runs))

        # job ids are 6-digit numbers
        max_key_len["JobID"] = max(max_key_len["JobID"], 6)

        # fill up header with spaces according to max length of strings in
        # column
        new_header = []
        for i, key in enumerate(header):
            new_key = key.ljust(max_key_len[key])
            new_header.append(new_key)

        # sort the rows
        key_list = [int(key) for key in self.dep_overview
                    if not key.startswith("all")]
        sorted_keys = sorted(key_list)
        # to sort the list numerically the entries have to be ints
        # but to access the dictionary they have to be converted back to string
        sorted_keys = list(map(str, sorted_keys))
        sorted_keys = ["all_before"] + sorted_keys + ["all_after"]

        # generate header
        header_str = sep.join(list(map(str, new_header)))
        header_str += "\n"

        # add separation between column description and rows
        for key in header:
            value = max_key_len[key]
            header_str += "-" * value + sep

        return max_key_len, header_str, sorted_keys


    def generate_mail_body(self):
        header = ["Panel", "Run type", "Runs", "JobID", "State"]
        sep = " "

        time.sleep(3)
        # get status of jobs
        status = {}
        for panel in self.dep_overview:
            for run_type in self.dep_overview[panel]:
                for runs in self.dep_overview[panel][run_type]:
                    d_o = self.dep_overview[panel][run_type][runs]

                    if d_o["jobnum"] is not None:
                        cmd = ["sacct", "--brief", "-p", "-j", d_o["jobnum"]]
#                        os.system("squeue --user $USER")
                        result = self.submit_job(cmd, jobname="sacct")
                        result = result.split()

                        # the line ends with a separator
                        # -> remove last character
#                        status_header = result[0][:-1].split("|")
#                        print(status_header)
                        for res in result[1:]:
                            # the line ends with a separator
                            status_result = res[:-1].split("|")
#                            print("status_result", status_result)
                            # sacct give the two results for every job
                            # <jobid> and <jobid>.batch
                            if status_result[0] == d_o["jobnum"]:
                                # status results as the entries
                                # 'JobID', 'State', 'ExitCode'
                                status[d_o["jobnum"]] = status_result[1]

        max_key_len, header_str, sorted_keys = self.prepare_overview(header,
                                                                     sep)

        print("\nMail Body")
        print(header_str)

        # print overview
        #for panel in self.dep_overview:
        for panel in sorted_keys:
            for run_type in self.dep_overview[panel]:
                for runs in self.dep_overview[panel][run_type]:
                    d_o = self.dep_overview[panel][run_type][runs]

                    row = []
                    row.append(panel.ljust(max_key_len[header[0]]))
                    row.append(run_type.ljust(max_key_len[header[1]]))
                    row.append(runs.ljust(max_key_len[header[2]]))
                    row.append(str(d_o["jobnum"]).ljust(max_key_len[header[3]]))
                    row.append(status[d_o["jobnum"]])

                    print(sep.join(row))

    def create_job(self, run_type, runs, run_name, dep_jobs):
        print("runs", runs, type(runs))
        self.asic_lists = None
        self.generate_asic_lists(self.asic_set, self.n_jobs[run_type])

        # work_dir is the directory where the sbatch log files are stored
        if self.use_xfel:
            subdir = "-".join(list(map(str, self.run_list)))

            # getting date and time
            now = datetime.datetime.now()
            dt = now.strftime("%Y-%m-%d_%H:%M")

            subdir += "_" + dt

            work_dir = os.path.join(self.output_dir[run_type],
                                    "sbatch_out",
                                    subdir)
        else:
            work_dir = os.path.join(self.output_dir[run_type],
                                    self.panel,
                                    self.temperature,
                                    "sbatch_out")

        if not os.path.exists(work_dir) and not self.no_slurm:
            os.makedirs(work_dir)
            print("Creating sbatch working dir: {}\n".format(work_dir))

        self.set_script_parameters(run_type, runs, run_name, work_dir)

        print("self.sbatch_params")
        print(self.sbatch_params)
        print()
        print("self.script_params")
        print(self.script_params)

        for meas_spec in self.meas_spec:
            if meas_spec is not None:
                self.script_params += ["--meas_spec", meas_spec]

            print("start_job ({}) for {}".format(run_type, meas_spec))
            jobnum = self.start_job(run_type=run_type,
                                    runs=runs,
                                    meas_spec=meas_spec,
                                    dep_jobs=dep_jobs)
        return jobnum

    def set_script_parameters(self, run_type, runs, run_name, work_dir):
        self.sbatch_params = [
            "--partition", self.partition,
            "--time", self.time_limit[run_type],
            "--nodes", "1",
            "--workdir", work_dir,
            "--parsable"
        ]

        if self.mail_address is not None:
            self.sbatch_params += [
                "--mail-type", "END",
                "--mail-user", self.mail_address
            ]

        self.script_params = dict(
            run_type=run_type,
            measurement=self.measurement,
            meas_spec=None,
            input_dir=self.input_dir[run_type],
            output_dir=self.output_dir[run_type],
            n_processes=self.n_processes[run_type],
            run_list=runs,
            run_name=run_name,
            temperature=self.temperature,
            safety_factor=self.safety_factor,
            max_part=self.max_part,
            use_interleaved=self.use_interleaved,
            current_list=None,
            use_xfel_out_format=False,
            overwrite=self.overwrite,
        )

        if self.use_xfel:
            self.script_params["channel"] = [self.panel]
            self.script_params["use_xfel_in_format"] = True
            self.script_params["module"] = []
        else:
            self.script_params["channel"] = []
            self.script_params["use_xfel_in_format"] = False
            self.script_params["module"] = [self.panel]

    def start_job(self, run_type, runs, meas_spec, dep_jobs):
        global BATCH_JOB_DIR

        print("run asic_lists", self.asic_lists)
        for asic_set in self.asic_lists:
            if self.use_xfel:
                if type(runs) == list:
                    runs = "-".join(list(map(str, runs)))

                job_name = ("{}_{}_{}_ch{}"
                            .format(run_type,
                                    self.measurement,
                                    runs,
                                    self.panel))

            else:
                short_temperature = self.temperature[len("temperature_"):]

                job_name = ("{}_{}_{}_{}_{}"
                            .format(run_type,
                                    self.measurement,
                                    self.panel,
                                    short_temperature,
                                    meas_spec))

            if asic_set is not None:
                # map to string to be able to call shell script
                asic_set = " ".join(map(str, asic_set))
                print("Starting job for asics {}\n".format(asic_set))
                job_name = job_name + "_{}".format(asic_set)

            log_name = "{}_%j".format(job_name)

            self.sbatch_params += [
                "--job-name", job_name,
                "--output", log_name + ".out",
                "--error", log_name + ".err"
            ]

            self.script_params["asic_list"] = asic_set
            script_params = json.dumps(self.script_params)

            shell_script = os.path.join(BATCH_JOB_DIR, "start_analyse.sh")
            cmd = [shell_script, BATCH_JOB_DIR, script_params]

            jobnum = None
            if not self.no_slurm:
                if dep_jobs != "":
                    self.sbatch_params += [
                        "--depend=afterok:{}".format(dep_jobs),
                        "--kill-on-invalid-dep=yes"
                    ]

                cmd = ["sbatch"] + self.sbatch_params + cmd
#                print("submitting job with command:", cmd)

                jobnum = self.submit_job(cmd, "{} job".format(run_type))
            else:
                print("cmd {}".format(cmd))
                try:
                    subprocess.call(cmd)
                except:
                    raise

            return jobnum

    def submit_job(self, cmd, jobname):
        p = subprocess.Popen(cmd,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        output, err = p.communicate()
        rc = p.returncode

        # remove newline and "'"
#        jobnum = str(output.rstrip())[:-1]
#        jobnum = jobnum.split("batch job ")[-1]

        jobnum = output.rstrip().decode("unicode_escape")

        if rc != 0:
            print("Error submitting {}".format(jobname))
            print("Error:", err)

        return jobnum


if __name__ == "__main__":
    obj = SubmitJobs()
    obj.run()
