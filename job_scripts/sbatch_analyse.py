#!/usr/bin/python3

from __future__ import print_function

import os
import sys
import datetime
import configparser
import subprocess
import argparse

batch_job_dir = os.path.dirname(os.path.realpath(__file__))
script_base_dir = os.path.dirname(batch_job_dir)
conf_dir = os.path.join(script_base_dir, "conf")

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file",
                        type=str,
                        required=True,
                        help="Config file name to get config parameters from")

    parser.add_argument("--run_type",
                        type=str,
                        choices = ["gather", "process"],
                        help="Run type of the analysis")

    args = parser.parse_args()

    return args

class SubmitJobs():
    def __init__(self):
        global script_base_dir
        global conf_dir

        args = get_arguments()

        config_name = args.config_file

        ini_file = os.path.join(conf_dir, "{}.ini".format(config_name))
        print("Using ini_file: {}".format(ini_file))

        config = configparser.ConfigParser()
        config.read(ini_file)

        if not config.sections():
            print("Non ini file found")
            sys.exit(1)

        self.mail_address=config["general"]["mail_address"]
        run_type=config["general"]["run_type"]

        # console parameters overwrite the config file ones
        self.run_type = args.run_type or run_type

        self.module = config["general"]["module"]
        self.temperature = config["general"]["temperature"]
        self.measurement = config["general"]["measurement"]
        current = config["general"]["current"]
        self.current = None

        self.n_jobs = int(config[run_type]["n_jobs"])
        self.n_processes = config[run_type]["n_processes"]

        self.input_dir = config[run_type]["input_dir"]
        self.time_limit = config[run_type]["time_limit"]
        self.output_dir = config[run_type]["output_dir"]

        ### Needed for gather ###
        try:
            self.max_part = config["gather"]["max_part"]
        except KeyError:
            self.max_part = False
        try:
            self.column_spec = config["gather"]["column_spec"]
        except KeyError:
            self.column_spec = False

        # convert str into list
        asic_set = config["general"]["asic_set"][1:-1].split(", ")
        # convert list entries into ints
        asic_set = list(map(int, asic_set))

        self.asic_lists = None
        self.generate_asic_lists(asic_set, self.n_jobs)

        work_dir = os.path.join(self.output_dir, self.module, self.temperature, "sbatch_out")
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
            print("Creating sbatch working dir: {}\n".format(work_dir))

        self.sbatch_params = [
            "--partition", "all",
            "--time", self.time_limit,
            "--nodes", "1",
            "--mail-type", "END",
            "--mail-user", self.mail_address,
            "--workdir", work_dir,
        ]

        script_params = [
            "--script_base_dir", script_base_dir,
            "--run_type", self.run_type,
            "--measurement", self.measurement,
            "--input_dir", self.input_dir,
            "--output_dir", self.output_dir,
            "--n_processes", self.n_processes,
            "--module", self.module,
            "--temperature", self.temperature,
        ]

        if self.run_type == "gather":
            if self.max_part:
                script_params += ["--max_part", self.max_part]

            if self.column_spec:
                script_params += ["--column_spec", self.column_spec]


        #comma seperated string into into list
        current_list = [c.split()[0] for c in current.split(",")]

        for current in current_list:
            self.current = current
            self.script_params = script_params + \
                                 ["--current", current]

            print("run:", current)
            self.run()

    def generate_asic_lists(self, asic_set, n_jobs):

        if len(asic_set) <= n_jobs:
            # if there are less tasks than jobs, start a job for every task
            self.asic_lists = [[i] for i in asic_set]
        else:
            size = int(len(asic_set) / n_jobs)
            rest = len(asic_set) % n_jobs

            # distribute the workload
            self.asic_lists = [asic_set[i:i + size]
                for i in range(0, len(asic_set) - size * (rest + 1), size)]

            # if list to split is not a multiple of size, the rest is equaly
            # distributed over the remaining jobs
            self.asic_lists += [asic_set[i:i + size + 1]
                for i in range(len(self.asic_lists) * size, len(asic_set), size + 1)]

    def run(self):
        global batch_job_dir

        # getting date and time
        now = datetime.datetime.now()
        dt = now.strftime("%Y-%m-%d_%H:%M:%S")

        print("run", self.asic_lists)
        for asic_set in self.asic_lists:
            # map to string to be able to call shell script
            asic_set = " ".join(map(str, asic_set))
            print("Starting job for asics {}\n".format(asic_set))

            self.sbatch_params += [
                "--job-name", "{}_{}_{}_{}_{}_{}".format(self.run_type,
                                                   self.measurement,
                                                   self.module,
                                                   self.temperature,
                                                   self.current,
                                                   asic_set),
                "--output", "{}_{}_{}_{}_{}_{}_{}_%j.out".format(self.run_type,
                                                           self.measurement,
                                                           self.module,
                                                           self.temperature,
                                                           self.current,
                                                           asic_set,
                                                           dt),
                "--error", "{}_{}_{}_{}_{}_{}_{}_%j.err".format(self.run_type,
                                                          self.measurement,
                                                          self.module,
                                                          self.temperature,
                                                          self.current,
                                                          asic_set,
                                                          dt)
            ]

#            #shell_script = os.path.join(batch_job_dir, "analyse.sh")

            # split of the cmd is unneccessary but easier for debugging
            # (e.g. no job should be launched)
#            cmd = [shell_script] + self.script_params + \
#                  [asic_set]


            shell_script = os.path.join(batch_job_dir, "start_analyse.sh")
            cmd = [shell_script, batch_job_dir] + self.script_params + \
                  ["--asic_list", asic_set]

            cmd = ["sbatch"] + self.sbatch_params + cmd

            subprocess.call(cmd)

if __name__ == "__main__":
    SubmitJobs()
