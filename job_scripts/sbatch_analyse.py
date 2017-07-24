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

        mail_address=config["general"]["mail_address"]
        run_type=config["general"]["run_type"]

        module = config["general"]["module"]
        temperature = config["general"]["temperature"]
        measurement = config["general"]["measurement"]
        current = config["general"]["current"]

        n_jobs = int(config[run_type]["n_jobs"])
        n_processes = config[run_type]["n_processes"]

        input_dir = config[run_type]["input_dir"]
        time_limit = config[run_type]["time_limit"]
        output_dir = config[run_type]["output_dir"]

        ### Needed for gather ###
        max_part = config["gather"]["max_part"]
        column_spec = config["gather"]["column_spec"]

        # convert str into list
        asic_set = config["general"]["asic_set"][1:-1].split(", ")
        # convert list entries into ints
        asic_set = list(map(int, asic_set))

        self.asic_lists = None
        self.generate_asic_lists(asic_set, n_jobs)

        work_dir = os.path.join(output_dir, module, temperature, "sbatch_out")
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
            print("Creating sbatch working dir: {}\n".format(work_dir))

        # getting date and time
        now = datetime.datetime.now()
        dt = now.strftime("%Y-%m-%d_%H:%M:%S")

        self.sbatch_params = [
            "--partition", "all",
            "--time", time_limit,
            "--nodes", "1",
            "--mail-type", "END",
            "--mail-user", mail_address,
            "--workdir", work_dir,
            "--job-name", "{}_{}_{}".format(run_type, measurement, module),
            "--output", "{}_{}_{}_{}_%j.out".format(run_type, measurement, module, dt),
            "--error", "{}_{}_{}_{}_%j.err".format(run_type, measurement, module, dt)
        ]

        self.script_params = [
            "--script_base_dir", script_base_dir,
            "--run_type", run_type,
            "--measurement", measurement,
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--n_processes", n_processes,
            "--module", module,
            "--temperature", temperature,
            "--current", current
        ]

        if run_type == "gather":
            self.script_params += ["--max_part", max_part,
                                   "--column_spec", column_spec]

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

        print("run", self.asic_lists)
        for asic_set in self.asic_lists:
            # map to string to be able to call shell script
            asic_set = " ".join(map(str, asic_set))
            print("Starting job for asics {}\n".format(asic_set))

            shell_script = os.path.join(batch_job_dir, "analyse.sh")

            # split of the cmd is unneccessary but easier for debugging
            # (e.g. no job should be launched)
            cmd = [shell_script] + self.script_params + \
                  [asic_set]
            cmd = ["sbatch"] + self.sbatch_params + cmd
            #print("command", cmd)

            subprocess.call(cmd)

if __name__ == "__main__":
    SubmitJobs()
