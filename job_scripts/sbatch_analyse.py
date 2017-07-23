import os
import sys
import datetime
import configparser
import subprocess

batch_job_dir = os.path.dirname(os.path.realpath(__file__))
script_base_dir = os.path.dirname(batch_job_dir)

class SubmitJobs():
    def __init__(self):
        global batch_job_dir
        global script_base_dir

        ini_file = os.path.join(batch_job_dir, "sbatch.ini")

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

        n_jobs = int(config["general"]["n_jobs"])
        n_processes = config["general"]["n_processes"]

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

        self.asic_lists = [asic_set[i:i+n_jobs] for i in range(0, len(asic_set), n_jobs)]

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
    #        sbatch ${sbatch_params} \
    #               ${batch_job_dir}/analyse.sh ${script_params} asic_set

if __name__ == "__main__":
    SubmitJobs()
