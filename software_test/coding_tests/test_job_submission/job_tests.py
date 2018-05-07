#!/usr/local/bin/python

import subprocess
import os


def submit_job(cmd, jobname):
    p = subprocess.Popen(cmd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    output, err = p.communicate()
    rc = p.returncode

    jobnum = str(output.rstrip())[:-1]
    jobnum = jobnum.split("batch job ")[-1]

    if rc == 0:
        print("{} is {}".format(jobname, jobnum))
    else:
        print("Error submitting {}".format(jobname))
        print("Error:", err)

    return jobnum


path = "/home/kuhnm/calibration/job_scripts/test"

# submit the first job
cmd = ["sbatch", "{}/job1.sh".format(path)]

print("Submitting Job1 with command:", cmd)
jobnum = submit_job(cmd, "job1")

# submit the second job to be dependent on the first
cmd = ["sbatch",
       "--depend=afterok:{}".format(jobnum),
       "--kill-on-invalid-dep=yes",
       "{}/job2.sh".format(path)]

print("submitting job2 with command:", cmd)
jobnum = submit_job(cmd, "job2")

# submit the third job to be dependent on the second
cmd = ["sbatch",
       "--depend=afterok:{}".format(jobnum),
       "--kill-on-invalid-dep=yes",
       "{}/job3.sh".format(path)]

print("submitting job3 with command:", cmd)
jobnum = submit_job(cmd, "job3")

print("\nCurrent status:\n")
os.system("squeue --user $USER")
