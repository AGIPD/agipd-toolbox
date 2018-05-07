import argparse
import subprocess
import os
import sys

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--jobid",
                        type=str)
    args = parser.parse_args()

    return args

def submit_job(cmd, jobname):
    p = subprocess.Popen(cmd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    output, err = p.communicate()
    rc = p.returncode

    print("cmd", cmd)
    print("output", output)
#    print(output.rstrip())
    print()
    jobnum = output.rstrip().decode("unicode_escape")

    if rc != 0:
        print("Error submitting {}".format(jobname))
        print("Error:", err)

    return jobnum

if __name__ == "__main__":
    args = get_arguments()

    jobid = args.jobid

    cmd = ["sacct", "--brief", "-p", "-j", str(jobid)]
    result = submit_job(cmd, jobname="sacct")
    print("result", result)
