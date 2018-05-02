#!/usr/bin/python3

import argparse
import json
import multiprocessing
import os
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print("BASE_PATH", BASE_PATH)
SRC_PATH = os.path.join(BASE_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from analyse import Analyse  # noqa E402


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("params",
                        type=str,
                        help="All parameter")
    args = parser.parse_args()

    args = json.loads(args.params)

    return args


class StartAnalyse(object):
    def __init__(self):
        args = get_arguments()

        # add all entries of config into the class namespace
        for k, v in args.items():
            setattr(self, k, v)

        self.asic_list = self.asic_list or[None]
        self.current_list = self.current_list if self.current_list else None
        if type(self.run_list) != list:
            self.run_list = [self.run_list]

        print("====== Configured parameter in class StartAnalyse ======")
        print(json.dumps(vars(self), sort_keys=True, indent=4))
        print("===================================================")

        self.run()

    def run(self):
        args = vars(self)

        jobs = []
        if self.run_type == "merge":
            Analyse(args)

        elif self.run_type == "preprocess":
            for run in self.run_list:
                print("Starting script for run {}\n".format(run))

                args["asic"] = 0
                args["runs"] = [run]

                p = multiprocessing.Process(target=Analyse, args=(args,))
                jobs.append(p)
                p.start()

            for job in jobs:
                job.join()
        else:
            for m in self.module:
                for asic in self.asic_list:
                    print("Starting script for module {} and asic {}\n"
                          .format(m, asic))

                    args["module"] = m
                    args["channel"] = None
                    args["asic"] = asic

                    p = multiprocessing.Process(target=Analyse, args=(args,))
                    jobs.append(p)
                    p.start()

            for ch in self.channel:
                for asic in self.asic_list:
                    print("Starting script for channel {} and asic {}\n"
                          .format(ch, asic))

                    args["module"] = None
                    args["channel"] = ch
                    args["asic"] = asic

                    print(args)
                    p = multiprocessing.Process(target=Analyse, args=(args,))
                    jobs.append(p)
                    p.start()

            for job in jobs:
                job.join()


if __name__ == "__main__":
    StartAnalyse()
