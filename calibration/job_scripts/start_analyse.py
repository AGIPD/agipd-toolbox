#!/usr/bin/env python

# (c) Copyright 2017-2018 DESY, FS-DS
#
# This file is part of the FS-DS AGIPD toolbox.
#
# This software is free: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software.  If not, see <http://www.gnu.org/licenses/>.

"""
@author: Manuela Kuhn <manuela.kuhn@desy.de>
         Jennifer Poehlsen <jennifer.poehlsen@desy.de>
"""

import argparse
import copy
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
        if type(self.runs) != list:
            self.runs = [self.runs]
        if type(self.run_name) != list:
            self.run_name = [self.run_name]

        print("====== Configured parameter in class StartAnalyse ======")
        print(json.dumps(vars(self), sort_keys=True, indent=4))
        print("===================================================")

        self.run()

    def run(self):
        args = copy.deepcopy(vars(self))

        jobs = []
        if self.run_type == "merge":
            Analyse(args)

        elif self.run_type == "preprocess":
            for run in self.runs:
                print("Starting script for run {}\n".format(run))

                args["asic"] = 0
                args["runs"] = [run]

                p = multiprocessing.Process(target=Analyse, args=(args,))
                jobs.append(p)
                p.start()

            for job in jobs:
                job.join()
        else:
            args_ch = copy.deepcopy(args)

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

                    args_ch["module"] = None
                    args_ch["channel"] = ch
                    args_ch["asic"] = asic

                    p = multiprocessing.Process(target=Analyse, args=(args_ch,))
                    jobs.append(p)
                    p.start()

            for job in jobs:
                job.join()


if __name__ == "__main__":
    StartAnalyse()
