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

from collections import namedtuple
import glob
import json
import os
import sys

class RunType(object):
    Rtl = namedtuple("rtl", ["panel_dep_before",
                             "per_panel",
                             "panel_dep_after"])

    def __init__(self):
        self.run_type = None

    def get_channel_list(self, l):
        if type(l) == list:
            return l
        else:
            return [l]

    def get_module_list(self, l):
        return []

    def get_temperature(self, config):
        return None

    def get_max_part(self, config):
        return None

    def get_run_list(self,
                     c_run_list,
                     measurement,
                     module_list,
                     channel_list,
                     temperature,
                     subdir,
                     meas_spec,
                     input_dir,
                     meas_conf,
                     run_name):
        return c_run_list

    def get_run_type_lists_split(self, run_type_list):
        rtl_panel_dep_before = []
        rtl_per_panel = [self.run_type]
        rtl_panel_dep_after = []

        return RunType.Rtl(panel_dep_before=rtl_panel_dep_before,
                           per_panel=rtl_per_panel,
                           panel_dep_after=rtl_panel_dep_after)

    def get_list_and_name(self, measurement, run_list, run_name, run_type):
        new_run_list = [run_list]
        new_run_name = [run_name]

        return new_run_list, new_run_name


class Preprocess(RunType):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_type = "preprocess"

    def get_channel_list(self, conf):
        # preprocess only should run once and not for every channel
        # -> because the channel is not used in prepocess at all use
        # channel 0 as placeholder
        channel_list = [0]

        return channel_list

    def get_run_type_lists_split(self, run_type_list):
        rtl_panel_dep_before = [self.run_type]
        rtl_per_panel = []
        rtl_panel_dep_after = []

        return RunType.Rtl(panel_dep_before=rtl_panel_dep_before,
                           per_panel=rtl_per_panel,
                           panel_dep_after=rtl_panel_dep_after)


class Gather(RunType):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_type = "gather"

    def get_run_list(self,
                     c_run_list,
                     measurement,
                     module_list,
                     channel_list,
                     temperature,
                     subdir,
                     meas_spec,
                     input_dir,
                     meas_conf,
                     run_name):
        return c_run_list

    def get_list_and_name(self, measurement, run_list, run_name, run_type):
        if measurement == "dark":
            return run_list, run_name
        else:
            return super().get_list_and_name(measurement=measurement,
                                             run_list=run_list,
                                             run_name=run_name,
                                             run_type=run_type)


class Process(RunType):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_type = "process"


class Merge(RunType):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_type = "merge"


class Join(RunType):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_type = "join"

    def get_run_type_lists_split(self, run_type_list):
        rtl_panel_dep_before = []
        rtl_per_panel = []
        rtl_panel_dep_after = [self.run_type]

        return RunType.Rtl(panel_dep_before=rtl_panel_dep_before,
                           per_panel=rtl_per_panel,
                           panel_dep_after=rtl_panel_dep_after)


class All(RunType):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_run_list(self,
                     c_run_list,
                     measurement,
                     module_list,
                     channel_list,
                     temperature,
                     subdir,
                     meas_spec,
                     input_dir,
                     meas_conf,
                     run_name):

        return c_run_list

    def get_list_and_name(self, measurement, run_list, run_name, run_type):
        if run_type == "gather" and measurement == "dark":
            return run_list, run_name
        else:
            return super().get_list_and_name(measurement=measurement,
                                             run_list=run_list,
                                             run_name=run_name,
                                             run_type=run_type)

    def get_run_type_lists_split(self, run_type_list):
        rtl_panel_dep_before = ["preprocess"]
        rtl_per_panel = [t for t in run_type_list
                         if t not in ["preprocess", "join"]]
        rtl_panel_dep_after = ["join"]

        return RunType.Rtl(panel_dep_before=rtl_panel_dep_before,
                           per_panel=rtl_per_panel,
                           panel_dep_after=rtl_panel_dep_after)
