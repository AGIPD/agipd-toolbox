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

import sys
from __init__ import FACILITY_DIR

if FACILITY_DIR not in sys.path:
    sys.path.insert(0, FACILITY_DIR)

from cfel.measurement_specifics import Measurement as CfelMeasurement


# overwrite the cfel measurements -> only xfel specify differences
class Measurement(CfelMeasurement):

    def get_run_type_list(self):
        """ Get the list of steps to be taken to get the constants.

        Return:
            A list of run_types, e.g. ["gather", "process"].
        """

        run_type_list = ["preprocess", "gather", "process", "join"]

        return run_type_list


class Dark(Measurement):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measurement = "dark"


class Drspc(Measurement):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measurement = "drspc"


class Drscs(Measurement):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measurement = "drscs"

    def get_run_type_list(self):
        """ Get the list of steps to be taken to get the constants.

        Return:
            A list of run_types, e.g. ["gather", "process"]
        """

        run_type_list = ["preprocess", "gather", "process", "merge", "join"]

        return run_type_list

    def mod_params(self, param_list, run_type):
        """ Add measurement specific parameters.

        Args:
            param_list (list): A list with the already set parameters.
            run_type (str): Which run type is used.

        Return:
            A list with the modified parameters.
        """

        currents = "-".join(self.meas_spec)
        self.script_params += ["--current_list", currents]

        return param_list


class Xray(Measurement):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measurement = "xray"

    def get_run_type_list(self):
        """ Get the list of steps to be taken to get the constants.

        Return:
            A list of run_types, e.g. ["gather", "process"]
        """

        run_type_list = ["preprocess", "gather", "process", "join"]

        return run_type_list
