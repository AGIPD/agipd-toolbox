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

import numpy as np
from gather_base import GatherBase


class GatherDrspc(GatherBase):
    def __init__(self, **kwargs):

        self.n_runs = 8

        super().__init__(**kwargs)

    def set_pos_indices(self, run_idx, asic):

        # TODO instead of concatenate use two lists

        start = self.n_runs - 1 - run_idx
        stop = self._n_rows_total // 2
        idx_upper = np.arange(start, stop, self.n_runs)
#        print("idx_upper", idx_upper)

#        start = self.n_rows // 2 + run_idx
#        stop = self._n_rows_total
        start = run_idx
        stop = self._n_rows_total // 2
        idx_lower = np.arange(start, stop, self.n_runs)
#        print("idx_lower", idx_lower)

        pos_idx_cols = slice(None)

        if asic is None:
            pos_idx_rows = np.concatenate((idx_upper, idx_lower))
    #        print("pos_idx_rows", pos_idx_rows)
    #        print("pos_idx_cols", pos_idx_cols)
            return [[pos_idx_rows, pos_idx_cols]]

        elif self.asic_in_upper_half():
            return [[idx_upper, pos_idx_cols]]
        else:
            return [[idx_lower, pos_idx_cols]]


