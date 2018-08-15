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

from process_base import ProcessBase, NotSupported


class ProcessDrspc(ProcessBase):
    def __init__(self, **kwargs):

        self.fit_interval = None
        self.n_offsets = 2

        super().__init__(**kwargs)

    def initiate(self):
        print("n_memcell={}, n_rows={}, n_cols={}"
              .format(self.n_memcells, self.n_rows, self.n_cols))

        self.shapes = {
            "offset": [None, None, None, None],
            # use already transposed layout
            "threshold": (
                self.n_offsets - 1,
                self.n_memcells,
                self.n_rows,
                self.n_cols,
            )
        }
        # Have the offset shape the same data layout as the gathered data
        # for performance
        self.shapes["offset"][self._frame_location] = self.n_offsets
        self.shapes["offset"][self._memcell_location] = self.n_memcells
        self.shapes["offset"][self._row_location] = self.n_rows
        self.shapes["offset"][self._col_location] = self.n_cols

        # transpose the results to match raw data layout
        self.transpose_order = (self._frame_location, # offset
                                self._memcell_location,
                                self._row_location,
                                self._col_location)

        self.result = {
            "offset": {
                "data": np.empty(self.shapes["offset"]),
                "path": "offset",
                "type": np.int
            },
            "gainlevel_mean": {
                "data": np.empty(self.shapes["offset"]),
                "path": "gainlevel_mean",
                "type": np.int
            },
            "slope": {
                "data": np.empty(self.shapes["offset"]),
                "path": "slope",
                "type": np.float
            },
            "threshold": {
                "data": np.empty(self.shapes["threshold"]),
                "path": "threshold",
                "type": np.float
            }
        }

    def determine_fit_intervals(self):
        self.fit_interval = [[42, 122], [402, 552]]
#        self.fit_interval = [[40, 120], [400, 550]]

    def calculate(self):
        if len(self.runs) != 1:
            print("runs", self.runs)
            print("run_name", self.run_names)
            raise NotSupported("Input contains of multiple runs. Abort")

        in_fname = self.in_fname.format(run_number=self.runs[0],
                                        run_name=self.run_names[0])
        analog, digital = self.load_data(in_fname)

        mask = self.get_mask(analog=analog, digital=digital)

        self.determine_fit_intervals()

        # make sure that the data looks like expected
        if (self._row_location > self._col_location
                and self._col_location > self._memcell_location):
            print("found location:\nrow: {}, col: {}, memcell: {}"
                  .format(self._row_location,
                          self._col_location,
                          self._memcell_location))
            print("required order: "
                  "row_location, col_location, memcell_location")
            raise Exception("data layout does not fit algorithm")

        # instead of directly access the data
        # like analog[slice(*fit_interval), mc, row, col]
        # use a more generic approach to make changes in
        # the data layout easier to fix
        data_slice = [slice(None), slice(None),
                      slice(None), slice(None)]

        offset = self.result["offset"]["data"]
        slope = self.result["slope"]["data"]
        gainlevel_mean = self.result["gainlevel_mean"]["data"]

        print("Start fitting")
        for row in range(self.n_rows):
            print("row {}".format(row))
            data_slice[self._row_location] = row

            for col in range(self.n_cols):
                data_slice[self._col_location] = col

                for mc in range(self.n_memcells):
                    data_slice[self._memcell_location] = mc

                    for i in range(self.n_offsets):
                        fit_interval = self.fit_interval[i]

                        data_slice[self._frame_location] = (
                            slice(*fit_interval)
                        )

                        try:
                            x = np.arange(*fit_interval)
                            y = analog[data_slice]

                            # mask out lost frames,...
                            missing = mask[data_slice]

                            res = self.fit_linear(x, y, missing)

                            idx = (row, col, mc, i)
                            slope[idx] = res[0][0]
                            offset[idx] = res[0][1]
                            gainlevel_mean[idx] = np.mean(digital[data_slice])
                        except:
                            print("row, col, memcell", row, col, mc)
                            print("analog.shape", analog.shape)
                            print("res", res)
                            raise

        print("Transpose")
        offset = offset.transpose(self.transpose_order)
        slope = slope.transpose(self.transpose_order)
        gainlevel_mean = gainlevel_mean.transpose(self.transpose_order)

        self.result["offset"]["data"] = offset
        self.result["slope"]["data"] = slope
        self.result["gainlevel_mean"]["data"] = gainlevel_mean

        print("Calculate threshold")
        t = self.result["threshold"]["data"]
        gm = self.result["gainlevel_mean"]["data"]
        for i in range(self.n_offsets - 1):
            t[i, ...] = (gm[i, ...] + gm[i + 1, ...]) // 2
