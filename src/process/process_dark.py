import sys
import numpy as np
import os

from process_base import AgipdProcessBase


class AgipdProcessDark(AgipdProcessBase):
    def __init__(self, in_fname, out_fname, runs, use_xfel_format=False):

        self.n_offsets = None

        super().__init__(in_fname=in_fname,
                         out_fname=out_fname,
                         runs=runs,
                         use_xfel_format=use_xfel_format)

    def initiate(self):
        self.n_offsets = len(self.runs)

        self.shapes = {
            "offset": (self.n_offsets,
                       self.n_memcells,
                       self.n_rows,
                       self.n_cols),
            "threshold": (self.n_offsets - 1,
                          self.n_memcells,
                          self.n_rows,
                          self.n_cols)
        }

        self.result = {
            "offset": {
                "data": np.empty(self.shapes["offset"]),
                "path": "offset",
                "type": np.int16
            },
            "gainlevel_mean": {
                "data": np.empty(self.shapes["offset"]),
                "path": "gainlevel_mean",
                "type": np.int16
            },
            "stddev": {
                "data": np.empty(self.shapes["offset"]),
                "path": "stddev",
                "type": np.int16
            },
            "threshold": {
                "data": np.empty(self.shapes["threshold"]),
                "path": "threshold",
                "type": np.float
            }
        }

    def calculate(self):
        for i, run_number in enumerate(self.runs):
            in_fname = self.in_fname.format(run_number=run_number)

            print("Start loading data from {} ...".format(in_fname), end="")
            analog, digital = self.load_data(in_fname)
            print("Done.")

            m_analog, m_digital = self.mask_out_problems(analog=analog,
                                                         digital=digital)

            print("Start computing means and standard deviations ...", end="")
            offset = np.mean(m_analog, axis=0).astype(np.int)
            gainlevel_mean = np.mean(m_digital, axis=0).astype(np.int)

            self.result["offset"]["data"][i, ...] = offset
            self.result["gainlevel_mean"]["data"][i, ...] = gainlevel_mean

            s = self.result["stddev"]["data"][i, ...]
            for cell in np.arange(self.n_memcells):
                s[cell, ...] = m_analog[:, cell, :, :].std(axis=0)
            print("Done.")

        t = self.result["threshold"]["data"]
        md = self.result["gainlevel_mean"]["data"]
        for i in range(self.n_offsets - 1):
            t[i, ...] = (md[i, ...] + md[i + 1, ...]) // 2
