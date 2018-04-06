import numpy as np
from process_base import ProcessBase


class ProcessDark(ProcessBase):
    def __init__(self, **kwargs):

        self.n_offsets = None

        super().__init__(**kwargs)

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

        self.transpose_order = (self._memcell_location,
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
            "stddev": {
                "data": np.empty(self.shapes["offset"]),
                "path": "stddev",
                "type": np.float
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

            print("Start loading data from {} ... ".format(in_fname),
                  end="", flush=True)
            analog, digital = self.load_data(in_fname)
            print("Done.")

            print("Start masking problems ... ", end="", flush=True)
            m_analog, m_digital = self.mask_out_problems(analog=analog,
                                                         digital=digital)
            print("Done.")

            print("Start computing means and standard deviations ... ",
                  end="", flush=True)
            offset = np.mean(m_analog, axis=self._frame_location)
            offset = offset.astype(np.int)
            offset = offset.transpose(self.transpose_order)
            self.result["offset"]["data"][i, ...] = offset

            gainlevel_mean = np.mean(m_digital, axis=self._frame_location)
            gainlevel_mean = gainlevel_mean.astype(np.int)
            gainlevel_mean = gainlevel_mean.transpose(self.transpose_order)
            self.result["gainlevel_mean"]["data"][i, ...] = gainlevel_mean

            s = self.result["stddev"]["data"][i, ...]
            for cell in np.arange(self.n_memcells):
                # generic way to be independend from gathered data reordering
                data_slice = [slice(None), slice(None), slice(None), slice(None)]
                data_slice[self._memcell_location] = cell

                # if the frames are stored after the memory cells
                if self._frame_location > self._memcell_location:
                    fr_loc  = self._frame_location - 1

                s[cell, ...] = m_analog[data_slice].std(axis=fr_loc)
            print("Done.")
            print()

        t = self.result["threshold"]["data"]
        md = self.result["gainlevel_mean"]["data"]
        for i in range(self.n_offsets - 1):
            t[i, ...] = (md[i, ...] + md[i + 1, ...]) // 2
