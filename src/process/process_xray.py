import numpy as np
from process_base import ProcessBase


class ProcessXray(ProcessBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initiate(self):

        n_offsets = len(self.runs)

        self.shapes = {
            "offset": (n_offsets,
                       self.n_memcells,
                       self.n_rows,
                       self.n_cols)
        }

        self.result = {
            "offset": {
                "data": np.empty(self.shapes["offset"]),
                "path": "offset",
                "type": np.int
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
            offset = np.mean(m_analog, axis=0).astype(np.int)

            self.result["offset"]["data"][i, ...] = offset
