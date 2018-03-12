import numpy as np
from process_base import ProcessBase


class ProcessXray(ProcessBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initiate(self):

        n_offsets = len(self.runs)

        self.shapes = {
            "photon_spacing": (self.n_memcells,
                               self.n_rows,
                               self.n_cols)
        }

        self.result = {
            "photon_spacing": {
                "data": np.empty(self.shapes["photon_spacing"]),
                "path": "photon_spacing",
                "type": np.int16
            },
            "spacing_error": {
                "data": np.empty(self.shapes["photon_spacing"]),
                "path": "spacing_error",
                "type": np.int16
            },
            "peak_stddev": {
                "data": np.empty(self.shapes["photon_spacing"]),
                "path": "peak_stddev",
                "type": np.float
            },
            "peak_error": {
                "data": np.empty(self.shapes["photon_spacing"]),
                "path": "peak_error",
                "type": np.float
            },
            "quality": {
                "data": np.empty(self.shapes["photon_spacing"]),
                "path": "quality",
                "type": np.int16
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

            print("Start computing photon spacing ... ",
                  end="", flush=True)

            for mc in range(self.n_memcells):
                print("memcell {}".format(mc))
                for row in range(self.n_rows):
                    for col in range(self.n_cols):

                        try:
                            (photon_spacing, spacing_error, peak_stddev, peak_errors, quality) = self.get_photon_spacing(analog)
                            idx = (mc, row, col)
                            self.result["photon_spacing"]["data"][idx] = photon_spacing
                            #self.result["spacing_error"]["data"][idx] = spacing_error
                            self.result["peak_stddev"]["data"][idx] = peak_stddev
                            #self.result["peak_error"]["data"][idx] = peak_error
                            self.result["quality"]["data"][idx] = quality


                        except:
                            print("memcell, row, col", mc, row, col)
                            print("analog.shape", analog.shape)
                            raise



            print("Done.")
