import numpy as np
from process_base import ProcessBase


class ProcessPcdrs(ProcessBase):
    def __init__(self, **kwargs):

        self.fit_interval = None
        self.n_offsets = 2

        super().__init__(**kwargs)

    def initiate(self):
        print("n_memcell={}, n_rows={}, n_cols={}"
              .format(self.n_memcells, self.n_cols, self.n_rows))

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
        analog, digital = self.load_data(self.in_fname)

        mask = self.get_mask(analog=analog, digital=digital)

        self.determine_fit_intervals()

        print("Start fitting")
        for i in range(self.n_offsets):
            fit_interval = self.fit_interval[i]
            x = np.arange(*fit_interval)

            for mc in range(self.n_memcells):
                print("gain stage {}, memcell {}".format(i, mc))
                for row in range(self.n_rows):
                    for col in range(self.n_cols):

                        try:
                            y = analog[slice(*fit_interval),
                                       mc, row, col]

                            # mask out lost frames,...
                            missing = mask[slice(*fit_interval),
                                           mc, row, col]

                            res = self.fit_linear(x, y, missing)

                            gain_mean = np.mean(digital[slice(*fit_interval),
                                                        mc, row, col])

                            result = self.result
                            idx = (i, mc, row, col)
                            result["slope"]["data"][idx] = res[0][0]
                            result["offset"]["data"][idx] = res[0][1]
                            result["gainlevel_mean"]["data"][idx] = gain_mean
                        except:
                            print("memcell, row, col", mc, row, col)
                            print("analog.shape", analog.shape)
                            print("res", res)
                            raise

        print("Calculate threshold")
        t = self.result["threshold"]["data"]
        md = self.result["gainlevel_mean"]["data"]
        for i in range(self.n_offsets - 1):
            t[i, ...] = (md[i, ...] + md[i + 1, ...]) // 2
