import numpy as np
from gather_base import GatherBase


class GatherPcdrs(GatherBase):
    def __init__(self, **kwargs):

        self.n_runs = 8

        super().__init__(**kwargs)

    def set_pos_indices(self, run_idx):

        # TODO instead of concatenate use two lists

        start = self.n_runs - 1 - run_idx
        stop = self.n_rows // 2
        idx_upper = np.arange(start, stop, self.n_runs)
#        print("idx_upper", idx_upper)

        start = self.n_rows // 2 + run_idx
        stop = self.n_rows
        idx_lower = np.arange(start, stop, self.n_runs)
#        print("idx_lower", idx_lower)

        pos_idx_rows = np.concatenate((idx_upper, idx_lower))
        pos_idx_cols = slice(None)

#        print("pos_idx_rows", pos_idx_rows)
#        print("pos_idx_cols", pos_idx_cols)

        return [[pos_idx_rows, pos_idx_cols]]
