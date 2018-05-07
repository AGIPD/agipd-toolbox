import numpy as np
import h5py

fname = "test.h5"

entries = [53732137, 53732138, 53732139, 53732140, 53732141, 53732142, 53732143, 53732144, 53732145, 53732146, 53732147, 53732148, 53732149, 53732150, 53732151, 53732152, 53732153, 53732154, 53732155, 53732156, 53732157, 53732158]
data = np.empty(150)
data[-len(entries):] = entries

entry_to_test = "INDEX/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/image/count"
count_path = "INSTRUMENT/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/header/pulseCount"
trainid_path = "INDEX/trainId"

with h5py.File(fname, "w") as f:
    f[entry_to_test] = 1
    f[count_path] = 1
    f[trainid_path] = data#.astype(np.uint64)
