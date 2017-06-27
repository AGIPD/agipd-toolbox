from __future__ import print_function

import os
from gatherdata import GatherData

if __name__ == "__main__":
    rel_file_path = "311-312-301-300-310-234/temperature_m20C/drscs/itestc150"

    module = "M234_m8"
    current = "itestc150"

    # [[<column>, <file index>],...]
    # e.g. for a file name of the form M234_m8_drscs_itestc150_col15_00001_part00000.nxs
    # the entry would be                                         [15,   1]
    column_specs = [[15, 1], [26, 2], [37, 3], [48, 4]]

    #max_part = False
    max_part = 10

    output_file_path = "/gpfs/cfel/fsds/labs/processed/kuhnm/"

    ##############################
    # DO NOT MODIFY FROM HERE ON #
    ##############################

    file_base_name = "{}_drscs_{}".format(module, current)
    #file_base_name = "M301_m3_drscs_itestc150"

    output_file_name = "{}_chunked.h5".format(file_base_name)
    output_file = os.path.join(output_file_path, output_file_name)

    GatherData(rel_file_path, file_base_name, output_file, column_specs, max_part)
