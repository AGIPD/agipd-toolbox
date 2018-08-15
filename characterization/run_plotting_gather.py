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

import os

from plot_gathered import PlotGathered

if __name__ == "__main__":
    input_dir = "/gpfs/exfel/exp/SPB/201830/p900019/scratch/user/kuhnm/pcdrs/r0459-r0460-r0463-r0464-r0465-r0466-r0467-r0468/gather"
    #input_dir = "/gpfs/exfel/exp/SPB/201701/p002042/scratch/user/kuhnm/tmp/dark/r0257/gather"
    base_output_dir = "/gpfs/exfel/exp/SPB/201830/p900019/scratch/user/kuhnm/plots/"

    runs = [459, 460, 463, 464, 465, 466, 467, 468]
    channel = 0

    pixel = (30, 50)
    memcell = 3

    run_str = "R" + "-R".join(str(r).zfill(4) for r in runs)
#    run_str = "R{:04}".format(runs)

    input_fname = os.path.join(input_dir,
                               "{}-AGIPD{:02}-gathered.h5"
                               .format(run_str, channel))

    output_dir = os.path.join(base_output_dir, run_str)

    plotter = PlotGathered(input_fname=input_fname,
                           output_dir=output_dir,
                           row=pixel[0],
                           col=pixel[1],
                           memcell=memcell)

    plotter.plot()
