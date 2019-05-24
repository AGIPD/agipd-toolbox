#!/bin/zsh

#source /etc/profile.d/modules.sh
#source /gpfs/cfel/cxi/common/cfelsoft-rh7/setup.sh
#module load cfel-anaconda/py3-4.3.0

source /gpfs/cfel/cxi/common/public/cfelsoft-rh7-public/conda-setup.sh; conda activate base

SCRIPT_DIR=$1
shift

$SCRIPT_DIR/start_analyse.py $*
