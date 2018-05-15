#!/bin/zsh

#source /etc/profile.d/modules.sh
#source /gpfs/cfel/cxi/common/cfelsoft-rh7/setup.sh
#module load cfel-anaconda/py3-4.3.0

source /etc/profile.d/modules.sh; source /gpfs/cfel/cxi/common/cfelsoft-rh7/setup.sh; module load cfel-python3/latest

SCRIPT_DIR=$1
shift

$SCRIPT_DIR/start_analyse.py $*
