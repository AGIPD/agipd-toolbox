#!/bin/zsh

#source /etc/profile.d/modules.sh
#source /gpfs/cfel/cxi/common/cfelsoft-rh7/setup.sh
#module load cfel-anaconda/py3-4.3.0

source /etc/profile.d/modules.sh; module load anaconda/3; module load texlive/2017; source /gpfs/cfel/cxi/common/cfelsoft-rh7/setup.sh; module load cfel-anaconda/py3-4.3.0


SCRIPT_DIR=$1
shift

$SCRIPT_DIR/start_analyse.py $*
