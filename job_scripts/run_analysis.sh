#!/bin/zsh

##### modify this only
git_dir=/home/kuhnm/calibration

#raw_dir=/gpfs/exfel/exp/SPB/201730/p900009
raw_dir=/gpfs/exfel/exp/SPB/201730/p900009/scratch/p002017_raw
output_dir=/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm
#channel="0"
channel="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"

meas_type=dark
# run numbers contained in run list have to have 4 digits
run_list="0004 0005 0006"
#run_list="0428 0429 0430"

#meas_type=pcdrs
#run_list="0488 0489 0490 0491 0492 0493 0494 0495"

#####

source /etc/profile.d/modules.sh; module load anaconda/3; module load texlive/2017; source /gpfs/cfel/cxi/common/cfelsoft-rh7/setup.sh; module load cfel-python3/latest
#source /etc/profile.d/modules.sh; module load anaconda/3; module load texlive/2017; source /gpfs/cfel/cxi/common/cfelsoft-rh7/setup.sh; module load cfel-anaconda/py3-4.3.0

joined_runs="r${run_list// /-r}"
echo $joined_runs

joined_dir=$output_dir/$meas_type/$joined_runs
#joined_dir=$output_dir/$meas_type

# https://hpc.nih.gov/docs/job_dependencies.html
# first job - no dependencies
#jid1=$(sbatch  job1.sh)
# jobs can depend on a single job
#jid2=$(sbatch  --dependency=afterany:$jid1 job2.sh)
# a single job can depend on multiple jobs
#jid4=$(sbatch  --dependency=afterany:$jid2:$jid3 job4.sh)

for ch in ${channel}
do
    printf "\nWorking on channel $ch\n"
    if [ "$meas_type" == "pcdrs" ]
    then
        ${git_dir}/job_scripts/start_analyse.py --input_dir $raw_dir \
                                                --output_dir $output_dir \
                                                --channel $ch \
                                                --run_type gather \
                                                --type $meas_type \
                                                --run_list ${run_list} \
                                                --use_xfel_in_format

    elif  [ "$meas_type" == "dark" ]
    then
        for run in $run_list
        do
            printf "run $run\n"
            ${git_dir}/job_scripts/start_analyse.py --input_dir $raw_dir \
                                                    --output_dir $output_dir \
                                                    --channel $ch \
                                                    --run_type gather \
                                                    --type $meas_type \
                                                    --run_list $run \
                                                    --use_xfel_in_format
        done

    fi

    # generates constants in the agipd format
    ${git_dir}/job_scripts/start_analyse.py --input_dir $output_dir \
                                            --output_dir $output_dir \
                                            --channel $ch \
                                            --run_type process \
                                            --type $meas_type \
                                            --run_list ${run_list} \
                                            --use_xfel_in_format

    # generates constants in the xfel format
    ${git_dir}/src/convert_format.py --base_dir ${joined_dir}/process \
                                     --input_file ${meas_type}_AGIPD{:02d}_agipd.h5 \
                                     --output_file ${meas_type}_AGIPD{:02d}_{}.h5 \
                                     --channel ${ch} \
                                     --output_format xfel
#    ${git_dir}/job_scripts/start_analyse.py --input_dir $output_dir --output_dir $output_dir --channel $ch --run_type process --type $meas_type --run_list ${run_list} --use_xfel_in_format --use_xfel_out_format
done


# joins constants in the agipd format
${git_dir}/src/join_constants.py --input_dir $joined_dir/process \
                                 --input_file ${meas_type}_AGIPD{:02d}_agipd.h5 \
                                 --output_dir $joined_dir \
                                 --output_file ${meas_type}_joined_constants_agipd.h5

# joins constants in the xfel format
${git_dir}/src/join_constants.py --input_dir $joined_dir/process \
                                 --input_file ${meas_type}_AGIPD{:02d}_xfel.h5 \
                                 --output_dir $joined_dir \
                                 --output_file ${meas_type}_joined_constants_xfel.h5
