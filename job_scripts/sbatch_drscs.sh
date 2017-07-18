#!/usr/bin/env bash

mail_address=manuela.kuhn@desy.de

script_base_dir=/home/kuhnm/agipd-calibration
batch_job_dir=$script_base_dir/job_scripts

run_type=gather
#run_type=process

module=M314
temperature=temperature_m15C
current=itestc20

if [ "$run_type" == "gather" ]
then
    input_dir=/gpfs/cfel/fsds/labs/calibration/current/302-303-314-305
else
    input_dir=/gpfs/cfel/fsds/labs/processed/calibration/processed
fi

output_dir=/gpfs/cfel/fsds/labs/processed/calibration/processed

### Needed for gather ###
max_part=false
#column_spec="22 23 24 25"
column_spec=false

#asic_set1="1"
asic_set1="1 2 3 4 5 6 7 8"
asic_set2="9 10 11 12 13 14 15 16"

work_dir=$output_dir/$module/$temperature/sbatch_out
if [ ! -d "$work_dir" ]; then
    printf "Creating sbatch working dir: ${work_dir}\n"
    mkdir $work_dir
fi

# getting date and time
dt=$(date '+%Y-%m-%d_%H:%M:%S')

call_sbatch()
{
    sbatch_params="--partition=all \
                   --time=00:30:00 \
                   --nodes=1 \
                   --mail-type END \
                   --mail-user ${mail_address} \
                   --workdir=${work_dir} \
                   --job-name=${run_type}_drscs_${module} \
                   --output=${run_type}_drscs_${module}_$dt_%j.out \
                   --error=${run_type}_drscs_${module}_$dt_%j.err "

    script_params="--script_base_dir ${script_base_dir} \
                   --run_type ${run_type} \
                   --input_dir ${input_dir} \
                   --output_dir ${output_dir} \
                   --module ${module} \
                   --temperature ${temperature} \
                   --current ${current}"

    if [ "$run_type" == "gather" ]
    then
        script_params="${script_params} \
                       --max_part ${max_part} \
                       --column_spec ${column_spec}"
    fi

    sbatch ${sbatch_params} \
           ${batch_job_dir}/drscs.sh ${script_params} $*
}

nasics=${asic_set1}
printf "Starting job for asics ${nasics}\n"
call_sbatch ${nasics}

nasics=${asic_set2}
printf "Starting job for asics ${nasics}\n"
call_sbatch ${nasics}
