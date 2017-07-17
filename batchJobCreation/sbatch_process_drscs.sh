#!/usr/bin/env bash

mail_address=manuela.kuhn@desy.de

script_base_dir=/home/kuhnm/agipd-calibration
batch_job_dir=$script_base_dir/batchJobCreation
work_dir=$script_base_dir/sbatch_out

input_dir=/gpfs/cfel/fsds/labs/processed/calibration/processed
output_dir=/gpfs/cfel/fsds/labs/processed/calibration/processed
module=M314
temperature=temperature_m15C
current=itestc150
max_part=false
#column_spec="22 23 24 25"
column_spec=false

#asic_set1="16"
asic_set1="1 2 3 4 5 6 7 8"
asic_set2="9 10 11 12 13 14 15 16"

if [ ! -d "$work_dir" ]; then
  mkdir $work_dir
fi

call_sbatch()
{
    sbatch_params="--partition=all \
                   --time=00:30:00 \
                   --nodes=1 \
                   --mail-type END \
                   --mail-user ${mail_address} \
                   --workdir=${work_dir} \
                   --job-name=process_drscs_${module} \
                   --output=process_drscs_${module}-%j.out \
                   --error=process_drscs_${module}-%j.err "

    script_params="--input_dir ${input_dir} \
                   --output_dir ${output_dir} \
                   --module ${module} \
                   --temperature ${temperature} \
                   --current ${current}"

    sbatch ${sbatch_params} \
           ${batch_job_dir}/batchJob_gather_asics.sh ${script_params}\
                                                     $*
}

nasics=${asic_set1}
printf "Starting job for asics ${nasics}\n"
call_sbatch ${nasics}

nasics=${asic_set2}
printf "Starting job for asics ${nasics}\n"
call_sbatch ${nasics}
