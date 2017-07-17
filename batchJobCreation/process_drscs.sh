#!/usr/bin/env bash

# this script starts mutiple processes in the background on one node

base_path=/home/kuhnm/agipd-calibration/agipdCalibration/batchProcessing

input_dir=
output_dir=
module=
temperature=
current=
nasics=
while test $# -gt 0
do
    case $1 in
        --input_dir)
            input_dir=$2
            shift
            ;;
        --output_dir)
            output_dir=$2
            shift
            ;;
        --module)
            module=$2
            shift
            ;;
        --temperature)
            temperature=$2
            shift
            ;;
        --current)
            current=$2
            shift
            ;;
        -h | --help ) usage
            exit
            ;;
        * ) break;  # end of options
    esac
    shift
done
nasics=$*

script_params="--input_dir ${input_dir} \
               --output_dir ${output_dir} \
               --module ${module} \
               --temperature ${temperature} \
               --current ${current}"

printf "script_params: ${script_params}\n"
printf "nasics: ${nasics}\n"
printf "\n"

tmp=
for asic in $(echo ${nasics})
do
    printf "Starting script for asic ${asic}\n"

    /usr/bin/python ${base_path}/process_data_per_asic.py \
        ${script_params} --asic ${asic} &
    tmp+=( ${!} )

done

wait $tmp

## wait ${!}	# won't work' cause only last bg process would be covered - what
                # if last process ends earlier then previous ones -> script
                # ends and shus sbatch
