#!/usr/bin/env bash

# this script starts mutiple processes in the background on one node

source /etc/profile.d/modules.sh
source /gpfs/cfel/cxi/common/cfelsoft-rh7/setup.sh
module load cfel-anaconda/py3-4.3.0

base_dir=
run_type=
input_dir=
output_dir=
n_processes=
module=
temperature=
current=
max_part=
column_spec=
nasics=
while test $# -gt 0
do
    case $1 in
        --script_base_dir)
            base_dir=$2
            shift
            ;;
        --run_type)
            run_type=$2
            shift
            ;;
        --measurement)
            measurement=$2
            shift
            ;;
        --input_dir)
            input_dir=$2
            shift
            ;;
        --output_dir)
            output_dir=$2
            shift
            ;;
        --n_processes)
            n_processes=$2
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
        --max_part)
            if [ "$2" != "false" ]
            then
                max_part=$2
            fi
            shift
            ;;
        --column_spec)
            if [ "$2" == "false" ]
            then
                shift
            else
                column_spec="$2 $3 $4 $5"
                shift 4
            fi
            ;;
        -h | --help ) usage
            exit
            ;;
        * ) break;  # end of options
    esac
    shift
done
nasics=$*

#TODO check for required parameters and stop if they are not set

script_dir=$base_dir/src

script_params="--type ${run_type} \
               --measurement ${measurement} \
               --input_dir ${input_dir} \
               --output_dir ${output_dir} \
               --n_processes ${n_processes} \
               --module ${module} \
               --temperature ${temperature} \
               --current ${current}"

printf "script_params: ${script_params}\n"
printf "max_part: ${max_part}\n"
printf "column_spec: ${column_spec}\n"
printf "nasics: ${nasics}\n"
printf "\n"

tmp=
for asic in $(echo ${nasics})
do
    printf "Starting script for asic ${asic}\n"

    if [ -z ${column_spec+x} ]
    then
        printf "column index list is set to '$column_spec'\n"
        script_params="${script_params} --column_spec ${column_spec}"
    fi

    if [ -z ${max_part+x} ]
    then
        script_params="${script_params} --max_part ${max_part}"
    fi

    /usr/bin/python3 ${script_dir}/analyse.py \
        ${script_params} --asic ${asic} &
    tmp+=( ${!} )
done

wait $tmp

## wait ${!}	# won't work' cause only last bg process would be covered - what
                # if last process ends earlier then previous ones -> script
                # ends and shus sbatch
