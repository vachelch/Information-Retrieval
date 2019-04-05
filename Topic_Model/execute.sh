#!/bin/bash
# Put your command below to execute your program.
# Replace "./my-program" with the command that can execute your program.
# Remember to preserve " $@" at the end, which will be the program options we give you.

best=0

while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -d)
    input_file="$2"
    shift # past argument
    ;;
    -o)
    output_file="$2"
    shift # past argument
    ;;
    -g)
    group_file="$2"
    shift # past argument
    ;;
    -e)
    best=1
    ;;
    -b)
    best=1
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done


if [ best=1 ]; then
    python3 predict.py --seed --input_file=${input_file} --output_file=${output_file} --group_file=${group_file}
else
    python3 predict.py --input_file=${input_file} --output_file=${output_file} --group_file=${group_file}
fi

# ./my-program $@













