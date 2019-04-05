#!/bin/bash

feedback=0
best=0

while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -i)
    input_file="$2"
    shift 
    ;;
    -o)
    output_file="$2"
    shift 
    ;;
    -m)
    model_dir="$2"
    shift 
    ;;
    -d)
    NTCIR_dir="$2"
    shift 
    ;;
    -r)
    feedback=1
    ;;
    -b)
    best=1
    ;;
    *)
    ;;
esac
shift 
done

python3 build_tf_idf.py --model_dir=${model_dir}

if [ feedback=1 ]; then
    python3 predict.py --feedback --input_file=${input_file} --output_file=${output_file} --model_dir=${model_dir}
elif [ best=1 ];then
    python3 predict.py --input_file=${input_file} --output_file=${output_file} --model_dir=${model_dir}
else
    python3 predict.py --input_file=${input_file} --output_file=${output_file} --model_dir=${model_dir}
fi














