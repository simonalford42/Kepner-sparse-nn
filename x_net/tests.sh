#!/bin/bash

rm -f test/emr.txt test/kemr.txt

nums=(1 2 3)
for j in ${nums[@]}
do
	python3 run_tests.py $j
    echo python $j done
	julia run_tests.jl $j
    echo julia $j done
done
