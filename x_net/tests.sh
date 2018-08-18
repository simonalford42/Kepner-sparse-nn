#!/bin/bash

rm -f test/emr.txt test/kemr.txt

nums=(1 2 3)
for j in ${nums[@]}
do
	python3 run_tests.py $j
	julia run_tests.jl $j
    echo j
    echo done
done
