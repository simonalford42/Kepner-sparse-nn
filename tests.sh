#!/bin/bash

rm -rf test
mkdir test

python3 run_tests.py
julia run_tests.jl
