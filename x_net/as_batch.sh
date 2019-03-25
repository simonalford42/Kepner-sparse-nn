#!/bin/sh
#To be run on the TX-E1
sbatch -p normal --job-name=tests.sh  -o log-%j -N 4 tests.sh
