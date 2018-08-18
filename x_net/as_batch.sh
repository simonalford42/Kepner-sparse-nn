#!/bin/sh
#To be run on the TX-E1
rm -f log* error.txt

sbatch -p normal -e error.txt --job-name=tests.sh  -o log-%j -N 4 tests.sh
