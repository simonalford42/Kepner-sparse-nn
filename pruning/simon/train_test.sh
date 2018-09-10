#!/bin/sh
sbatch -c 1 --job-name=$1_test.sh -o "aug27/$1_test" test.sh $1
#sbatch -c 4 --job-name=$1_train.sh -o "aug27/$1_train" -p gpu --gres=gpu:volta:2 train.sh $1
sbatch -c 4 --job-name=$1_train.sh -o "aug27/$1_train" train.sh $1
