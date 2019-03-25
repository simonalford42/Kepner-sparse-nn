sbatch -c 4 --job-name=gpu_test.sh -o "aug27/gpu_test" -p gpu --gres=gpu:volta:2 gpu.sh $1
