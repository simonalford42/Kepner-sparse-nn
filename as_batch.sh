./clean.sh

sbatch -p normal -e error.txt --job-name=tests.sh  -o log-%j tests.sh
