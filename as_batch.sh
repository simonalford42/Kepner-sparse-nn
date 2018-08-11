#To be run on the TX-E1
rm -f fog* sparta.txt

sbatch -p normal -e error.txt --job-name=tests.sh  -o log-%j -N 4 tests.sh
