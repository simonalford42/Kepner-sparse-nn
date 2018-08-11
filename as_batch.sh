#To be run on the TX-E1
rm -f fog* sparta.txt

sbatch -p normal -e sparta.txt --job-name=tests.sh -o fog-%j -N 4 tests.sh
