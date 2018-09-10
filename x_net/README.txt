This README explains all files in this x_net folder.

The x_net folder contains code which creates and tests the RadiX-Nets designed by Ryan Robinett.

The RadiX-Nets were implemented in both python and julia. Code implementing them is found in x_net.py, x_net.jl

Ryan also made a test script which compares results of the python and julia filest omake sure that they give the same output.
This is done by as_batch.sh, which simply submits to supercloud the tests.sh. 
Tests.sh calls run_tests.py and run_tests.jl for different input parameters and compares the outputs to make sure they're equal. 
Run_tests.py and run_tests.jl call x_net.py and x_net.jl using the test parameters given in the test/ folder.
The test/ folder contains two sub-directories, emr/ and kemr/, each of which contain .txt files, each of which contains the parameters for a radix-net. The emr.txt and kemr.txt files contain output from the scripts which are used to compare whether the results match for julia and python.

To run the tests, run the following commands:
module load anaconda3-5.0.1 # load python
module load beta-julia-0.6 # load julia
export JULIA_PKGDIR=/home/gridsan/salford/.julia/packages/ # change to your home dir
bash as_batch.sh

running as_batch.sh will create an output file called log-j, where j is the job number. It should give something like: Bool[true] six times in a row.


