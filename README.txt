This README explains all of the contents of this directory, which contains the code developed by Simon Alford in the summer of 2018 researching under Dr. Jeremy Kepner.

The code used is for training and pruning sparse neural networks. It uses tensorflow.

There are the following directories:

caffe: contains caffe clone with some trained caffe models from earlier in the summer. Did one-time pruning on lenet5 and lenet300-100, but nothing in here is central to the work done.
caffe_pruning: contains more caffe stuff. Again, not central to the work done. However, to run the code, run the jupyter notebooks found in here.
data: contains datasets used to train on.
densenet: contains open-source implementation of densenet cloned from github
input-pipelines: contians open-source implementations of input pipelines for reading datasets to train with in tensorflow.
jupyter: contains jupyter notebooks, none of which are central to the work done.
old-saved: contains old tensorflow models saved.
pruning: contains the python code for training and pruning sparse neural networks.
saved: contains newer tensorflow models saved. All of the models used for the final data are here. It's structure is self explanatory based on the data collected.
x_net: contains code which creates and tests the radix-net structures used for sparse training.

There are also the following files:
- The powerpoint of my final presentaiton
- A csv file containing a spreadsheet where I sketch out the data used for the final data. The original google sheet can also be found here: https://docs.google.com/spreadsheets/d/1o4qe2Zsq7IIu9VsTJBD28gONPy962--OZnCbwwCNJes/edit?usp=sharing

The pruning code in the pruning directory originated from the open source pruning implementation from tensorflow/contrib/model_pruning in Tensorflow. However it has been modified / contributed to extensively. The examples folder contains the original cifar10 pruning example. The python folder contains other backend pruning code from the open source. The simon directory contains most of the new code. The current up-to-date version of the pruning is done using train.py and test.py. To see how the code works, I recommend starting by looking through train.py. An typical pruning run is done by running "train_test.sh (model_name)" in the command line. You need to load module anaconda2-5.0.1 or anaconda3-5.0.1 (I used anaconda2-5.0.1 but I believe it's compatible with python3).

The data I collected is all in the saved directory. More specifically, each model is in a directory. One can view the results of a model using tensorboard. This is what I found worked for viewing Tensorboard:
ssh -L 16006:127.0.0.1:6006 salford@txe1-login.mit.edu'
module load anaconda2-5.0.1
python -m tensorboard.main --logdir=./saved/model_dir

Then go to the following link in browser: http://127.0.0.1:16006/#scalars&_smoothingWeight=0&run=.

Note: many of the directories storing models have been moved around to restructure things. In doing so, the checkpoint file paths listed became out of date. This is only an issue if one of the models is being used to restore another model; in this case, you will have to go in and update the checkpoint file paths. This can also be done with the function provided in utils.py
