import numpy as np
from x_net import *
from sys import argv

# Read in EMR argument and evaluate using Python implementation
Ws = []
datas = []
print('testing')
with open("test/emr/emr"+argv[1]+".txt", "r") as f:
	for line in f:
		W, data = emr_net(eval(line[:-1]))
		Ws.append(W)
		datas.append(data)
print('done')

# Save Python implementation to .txt file
for W in Ws:
	W = [w.astype(int).tolist() for w in W]
	with open("test/emr.txt", "w") as f:
		f.write(str(W)+"\n")

# Read in KEMR arguments and evaluate using Python implementation
Ws = []
datas = []
print('testing 2')
with open("test/kemr/kemr"+argv[1]+".txt", "r") as f:
	for line in f:
		toople = eval(line[:-1])
		W, data = kemr_net(toople[0], toople[1])
		Ws.append(W)
		datas.append(data)
print('done 2')

# Save Python implementation to .txt file
for W in Ws:
	WW = [w.astype(int).tolist() for w in W]
	with open("test/kemr.txt", "w") as f:
		f.write(str(WW)+"\n")
