import numpy as np
from x_net import *
from sys import argv

Ws = []
datas = []
with open("test/emr/emr"+argv[1]+".txt", "r") as f:
	for line in f:
		W, data = extended_mixed_radix_network(eval(line[:-1]))
		Ws.append(W)
		datas.append(data)

for W in Ws:
	W = [w.astype(int).tolist() for w in W]
	with open("test/emr.txt", "w") as f:
		f.write(str(W)+"\n")

Ws = []
datas = []
with open("test/kemr/kemr"+argv[1]+".txt", "r") as f:
	for line in f:
		toople = eval(line[:-1])
		W, data = kronecker_emr_network(toople[0], toople[1])
		Ws.append(W)
		datas.append(data)

for W in Ws:
	WW = [w.astype(int).tolist() for w in W]
	with open("test/kemr.txt", "w") as f:
		f.write(str(WW)+"\n")
