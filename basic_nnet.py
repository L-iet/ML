#We want to approximate output = sum([x mod position for x in input])
import random

def sigmoid(x):
	return 1/(1 + 2.718281828**(-x))

def forward_pass(layer_weights, layer_biases, inp):
	layer_out = []
	for indx, n in enumerate(layer_weights):
		s = 0
		for ind, w in enumerate(n):
			s += w * inp[ind]
		s += layer_biases[indx]
		layer_out.append(s)
	layer_out = [sigmoid(x) for x in layer_out]
	return layer_out

def error(result, ideal):
	return sum([(i - r)**2 for i,r in zip(ideal,result)])

inp = random.choices(range(10),k=8)
output = sum([x % (inp.index(x)+1) for x in inp])
act_out = [(0 if i != output-1 else 1) for i in range(22)]

hl1 = [[random.uniform(-2,2) for j in range(8)] for i in range(10)]
bs1 = [random.uniform(-2,2) for i in range(10)]

oul = [[random.uniform(-2,2) for j in range(10)] for i in range(22)]
obs = [random.uniform(-2,2) for i in range(22)]

hl1_out = forward_pass(hl1, bs1, inp)
out_out = forward_pass(oul, obs, hl1_out)







