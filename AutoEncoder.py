# AutoEncoder.py

# Libraries
# Third Party Libraries
import matplotlib.cm as cm	
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theano
import theano.tensor as T

# User defined Libraries
from ConvLayers import FC
from GradientDescent import *


X = T.matrix(name="input", dtype=theano.config.floatX)
lr = T.scalar(name='learning_rate', dtype=theano.config.floatX)
act_f = T.nnet.sigmoid

# Autoencoder
enc_layer_1 = FC(X, 784, 512, activation_fn=act_f)
enc_layer_2 = FC(enc_layer_1.output, 512, 256, activation_fn=act_f)
dec_layer_1 = FC(enc_layer_2.output, 256, 512, activation_fn=act_f)
out_layer = FC(dec_layer_1.output, 512, 784, activation_fn=act_f) 
params = out_layer.params + dec_layer_1.params \
	   + enc_layer_2.params + enc_layer_1.params
cost = 0.5 * T.mean((X-out_layer.output)**2)
grads = T.grad(cost, wrt=params)

# training functions
f = theano.function(inputs=[X, lr], outputs=cost, 
					updates=momentum(l_rate=lr, parameters=params, grads=grads),
					allow_input_downcast=True)
f_rebuilt = theano.function(inputs=[X], outputs=out_layer.output,
							allow_input_downcast=True)

def train(training_data, learning_rate=1e-1, batch_size=256, epochs=30):
	total_values = len(training_data)
	
	print('---Training Model---')

	for epoch in range(epochs):
		print('Currently on epoch {}'.format(epoch+1))

		np.random.shuffle(training_data)
		mini_batches = [training_data[k: k+batch_size]
					for k in range(0, total_values, batch_size)]

		training_cost = 0

		for mini_batch in mini_batches:
			cost_ij = f(mini_batch, learning_rate)
			training_cost += cost_ij

		print('The loss is: {}'.format(training_cost/batch_size))
		print('--------------------------')


def main():
	location = "E:\\Projects\\MNIST\\train.csv"
	data = pd.read_csv(location)
	data = data.sample(41984, random_state=35)

	images = data.iloc[:, 1:].values
	images = images.astype(theano.config.floatX)
	images = np.multiply(images, 1.0/255.0)

	train(images)

if __name__ == "__main__":
	main()
