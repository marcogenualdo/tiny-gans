import numpy as np
from time import time
import torch
from torch import nn
from torchvision import utils

import struct as st
from mnist.load_mnist import load_all

from matplotlib import pyplot as plt


# setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# building network
dim_latent_space = 2

class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()

		# encoding layers
		self.encoder_conv_ops = nn.Sequential(
		    nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=1),
		    nn.MaxPool2d(2, stride=2),
		    nn.ReLU(),
		    nn.Conv2d(in_channels=2, out_channels=4, kernel_size=4, stride=1),
		    nn.MaxPool2d(2, stride=2),
		    nn.ReLU(),
		    nn.Conv2d(in_channels=4, out_channels=8, kernel_size=2, stride=1),
		    nn.ReLU()
		)
		self.encoder_dense_ops = nn.Sequential(
			nn.Linear(8 * 2 * 2, dim_latent_space),
			nn.ReLU()
		)

		# decoding layers
		self.first_conv_size = 3
		self.decoder_dense_ops = nn.Sequential(
			nn.Linear(dim_latent_space, self.first_conv_size ** 2),
			nn.ReLU()
		)
		self.decoder_conv_ops = nn.Sequential(
		    nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=1),
		    nn.BatchNorm2d(4),
		    nn.ReLU(),
		    nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=5, stride=2),
		    nn.BatchNorm2d(2),
		    nn.ReLU(),
		    nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=4, stride=2),
		    nn.Sigmoid()
		)

    
	def forward (self, x):
	    batch_size = x.shape[0]

	    # encoding input
	    x = self.encoder_conv_ops(x)
	    #x = x.view(batch_size, -1)
	    #x = self.encoder_dense_ops(x)

	    # decoding input
	    #x = self.decoder_dense_ops(x)
	    #x = x.view(batch_size, 1, self.first_conv_size, self.first_conv_size)
	    x = self.decoder_conv_ops(x)
	    
	    return x


class Small_Autoencoder (nn.Module):
	def __init__ (self):
		super(Small_Autoencoder, self).__init__()

		self.encoding_ops = nn.Sequential(
			nn.Linear(784, 64),
			nn.ReLU(),
			nn.Linear(64, 16),
			nn.ReLU(),
			nn.Linear(16, dim_latent_space),
			nn.Tanh()
		)

		self.decoding_ops = nn.Sequential(
			nn.Linear(dim_latent_space, 16),
			nn.ReLU(),
			nn.Linear(16, 64),
			nn.ReLU(),
			nn.Linear(64, 784),
			nn.Sigmoid()
		)


	def forward (self, x):
		batch_size = x.shape[0]
		x = x.view(batch_size, -1)
		x = self.decoding_ops(self.encoding_ops(x))
		x = x.view(batch_size, 1, 28, 28)
		return x


# compiling models
autoencoder = Autoencoder().to(device)

loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(autoencoder.parameters(), lr=0.6, momentum=0.5)
#optimizer = torch.optim.Adam(autoencoder.parameters())


# loading dataset
train_imgs, train_lbls, test_imgs, test_lbls = load_all(10000)

train_imgs = torch.from_numpy(train_imgs.astype(np.float32)).to(device)
test_imgs = torch.from_numpy(test_imgs.astype(np.float32)).to(device)

train_imgs.resize_(train_imgs.shape[0], 1, 28, 28)
test_imgs.resize_(test_imgs.shape[0], 1, 28, 28)

print('Finished loading data.')


# ----- INITIALIZING PLOTS ----- #
loss_metric = []

fig = plt.figure()
loss_ax = fig.add_subplot(111)

plt.ion()
plt.show()

image_fig = plt.figure()
image_ax = image_fig.add_subplot(111)


# ----- TRAINING ----- #
batch_size = 64
epochs = 50

def make_batches(dataset):
    pin = 0
    length = dataset.shape[0]
    while batch_size * pin < length:
        yield dataset[batch_size * pin : min(batch_size * (pin + 1), length)].to(device)
        pin += 1


starting_time = time()
for epoch in range(epochs):
	# shuffling training data
	rand_index = torch.randperm(train_imgs.shape[0]).to(device)
	train_imgs = train_imgs[rand_index]

	for imgs_batch in make_batches(train_imgs):
	    samples_number = imgs_batch.shape[0]

	    # SGD
	    batch_loss = loss_func(imgs_batch, autoencoder(imgs_batch))
	    loss_metric.append(batch_loss.item())

	    autoencoder.zero_grad()
	    batch_loss.backward()
	    optimizer.step()

	# plotting epoch performance
	points = np.arange(len(loss_metric))

	loss_ax.clear()
	loss_ax.plot(points, loss_metric)
	loss_ax.clear()
	loss_ax.plot(points, loss_metric)

	plt.draw()
	plt.pause(1e-3)

	print(f'Epoch {epoch} completed in {time() - starting_time} seconds.')
	starting_time = time()

	# display a few reconstructed images compared to the originals
	n_examples = 5
	with torch.no_grad():
	    test_idx = torch.randint(0, len(test_imgs), (n_examples,))
	    test_batch = test_imgs[test_idx]
	    test_output = autoencoder(test_batch)

	    picture_array = torch.cat((test_batch, test_output), 0)
	    picture = utils.make_grid(picture_array, n_examples).transpose(0,1).transpose(1,2)
	image_ax.clear()
	image_ax.imshow(picture)
	plt.draw()
	plt.pause(0.5)