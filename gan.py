import torch
from torch import nn
from torchvision import utils
from mnist.load_mnist import load_all

import numpy as np
from matplotlib import pyplot as plt
from time import time


# setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu') # <- current gpu hasn't got enough memory

# importing dataset
train_imgs, train_lbls, test_imgs, test_lbls = load_all(load_up_to='all')
train_imgs = torch.from_numpy(train_imgs.astype(np.float32)).to(device)
input_size = train_imgs[0].shape[0]


# ----- BUILDING NETWORKS ----- #
categorical_loss = nn.BCELoss()

# generator
noise_size = 100

generator = nn.Sequential(
    nn.Linear(noise_size, 128),
    nn.ReLU(),
    nn.Linear(128, input_size),
    nn.Sigmoid()
).to(device)

#gOptimizer = torch.optim.SGD(generator.parameters(), lr=1e-2, momentum=0.5)
gOptimizer = torch.optim.Adam(generator.parameters())
gLoss = lambda x: categorical_loss(x, torch.zeros(x.shape[0]).to(device)) # = log(1-x)

# discriminator
judge = nn.Sequential(
    nn.Linear(input_size, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
).to(device)

#jOptimizer = torch.optim.SGD(judge.parameters(), lr=1e-2, momentum=0.5)
jOptimizer = torch.optim.Adam(judge.parameters())
jLoss = lambda x: categorical_loss(x, torch.ones(x.shape[0]).to(device)) # = log(x)


# ----- INITIALIZING PLOTS ----- #
jLoss_metric = []
gLoss_metric = []

fig = plt.figure()
jLoss_ax = fig.add_subplot(211)
gLoss_ax = fig.add_subplot(212)

plt.ion()
plt.show()

image_fig = plt.figure()
image_ax = image_fig.add_subplot(111)


# ----- TRAINING ----- #
batch_size = 64
judge_reps = 1
epochs = 50

def make_batches(dataset):
    pin = 0
    length = dataset.shape[0]
    while batch_size * pin < length:
        yield dataset[batch_size * pin : min(batch_size * (pin + 1), length)]
        pin += 1


starting_time = time()
for epoch in range(epochs):
    # shuffling training data
    rand_index = torch.randperm(train_imgs.shape[0]).to(device)
    train_imgs = train_imgs[rand_index]

    for imgs_batch in make_batches(train_imgs):
        samples_number = imgs_batch.shape[0]

        # training the discriminator
        for k in range(judge_reps):
            reals = judge(imgs_batch).squeeze()
            noise = torch.randn(samples_number, noise_size).to(device)
            fakes = judge(generator(noise)).squeeze()
            
            loss = jLoss(reals) + gLoss(fakes)
            jLoss_metric.append(loss.item())

            judge.zero_grad()
            loss.backward()
            jOptimizer.step()

        # training the generator
        noise = torch.randn(samples_number, noise_size).to(device)
        beliefs = judge(generator(noise)).squeeze()

        loss = jLoss(beliefs)
        gLoss_metric.append(loss.item())

        generator.zero_grad()
        loss.backward()
        gOptimizer.step()

    # plotting epoch performance
    jpoints = np.arange(len(jLoss_metric))
    gpoints = np.arange(len(gLoss_metric))

    jLoss_ax.clear()
    jLoss_ax.plot(jpoints, jLoss_metric)
    gLoss_ax.clear()
    gLoss_ax.plot(gpoints, gLoss_metric)

    plt.draw()
    plt.pause(1e-3)

    # generate and display a few images
    nResults = 5
    with torch.no_grad():
        noise = torch.randn(nResults ** 2, noise_size).to(device)
        results = generator(noise).view(-1, 1, 28, 28)
        image = utils.make_grid(results, nResults).transpose(0,1).transpose(1,2)

        image_ax.clear()
        image_ax.imshow(image)
        plt.draw()
        plt.pause(0.5)

    print(f'Epoch {epoch} completed in {time() - starting_time} seconds.')
    starting_time = time()
