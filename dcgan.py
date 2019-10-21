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
train_imgs, train_lbls, test_imgs, test_lbls = load_all(load_up_to=10000)
train_imgs = torch.from_numpy(train_imgs.astype(np.float32)).to(device)
input_size = train_imgs[0].shape[0]


# ----- BUILDING NETWORKS ----- #
categorical_loss = nn.BCELoss()

# generator
noise_size = 100

class Generator (nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.dense_output_width = 8

        self.dense_ops = nn.Sequential(
            nn.Linear(noise_size, self.dense_output_width ** 2),
            nn.ReLU()
        )
        self.conv_ops = nn.Sequential(
            nn.ConvTranspose2d(1, 16, 4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, 2),
            nn.Sigmoid()
        )

    
    def forward (self, x):
        batch_size = x.shape[0]

        x = x.view(batch_size, -1)
        x = self.dense_ops(x)
        x = x.view(batch_size, 1, self.dense_output_width, self.dense_output_width)
        x = self.conv_ops(x)
        
        return x
    

class Discriminator (nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_ops = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 8, 3, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 4, 3),
            nn.LeakyReLU(0.2)
        )
        self.dense_ops = nn.Sequential(
            nn.Linear(4 ** 3, 1),
            nn.Sigmoid()
        )

    
    def forward (self, x):
        batch_size = x.shape[0]

        x = x.view(batch_size, 1, 28, 28)
        x = self.conv_ops(x)
        x = x.view(batch_size, -1)
        x = self.dense_ops(x)
        
        return x.squeeze()


# ----- COMPILING MODELS ----- #
generator = Generator().to(device)

#gOptimizer = torch.optim.SGD(generator.parameters(), lr=1e-2, momentum=0.5)
gOptimizer = torch.optim.Adam(generator.parameters())
gLoss = lambda x: categorical_loss(x, torch.zeros(x.shape[0]).to(device)) # = log(1-x)

# discriminator
judge = Discriminator().to(device)

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
            reals = judge(imgs_batch.unsqueeze(1))
            noise = torch.randn(samples_number, noise_size).to(device)
            fakes = judge(generator(noise))
            
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
