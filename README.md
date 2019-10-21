# tiny-gans
Training very small Generative Adversarial Networks on the MNIST dataset. Built using PyTorch.

## Dense Networks - gan.py
Although I never trained the model for more than a dozen epochs, it shows some very promising resluts even when the training loop is run for two or three minutes on a laptop.
An example of digits generated from random samples in the latent space is shown below.

![Dense GAN samples - Epoch 12](/images/tiny_gan_results.png)

## Convolutional Networks - dcgan.py
Here I implement the classic improvement over standard GANs from [this](https://arxiv.org/abs/1511.06434) famous paper, featuring deep convolutional networks with batch normalization.
Digits generated using this technique are evidently crisper, although since I haven't spent time cross-validating hyperparameters it took more epochs to get readable results.

![DCGAN samples - Epoch 42](/images/dcgan_results_epoch_42.png)
