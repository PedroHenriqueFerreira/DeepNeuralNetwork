import pickle as pkl

import numpy as np
from numpy.typing import NDArray

from os import listdir
from os.path import isfile

from PIL import Image
from time import time

from optimizers import *
from losses import *
from activations import *
from layers import *

from neural_network import NeuralNetwork


class GAN:
    def __init__(self, root_dir: str, output_dir: str, model_dir: str):
        self.noise_size = 100
        
        self.img_shape = (3, 32, 32)

        self.root_dir = root_dir
        self.output_dir = output_dir
        self.model_dir = model_dir

        self.noise = np.random.randn(5, self.noise_size)

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        self.discriminator.summary('Discriminator')
        self.generator.summary('Generator')

    def build_discriminator(self) -> NeuralNetwork:
        d_optimizer = AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        d_loss = CrossEntropyLoss()

        d = NeuralNetwork(d_optimizer, d_loss)

        d.add(Conv2D(64, (3, 3), stride=2, input_shape=self.img_shape))
        d.add(LeakyReLU())
        d.add(Dropout(0.3))

        d.add(Conv2D(128, (3, 3), stride=2))
        d.add(LeakyReLU())
        d.add(Dropout(0.3))

        d.add(Flatten())

        d.add(Dense(1))
        d.add(Sigmoid())

        return d

    def build_generator(self) -> NeuralNetwork:
        g_optimizer = AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        g_loss = MeanSquaredLoss()

        g = NeuralNetwork(g_optimizer, g_loss)

        g.add(Dense(128*8*8, input_shape=(self.noise_size,)))
        g.add(BatchNormalization())
        g.add(LeakyReLU())
        g.add(Reshape((128, 8, 8)))

        g.add(UpSampling2D())
        g.add(Conv2D(128, (3, 3)))
        g.add(BatchNormalization())
        g.add(LeakyReLU())

        g.add(UpSampling2D())
        g.add(Conv2D(64, (3, 3)))
        g.add(BatchNormalization())
        g.add(LeakyReLU())

        g.add(Conv2D(3, (3, 3)))

        g.add(TanH())

        return g

    def get_images(self) -> NDArray[np.float64]:
        images = []

        for file in listdir(self.root_dir):
            image = np.array(Image.open(f'{self.root_dir}/{file}').convert('RGB')).astype(np.float64)
            image = image.transpose((2, 0, 1))
            
            images.append((image - 127.5) / 127.5)

        return np.array(images)

    def save_images(self, images: NDArray[np.float64]) -> None:
        for i in range(images.shape[0]):
            image = (images[i] * 127.5 + 127.5).astype(np.uint8)
            image = image.transpose((1, 2, 0))
        
            Image.fromarray(image, mode='RGB').save(f'{self.output_dir}/{int(time())}{i}.png')

    def fit(
        self,
        batch_size: int = 64,
        epochs: int = 10000,
        save_interval: int = 5
    ) -> None:
        images = self.get_images()

        size = images.shape[0]

        gen_labels = np.ones((batch_size, 1))

        for epoch in range(epochs):
            print('=' * 75)

            print(f'Epoch: {epoch + 1} / {epochs}')

            print('Training discriminator...')

            real_images = images[np.random.randint(0, size, batch_size)]
            fake_images = self.generator.forward(
                np.random.randn(batch_size, self.noise_size), False)

            labels_real = np.random.uniform(0.95, 1, (batch_size, 1))
            labels_fake = np.random.uniform(0, 0.05, (batch_size, 1))

            d_loss_real = self.discriminator.train_on_batch(real_images, labels_real)
            d_loss_fake = self.discriminator.train_on_batch(fake_images, labels_fake)

            d_loss = d_loss_real + d_loss_fake

            print('Training generator...')

            gen_images = self.generator.forward(np.random.randn(batch_size, self.noise_size))

            g_loss, g_err = self.discriminator.not_train_on_batch(
                gen_images, gen_labels)

            self.generator.backward(g_err)
            self.generator.optimizer(self.generator.parameters, self.generator.gradients)

            print(f'Discriminator Loss: {d_loss} | Generator Loss: {g_loss}')

            if (epoch + 1) % save_interval == 0:
                print('Generating Images...')

                self.save_images(self.generator.forward(self.noise, False))

                print('Saving model...')

                with open(f'{self.model_dir}/GAN.pkl', 'wb') as f:
                    pkl.dump(self, f)


if isfile('./model/GAN.pkl'):
    gan = pkl.load(open('./model/GAN.pkl', 'rb'))
else:
    gan = GAN('./images', './generated', './model')

gan.fit()
