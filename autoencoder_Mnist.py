import keras
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

hidden_layer = 32

input = keras.layers.Input(shape=(784,))
encod = keras.layers.Dense(hidden_layer, activation='relu')(input)
decod = keras.layers.Dense(784, activation='sigmoid')(encod)

autoencoder = keras.models.Model(input, decod)
encoder = keras.models.Model(input, encod)

encoded_input = keras.layers.Input(shape=(hidden_layer,))
decoder_layer = autoencoder.layers[-1]

decoder = keras.models.Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.bar(n, 1 - autoencoder.losses)
plt.show()