import numpy as np
import keras
from keras.metrics import *
import matplotlib.pyplot as plt

input = keras.layers.Input(shape=(784,))
encoded = keras.layers.Dense(32, activation='relu')(input)
decoded = keras.layers.Dense(784, activation='sigmoid')(encoded)

model = keras.models.Model(inputLayer, [decoded, encoded])

model.compile(optimizer=keras.optimizers.sgd(lr=0.1), loss=[keras.losses.mse, keras.losses.mse], loss_weights=[1, 0], metrics=[categorical_accuracy])

x = np.concatenate((np.ones((20, 784)), np.zeros((20, 784))), axis=0)
Y = np.zeros((40, 2))

history = model.fit(x, [x, Y], epochs=2000)

model.predict(np.array([np.ones(784,)]))

print(history.history['loss'])
plt.plot(history.history['loss'])
plt.show()

plt.bar(x, [categorical_accuracy], align='center')
plt.show()