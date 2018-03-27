import numpy as np
import keras

inputLayer = keras.layers.Input(shape=(784,))
compressionLayer = keras.layers.Dense(2, activation=keras.activations.linear)(inputLayer)
reconstructionLayer = keras.layers.Dense(784, activation=keras.activations.linear)(compressionLayer)

model = keras.models.Model(inputLayer, [reconstructionLayer, compressionLayer])

model.compile(optimizer=keras.optimizers.sgd(lr=0.1), loss=[keras.losses.mse, keras.losses.mse], loss_weights=[1,0])

x = np.concatenate((np.ones((20,784)), np.zeros((20, 784))), axis=0)

fakeY = np.zeros((40, 2))

model.fit(x, [x, fakeY], epochs=1000)

model.predict(np.array([np.ones(784,)]))

print("inputlayer : ", inputLayer)
print("x : ", x)