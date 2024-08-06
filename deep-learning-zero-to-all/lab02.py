import tensorflow as tf
import numpy as np

# X and Y data
x_train = np.array([1, 2, 3, 4])
y_train = np.array([0, -1, -2, -3])

tf.model = tf.keras.Sequential([
    # units == output shape
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)
])

sgd = tf.keras.optimizers.SGD(learning_rate = 0.1) # SGD == Standard Gradient Descendent
tf.model.compile(loss = 'mse', optimizer = sgd) # mse == mean_squared_error, 1/m * sig (y' - y)^2

#prints summary of the model to the terminal
tf.model.summary()

# fit() executes training
tf.model.fit(x_train, y_train, epochs = 20)

# predict() returns predicted value
y_predict = tf.model.predict(np.array([5, 4]))
print(y_predict)
