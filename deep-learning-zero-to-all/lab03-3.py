import tensorflow as tf
import numpy as np

# Training data
X = np.array([1, 2, 3], dtype=np.float32)
Y = np.array([1, 2, 3], dtype=np.float32)

# Define a simple linear model using Keras Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], use_bias=False, kernel_initializer=tf.keras.initializers.Constant(5.0))
])

# Compile the model with Mean Squared Error loss and Stochastic Gradient Descent optimizer
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='mean_squared_error')

# Print model summary
model.summary()

# Train the model
model.fit(X, Y, epochs=101)

# Get the final weight
W_val = model.layers[0].get_weights()[0]
print("Learned Weight:", W_val)