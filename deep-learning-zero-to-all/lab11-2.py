import tensorflow as tf
import random
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# One hot encode y data
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Hyperparameters
lr = 0.001
training_epochs = 15
batch_size = 100
dropout_rate = 0.7

# Build the model
model = Sequential([
		# kernel size = filter size
		# Conv2D(num of filters, filter size, activation function, input shape)
		# In above, the default activation function is glorot uniform
		# input shape (28, 28, 1) means 28 x 28 image with 1 channel
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(dropout_rate),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(dropout_rate),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(dropout_rate),

    Flatten(), # multi dim -> 1 dim

		# kernel initializer: 가중치 초기화
		# glorot normal = Xavier normal
    Dense(625, activation='relu', kernel_initializer='glorot_normal'),
    Dropout(dropout_rate),

    Dense(10, activation='softmax', kernel_initializer='glorot_normal')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=training_epochs, batch_size=batch_size)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy:.4f}')

# Predict a random example from the test set
r = random.randint(0, x_test.shape[0] - 1)
prediction = model.predict(np.expand_dims(x_test[r], axis=0))
print(f'Label: {np.argmax(y_test[r])}')
print(f'Prediction: {np.argmax(prediction)}')
