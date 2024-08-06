import tensorflow as tf
import numpy as np
import time

x_data = np.array([[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]])

y_data = np.array([[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]])

# Evaluation our model using this test dataset
x_test = np.array([[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]])

y_test = np.array([[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]])

tf.model = tf.keras.Sequential([
    tf.keras.Input(shape=(3, )),
    tf.keras.layers.Dense(units=3, activation='softmax')
])

lr = [65535.0, 0.1, 1e-10]
for i in lr:
    tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=i), metrics=['accuracy'])

    tf.model.fit(x_data, y_data, epochs=1000)

    print(f'learning rate: {i}')

    # predict
    print("Prediction: ", tf.model.predict(x_test))

    # Calculate the accuracy
    print("Accuracy: ", tf.model.evaluate(x_test, y_test)[1])

    time.sleep(3)