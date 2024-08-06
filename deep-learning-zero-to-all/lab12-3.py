import tensorflow as tf
import numpy as np

# Sample data
sample = " if you want you"
idx2char = list(set(sample))  # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> index

# Hyperparameters
dic_size = len(char2idx)  # RNN input size (one hot size)
rnn_hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
sequence_length = len(sample) - 1  # number of LSTM rollings (unit #)
lr = 0.1

# Preparing data
sample_idx = [char2idx[c] for c in sample]  # char to index
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) if you want yo
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n) f you want you

x_one_hot = tf.one_hot(x_data, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
y_one_hot = tf.one_hot(y_data, num_classes)

# Building the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(sequence_length, num_classes)))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=num_classes, activation='softmax')))

# Compiling the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(x_one_hot, y_one_hot, epochs=3000, verbose=2)

# Predicting
predictions = model.predict(x_one_hot)
predicted_indices = np.argmax(predictions, axis=2)

# Printing results
result_str = [idx2char[c] for c in np.squeeze(predicted_indices)]
print("Prediction:", ''.join(result_str))
