import tensorflow as tf

# Create a constant op
# This op is added as a node to the default graph
hello = tf.constant("Hello, TensorFlow!")

# Run the op and get result
print(hello.numpy())

'''
result
b'Hello, TensorFlow!'
'''
