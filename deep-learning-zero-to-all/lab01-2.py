import tensorflow as tf

# build graph (tensors) using TensorFlow operations
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1, node2)

print("node1: ", node1) 
print("node2: ", node2)
print("node3: ", node3)

'''
result
node1: tf.Tensor(3.0, shape=(), dtype=float32)
node2: tf.Tensor(4.0, shape=(), dtype=float32)
node3: tf.Tensor(7.0, shape=(), dtype=float32)
'''

# Run the operations and get the results
print("node1, node2: ", node1.numpy(), node2.numpy())
print("node3: ", node3.numpy())

'''
result
node1, node2: [3.0, 4.0]
node3: 7.0
'''
