import tensorflow as tf

# build graph (tensors) using TensorFlow operations
@tf.function
def adder(a, b):
	return a+b

# feed data and run graph (operation)
# ipdate variables in the graph (and return values)
result1 = adder(tf.constant(3.0), tf.constant(4.5))
result2 = adder(tf.constant([1.0, 3.0]), tf.constant([2.0, 4.0]))

print(result1.numpy())
print(result2.numpy())

'''
result
7.5
[ 3. 7. ]
'''
