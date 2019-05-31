import tensorflow as tf

placeholder_1 = tf.placeholder(dtype=tf.float32)
placeholder_2 = tf.placeholder(dtype=tf.float32)

mult_node_1 = placeholder_1 * 3.0
mult_node_2 = placeholder_1 * placeholder_2

# shape equals the number of dimensions present
session = tf.Session()
print(session.run(mult_node_2, {placeholder_1: 4.0, placeholder_2: [2.0, 5.0]}))
