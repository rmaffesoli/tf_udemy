# Explore constant and operator nodes
# How to set up and use nodes
# How to run nodes to output their values

import tensorflow as tf

const_node_1 = tf.constant(1.0, dtype=tf.float32)
const_node_2 = tf.constant(2.0, dtype=tf.float32)
const_node_3 = tf.constant([3.0, 4.0, 5.0], dtype=tf.float32)

adder_node_1 = tf.add(const_node_1, const_node_2)
adder_node_2 = const_node_1 + const_node_2
mult_node_1 = adder_node_2 * const_node_3

session = tf.Session()
print(session.run(mult_node_1))



