import tensorflow as tf

a = tf.Variable([[1, 2, 3] for _ in range(5)])
with tf.GradientTape() as tape:
    loss = tf.reduced_mean(a**2)
grads = tape.gradient(loss,