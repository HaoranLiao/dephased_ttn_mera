import tensorflow as tf
import data

a = tf.Variable(9.0)
input = tf.cast(tf.reshape(list(range(100)), [10, -1]), dtype=tf.float32)
with tf.GradientTape() as tape:
    loss = tf.math.reduce_mean(input * a)
grads = tape.gradient(loss, a)
print(grads)

a = tf.Variable(9.0)
input_1 = tf.cast(tf.reshape(list(range(100)), [10, -1]), dtype=tf.float32)[:5]
input_2 = tf.cast(tf.reshape(list(range(100)), [10, -1]), dtype=tf.float32)[5:]
with tf.GradientTape() as tape:
    loss = tf.math.reduce_mean(input_1 * a)
grads_1 = tape.gradient(loss, a)
with tf.GradientTape() as tape:
    loss = tf.math.reduce_mean(input_2 * a)
grads_2 = tape.gradient(loss, a)
grads = tf.reduce_mean([grads_1, grads_2])
print(grads)


a = tf.Variable(9.0)
self_grads = None

def update(input, apply_grads=True, counter=1):
    global self_grads

    with tf.GradientTape() as tape:
        loss = tf.math.reduce_mean(input * a)
    grads = tape.gradient(loss, a)
    if not self_grads:
        self_grads = grads
    else:
        self_grads = self_grads + grads

    if apply_grads:
        if counter > 1:
            self_grads = self_grads / counter
        print(self_grads)
        self_grads = None

def run_epoch(input, batch_size=10):

    exec_batch_size = 5
    counter = batch_size // exec_batch_size
    assert not batch_size % exec_batch_size, 'batch_size not divisible by exec_batch_size'
    batch_iter = data.batch_generator_tf(input, input, exec_batch_size)
    for (train_image_batch, train_label_batch) in batch_iter:
        if counter > 1:
            counter -= 1
            update(train_image_batch, apply_grads=False)
        else:
            counter = batch_size // exec_batch_size
            update(train_image_batch, apply_grads=True, counter=counter)

input = tf.cast(tf.reshape(list(range(100)), [10, -1]), dtype=tf.float32)
run_epoch(input)