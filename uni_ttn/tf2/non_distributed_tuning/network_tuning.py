import tensorflow as tf
import numpy as np
from uni_ttn.tf2 import network


class Network(network.Network):
    def __init__(self, num_pixels, deph_p, num_anc, config, tune_config):
        super().__init__(num_pixels, deph_p, num_anc, tune_config['tune_init_std'], tune_config['tune_lr'], config)

    def update_no_processing(self, input_batch: np.ndarray, label_batch: np.ndarray):
        input_batch = tf.constant(input_batch, dtype=tf.complex64)
        label_batch = tf.constant(label_batch, dtype=tf.float32)
        with tf.GradientTape() as tape:
            loss = self.loss(input_batch, label_batch)
        grads = tape.gradient(loss, self.var_list)
        self.opt.apply_gradients(zip(grads, self.var_list))

