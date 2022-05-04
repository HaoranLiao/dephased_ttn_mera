import tensorflow as tf
import numpy as np
import string, sys
sys.path.append('../../uni_ttn/tf2.7/')
import spsa
sys.path.append('../')
import network

class Network(network.Network):
    def __init__(self, num_pixels, deph_p, num_anc, config, tune_config):
        super().__init__(num_pixels, deph_p, num_anc, -1, -1, config)

        self.init_std = tune_config['init_std']

        if config['tree']['opt']['opt'] == 'adam':
            if not tune_config.get('tune_lr', False): self.opt = tf.keras.optimizers.Adam()
            else: self.opt = tf.keras.optimizers.Adam(tune_config['tune_lr'])
        elif config['tree']['opt']['opt'] == 'spsa':
            self.opt = spsa.Spsa(self, tune_config)
        else:
            raise NotImplementedError

    def update_no_processing(self, input_batch: np.ndarray, label_batch: np.ndarray):
        input_batch = tf.constant(input_batch, dtype=tf.complex64)
        label_batch = tf.constant(label_batch, dtype=tf.float32)
        with tf.GradientTape() as tape:
            loss = self.loss(input_batch, label_batch)
        grads = tape.gradient(loss, self.var_list)
        self.opt.apply_gradients(zip(grads, self.var_list))


class Ent_Layer(network.Ent_Layer):
    def __init__(self, num_nodes, layer_idx, num_anc, init_mean, init_std):
        super().__init__(num_nodes, layer_idx, num_anc, init_mean, init_std)

class Iso_Layer(network.Iso_Layer):
    def __init__(self, num_nodes, layer_idx, num_anc, init_mean, init_std):
        super().__init__(num_nodes, layer_idx, num_anc, init_mean, init_std)