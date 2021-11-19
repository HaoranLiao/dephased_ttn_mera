import numpy as np
import tensorflow as tf
import sys


class Network:
    def __init__(self, num_pixels, bd_dim, deph_p, config):
        self.config = config
        self.bd_dim = bd_dim
        self.num_pixels = num_pixels
        self.num_layers = int(np.log2(num_pixels))
        self.init_mean = config['tree']['param']['init_mean']
        self.init_std = config['tree']['param']['init_std']
        self.deph_data = config['meta']['deph']['data']
        self.deph_net = config['meta']['deph']['network']
        self.deph_p = deph_p
        self.layers = []

        self.list_num_nodes = [int(self.num_pixels / 2 ** (i + 1)) for i in range(self.num_layers)]
        for i in range(self.num_layers):
            self.layers.append(
                Layer(bd_dim, self.list_num_nodes[i], i, self.init_mean, self.init_std)
            )

        self.opt = tf.keras.optimizers.Adam()

    def get_network_output(self, input_batch):
        input_batch = tf.einsum('zna, znb -> znab', input_batch, input_batch)
        input_batch = tf.cast(input_batch, dtype=tf.complex128)
        if self.deph_data: input_batch = self.dephase(input_batch, self.deph_p)

        layer_out = self.layers[0].get_layer_output(input_batch)
        if self.deph_net: layer_out = self.dephase(layer_out, self.deph_p)
        for i in range(1, self.num_layers):
            layer_out = self.layers[i].get_layer_output(layer_out)
            if self.deph_net: layer_out = self.dephase(layer_out, self.deph_p)

        output_probs = tf.math.abs(tf.linalg.diag_part(tf.squeeze(layer_out)))
        return output_probs

    def update(self, input_batch, label_batch):
        self.input_batch = input_batch
        self.label_batch = label_batch
        self.opt.minimize(self.loss, var_list=[layer.param_var_lay for layer in self.layers])

    @tf.function
    def loss(self):
        pred_batch = self.get_network_output(self.input_batch)
        return tf.reduce_sum(tf.square(pred_batch - self.label_batch))

    def dephase(self, tensor, p):
        if self.bd_dim == 2: return (1 - p) * tensor + p * tf.linalg.diag(tf.linalg.diag_part(tensor))
        elif self.bd_dim == 4: NotImplemented
        elif self.bd_dim == 8: NotImplemented
        elif self.bd_dim == 16: NotImplemented


class Layer:
    def __init__(self, bd_dim, num_nodes, layer_idx, init_mean, init_std):
        self.bd_dim = bd_dim
        self.layer_idx = layer_idx
        self.num_diags = self.bd_dim ** 2
        self.num_op_params = self.num_diags ** 2
        self.num_nodes = num_nodes
        self.init_mean, self.std = init_mean, init_std

        self.param_var_lay = tf.Variable(
            tf.random_normal_initializer(mean=init_mean, stddev=init_std)(
                shape=[self.num_op_params, num_nodes], dtype=tf.float64,
            ), name='param_var_lay_%s' % layer_idx, trainable=True
        )

    def get_unitary_tensor(self):
        num_off_diags = int(0.5 * (self.num_diags ** 2 - self.num_diags))
        self.real_off_params = self.param_var_lay[:num_off_diags]
        self.imag_off_params = self.param_var_lay[num_off_diags:2 * num_off_diags]
        self.diag_params = self.param_var_lay[2 * num_off_diags:]

        herm_shape = (self.num_diags, self.num_diags, self.num_nodes)
        diag_part = tf.transpose(tf.linalg.diag(tf.transpose(self.diag_params)), perm=[1, 2, 0])
        off_diag_indices = [[i, j] for i in range(self.num_diags) for j in range(i + 1, self.num_diags)]
        real_off_diag_part = tf.scatter_nd(
            indices=off_diag_indices,
            updates=self.real_off_params,
            shape=herm_shape)
        imag_off_diag_part = tf.scatter_nd(
            indices=off_diag_indices,
            updates=self.imag_off_params,
            shape=herm_shape)
        imag_whole = imag_off_diag_part - tf.transpose(imag_off_diag_part, perm=[1, 0, 2])
        real_whole = diag_part + real_off_diag_part + tf.transpose(real_off_diag_part, perm=[1, 0, 2])
        self.herm_matrix = tf.transpose(tf.complex(real_whole, imag_whole), perm=[2, 0, 1])

        (eigenvalues, eigenvectors) = tf.linalg.eigh(self.herm_matrix)
        eig_exp = tf.exp(1.0j * eigenvalues)
        diag_exp_mat = tf.linalg.diag(eig_exp)
        self.unitary_matrix = tf.einsum('nab, nbc, ndc -> nad',
                                        eigenvectors, diag_exp_mat, tf.math.conj(eigenvectors))
        unitary_tensor = tf.reshape(self.unitary_matrix, [self.num_nodes, *[self.bd_dim]*4])
        return unitary_tensor

    def get_layer_output(self, input):
        left_input, right_input = input[:, ::2], input[:, 1::2]
        unitary_tensor = self.get_unitary_tensor()
        contracted = tf.einsum('nabcd, znea, znfb -> znefcd', unitary_tensor, left_input, right_input)
        output = tf.einsum('znefcd, nefcg -> zngd', contracted, tf.math.conj(unitary_tensor))
        return output
