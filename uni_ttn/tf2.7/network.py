import numpy as np
import tensorflow as tf
import sys


def dephase(tensor):
    return tf.linalg.diag(tf.linalg.diag_part(tensor))


class Network:
    def __init__(self, num_pixels, bd_dim, config, deph_net):
        self.config = config
        self.bd_dim = bd_dim
        self.num_pixels = num_pixels
        self.num_layers = int(np.log2(num_pixels))
        self.init_mean = config['tree']['param']['init_mean']
        self.init_std = config['tree']['param']['init_std']
        self.deph_net = deph_net
        self.layers = []

        self.list_num_nodes = [int(self.num_pixels / 2 ** (i + 1)) for i in range(self.num_layers)]
        for i in range(self.num_layers):
            self.layers.append(
                Layer(bd_dim, self.list_num_nodes[i], i, self.init_mean, self.init_std)
            )

    def get_network_output(self, input_batch):
        layer_out = self.layers[0].get_layer_output(input_batch)
        if self.deph_net: layer_out = dephase(layer_out)
        for i in range(1, self.num_layers):
            layer_out = self.layers[i].get_layer_output(layer_out)
            if self.deph_net: layer_out = dephase(layer_out)
        return layer_out

    def train(self, input_batch, label_batch):
        pred_batch = self.get_network_output(input_batch)
        self.loss_config = self.config['tree']['loss']
        if self.loss_config == 'l2':
            self.loss = tf.reduce_sum(tf.square(pred_batch - label_batch)); print('L2 Loss')
        elif self.loss_config == 'l1':
            self.loss = tf.losses.absolute_difference(label_batch, pred_batch); print('L1 Loss')
        elif self.loss_config == 'log':
            self.loss = tf.losses.log_loss(label_batch, pred_batch); print('Log Loss')
        else:
            raise Exception('Invalid Loss')

        self.opt_config = self.config['tree']['opt']
        if self.opt_config['opt'] == 'adam':
            opt = tf.train.AdamOptimizer()
            self.grad_var = opt.compute_gradients(self.loss)
            self.train_op = opt.apply_gradients(self.grad_var)
            print('Adam Optimizer')
        elif self.opt_config['opt'] == 'sgd':
            step_size = self.opt_config['sgd']['step_size']
            self.train_op = tf.train.GradientDescentOptimizer(step_size).minimize(self.loss)
            print('SGD Optimizer')
        elif self.opt_config['opt'] == 'rmsprop':
            learning_rate = self.opt_config['rmsprop']['learning_rate']
            self.train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
            print('RMSProp Optimizer')
        else:
            raise Exception('Invalid Optimizer')

        sys.stdout.flush()


class Layer:
    def __init__(self, bd_dim, num_nodes, layer_idx, init_mean, init_std):
        self.bd_dim = bd_dim
        self.layer_idx = layer_idx
        self.num_diags = self.bd_dim ** 2
        self.num_op_params = self.num_diags ** 2 + self.num_diags
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
        diag_part = tf.linalg.diag(self.diag_params)
        off_diag_indices = [(i, j) for i in range(self.num_diags) for j in range(i + 1, self.num_diags)]
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
        unitary_tensor = tf.reshape(self.unitary_matrix, [self.num_nodes, *[self.bd_dim] * 4])
        return unitary_tensor

    def get_layer_output(self, input):
        left_input, right_input = input[::2], input[1::2]
        unitary_tensor = self.get_unitary_tensor()
        contracted = tf.einsum('nabcd, nzea, nzfb -> nzefcd', unitary_tensor, left_input, right_input)
        output = tf.einsum('nzefcd, nzefcg -> nzgd', contracted, tf.math.conj(unitary_tensor))
        return output
