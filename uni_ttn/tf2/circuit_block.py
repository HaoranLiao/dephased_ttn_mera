import tensorflow as tf
import numpy as np
from uni_ttn.tf2.network import Layer

Identity = tf.constant([[1, 0], [0, 1]], dtype=tf.complex64)
Hadamard = 1/np.sqrt(2) * tf.constant([[1, 1], [1, -1]], dtype=tf.complex64)

class RX:
    def __init__(self, init_mean=0, init_std=1):
        self.param = tf.cast(tf.Variable(
            0.5 * tf.random_normal_initializer(mean=init_mean, stddev=init_std)(shape=(), dtype=tf.float32),
            trainable=True), tf.complex64)
        self.Rx = tf.stack([tf.concat([tf.cos(self.param), -1.0j * tf.sin(self.param)], axis=0),
                            -1.0j * tf.concat([tf.cos(self.param), tf.sin(self.param)], axis=0)])

    def get_matrix(self, which_qb, total_num_qbs):
        return self.single_qubit_op(self.Rx, which_qb, total_num_qbs)

    @staticmethod
    def single_qubit_op(op, which_qb, total_num_qbs):
        assert -1 < which_qb < total_num_qbs and op.shape == (2, 2)
        if which_qb == 0:
            out = op
            for _ in range(total_num_qbs-1): out = tf.experimental.numpy.kron(out, Identity)
        elif which_qb == total_num_qbs-1:
            out = 1
            for _ in range(total_num_qbs-1): out = tf.experimental.numpy.kron(out, Identity)
            out = tf.experimental.numpy.kron(out, op)
        else:
            out = 1
            for _ in range(which_qb): out = tf.experimental.numpy.kron(out, Identity)
            out = tf.experimental.numpy.kron(out, op)
            for _ in range(which_qb+1, total_num_qbs): out = tf.experimental.numpy.kron(out, Identity)
        assert out.shape == (2**total_num_qbs, 2**total_num_qbs)
        return out

# def get_CZ(control_qb, qb, total_num_qbs):

a = RX().get_matrix(1, 2)
b = 1



class Block(Layer):
    def __init__(self, num_nodes, layer_idx, num_anc, init_mean, init_std):
        super().__init__(num_nodes, layer_idx, num_anc, init_mean, init_std)
        self.param_var_lay = None