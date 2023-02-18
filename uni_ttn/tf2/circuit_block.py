import tensorflow as tf
import numpy as np

# import uni_ttn.tf2.network

Identity = tf.constant([[1, 0], [0, 1]], dtype=tf.complex64)
Hadamard = 1 / np.sqrt(2) * tf.constant([[1, 1], [1, -1]], dtype=tf.complex64)


class RX:
    def __init__(self, params_):
        self.RXs_ = None
        self.params_ = params_

    def construct(self):
        num_qubits, num_nodes = self.params_.shape
        self.RXs_ = tf.zeros((num_nodes, 2**num_qubits, 2**num_qubits), dtype=tf.complex64)
        for n in range(num_nodes):  # so the first axis of self.RXs_ is num_nodes
            theta = self.params_[0, n]
            sin_t, cos_t = tf.cast(tf.sin(theta), tf.complex64), tf.cast(tf.cos(theta), tf.complex64)
            tensored = tf.stack([[cos_t, -1.0j * sin_t],
                                 [-1.0j * sin_t, cos_t]])
            for q in range(1, num_qubits):
                theta = self.params_[q, n]
                sin_t, cos_t = tf.cast(tf.sin(theta), tf.complex64), tf.cast(tf.cos(theta), tf.complex64)
                rx = tf.stack([[cos_t, -1.0j * sin_t],
                               [-1.0j * sin_t, cos_t]])
                tensored = tf.experimental.numpy.kron(tensored, rx)
            self.RXs_ = tf.tensor_scatter_nd_update(self.RXs_, [[n]], tensored[None, :])
        return self.RXs_


class CZ:
    def __init__(self, num_nodes, num_qubits):
        self.num_nodes = num_nodes
        self.num_qubits = num_qubits
        self.cz = tf.constant([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=tf.complex64)

    def construct(self):
        cz_layer = CZ.two_qubit_op(self.cz, 0, 1, self.num_qubits)
        for i in range(1, self.num_qubits - 1):
            cz_layer = cz_layer @ CZ.two_qubit_op(self.cz, i, i + 1, self.num_qubits)

        return tf.repeat(cz_layer[None, :], repeats=self.num_nodes, axis=0)

    @staticmethod
    def two_qubit_op(op, fir_qb, sec_qb, total_num_qbs):
        assert -1 < fir_qb < sec_qb < total_num_qbs and op.shape == (4, 4)
        assert fir_qb + 1 == sec_qb, NotImplementedError
        if fir_qb == 0:
            out = op
            for _ in range(total_num_qbs - 2): out = tf.experimental.numpy.kron(out, Identity)
        elif fir_qb == total_num_qbs - 2:
            out = 1.0 + 0.0j
            for _ in range(total_num_qbs - 2): out = tf.experimental.numpy.kron(out, Identity)
            out = tf.experimental.numpy.kron(out, op)
        else:
            out = 1.0 + 0.0j
            for _ in range(fir_qb): out = tf.experimental.numpy.kron(out, Identity)
            out = tf.experimental.numpy.kron(out, op)
            for _ in range(fir_qb + 2, total_num_qbs): out = tf.experimental.numpy.kron(out, Identity)
        assert out.shape == (2 ** total_num_qbs, 2 ** total_num_qbs)
        return tf.cast(out, tf.complex64)


class H:
    def __init__(self, num_nodes, num_qubits):
        self.num_nodes = num_nodes
        self.num_qubits = num_qubits
        self.h = Hadamard

    def construct(self):
        h_layer = 1.0 + 0.0j
        for _ in range(self.num_qubits): h_layer = tf.experimental.numpy.kron(h_layer, self.h)
        return tf.repeat(h_layer[None, :], repeats=self.num_nodes, axis=0)

    @staticmethod
    def single_qubit_op(op, which_qb, total_num_qbs):
        assert -1 < which_qb < total_num_qbs and op.shape == (2, 2)
        if which_qb == 0:
            out = op
            for _ in range(total_num_qbs - 1): out = tf.experimental.numpy.kron(out, Identity)
        elif which_qb == total_num_qbs - 1:
            out = 1
            for _ in range(total_num_qbs - 1): out = tf.experimental.numpy.kron(out, Identity)
            out = tf.experimental.numpy.kron(out, op)
        else:
            out = 1
            for _ in range(which_qb): out = tf.experimental.numpy.kron(out, Identity)
            out = tf.experimental.numpy.kron(out, op)
            for _ in range(which_qb + 1, total_num_qbs): out = tf.experimental.numpy.kron(out, Identity)
        assert out.shape == (2 ** total_num_qbs, 2 ** total_num_qbs)
        return out


class Block9:
    def __init__(self, num_nodes, layer_idx, num_anc, init_mean, init_std, num_repeat=5):
        assert num_anc == 1
        self.num_anc = num_anc
        self.layer_idx = layer_idx
        self.bond_dim = 2 ** (num_anc + 1)
        self.num_nodes = num_nodes
        self.init_mean, self.std = init_mean, init_std
        self.num_in_qbs = 2 * (1 + self.num_anc)
        self.num_repeat = num_repeat
        self.unitary_matrices = None

        self.param_var_lay = tf.Variable(
            0.5 * tf.random_normal_initializer(mean=init_mean, stddev=init_std)(
                shape=[num_repeat * self.num_in_qbs, num_nodes], dtype=tf.float32,
            ), name='param_var_lay_%s' % layer_idx, trainable=True)

        self.h = H(self.num_nodes, self.num_in_qbs).construct()
        self.cz = CZ(self.num_nodes, self.num_in_qbs).construct()

    def get_unitary_tensors(self):
        rx = RX(self.param_var_lay[0 * self.num_in_qbs: 0 * self.num_in_qbs + self.num_in_qbs]).construct()
        self.unitary_matrices = tf.einsum('nab, nbc, ncd -> nad', self.h, self.cz, rx)

        if self.num_repeat > 1:
            for i in range(1, self.num_repeat):
                rx = RX(self.param_var_lay[i * self.num_in_qbs: i * self.num_in_qbs + self.num_in_qbs]).construct()
                self.unitary_matrices = tf.einsum('nab, nbc, ncd, nde -> nae', self.h, self.cz, rx,
                                                  self.unitary_matrices)

        unitary_tensors = tf.reshape(self.unitary_matrices, [self.num_nodes, *[self.bond_dim] * 4])
        return unitary_tensors

    def get_layer_output(self, input):
        left_input, right_input = input[:, ::2], input[:, 1::2]
        unitary_tensor = self.get_unitary_tensors()
        left_contracted = tf.einsum('nabcd, znce, zndf -> znabef', unitary_tensor, left_input, right_input)
        output = tf.einsum('znabef, nagef -> znbg', left_contracted, tf.math.conj(unitary_tensor))
        return output


if __name__ == '__main__':
    block9 = Block9(100, 0, 1, 0, 1)
    unitary_tensors = block9.get_unitary_tensors()
    mat = block9.unitary_matrices
    for i in range(len(mat)):
        print(tf.linalg.trace(mat[i] @ tf.math.conj(tf.transpose(mat[i]))) / mat.shape[1])
