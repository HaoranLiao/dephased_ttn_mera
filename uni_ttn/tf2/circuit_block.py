import tensorflow as tf
import numpy as np
from uni_ttn.tf2.network import Layer

Identity = tf.constant([[1, 0], [0, 1]], dtype=tf.complex64)
Hadamard = 1 / np.sqrt(2) * tf.constant([[1, 1], [1, -1]], dtype=tf.complex64)


class RX:
    def __init__(self, num_nodes, num_parallel_rxs, init_mean=0, init_std=1):
        self.params_ = tf.cast(tf.Variable(
            0.5 * tf.random_normal_initializer(mean=init_mean, stddev=init_std)(
                shape=[num_nodes, num_parallel_rxs], dtype=tf.float32,
            ), trainable=True), tf.complex64)

        outer_lst = []
        for i in range(self.params_.shape[0]):
            lst = []
            for j in range(self.params_.shape[1]):
                theta = self.params_[i][j]
                rx = tf.stack([tf.concat([tf.cos(theta), -1.0j * tf.sin(theta)], axis=0),
                               tf.concat([-1.0j * tf.sin(theta), tf.cos(theta)], axis=0)])
                lst.append(rx)
            outer_lst.append(tf.stack(lst))
        self.RXs_ = tf.stack(outer_lst)

    def construct(self):
        return self.make_multi_q_op(self.RXs_)

    @staticmethod
    def make_multi_q_op(ops):
        out = []
        for i in range(ops.shape[0]):
            tensored = ops[i, 0]
            for j in range(1, ops.shape[1]):
                tensored = tf.experimental.numpy.kron(tensored, ops[i, j])
            out.append(tensored)
        return tf.stack(out)


class CZ:
    def __init__(self, num_nodes, num_qubits):
        self.num_nodes = num_nodes
        self.num_qubits = num_qubits
        self.cz = tf.stack([tf.concat([1.0 + 0.0j, 0.0j, 0.0j, 0.0j], axis=0),
                            tf.concat([0.0j, 1.0 + 0.0j, 0.0j, 0.0j], axis=0),
                            tf.concat([0.0j, 0.0j, 1.0 + 0.0j, 0.0j], axis=0),
                            tf.concat([0.0j, 0.0j, 0.0j, -1.0 + 0.0j], axis=0)])

    def construct(self):
        cz_layer = CZ.two_qubit_op(self.cz, 0, 1, self.num_qubits)
        for i in range(1, self.num_qubits - 1):
            cz_layer = cz_layer @ CZ.two_qubit_op(self.cz, i, i + 1, self.num_qubits)

        return tf.cast(tf.stack([cz_layer for _ in range(self.num_nodes)]), tf.complex64)

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
        return out


class H:
    def __init__(self, num_nodes, num_qubits):
        self.num_nodes = num_nodes
        self.num_qubits = num_qubits
        self.h = Hadamard

    def construct(self):
        h_layer = 1.0 + 0.0j
        for _ in range(self.num_qubits): h_layer = tf.experimental.numpy.kron(h_layer, self.h)
        return tf.stack([h_layer for _ in range(self.num_nodes)])

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


class Block9(Layer):
    def __init__(self, num_nodes, layer_idx, num_anc, init_mean, init_std):
        super().__init__(num_nodes, layer_idx, num_anc, init_mean, init_std)
        assert num_anc == 1
        self.num_anc = num_anc
        self.layer_idx = layer_idx
        self.bond_dim = 2 ** (num_anc + 1)
        self.num_nodes = num_nodes
        self.init_mean, self.std = init_mean, init_std
        self.num_in_qbs = 2 * (1 + self.num_anc)

    def get_unitary_tensors(self):
        h = H(self.num_nodes, self.num_in_qbs).construct()
        cz = CZ(self.num_nodes, self.num_in_qbs).construct()
        rx = RX(self.num_nodes, self.num_in_qbs).construct()
        self.unitary_matrices = tf.einsum('nab, nbc, ncd -> nad', h, cz, rx)
        unitary_tensors = tf.reshape(self.unitary_matrices, [self.num_nodes, *[self.bond_dim] * 4])
        return unitary_tensors


if __name__ == '__main__':
    block9 = Block9(100, 0, 1, 0, 1)
    unitary_tensors = block9.get_unitary_tensors()
    mat = block9.unitary_matrices
    for i in range(len(mat)):
        print(tf.linalg.trace(mat[i] @ tf.math.conj(tf.transpose(mat[i]))) / mat.shape[1])