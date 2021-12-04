import tensorflow as tf
import numpy as np


class Network:
    def __init__(self, num_pixels, deph_p, num_anc, config):
        self.config = config
        self.num_anc = num_anc
        self.num_pixels = num_pixels
        self.num_layers = int(np.log2(num_pixels))
        self.init_mean = config['tree']['param']['init_mean']
        self.init_std = config['tree']['param']['init_std']
        self.deph_data = config['meta']['deph']['data']
        self.deph_net = config['meta']['deph']['network']
        self.deph_p = float(deph_p)
        self.layers = []

        self.num_out_bonds = self.num_anc + 1
        if self.num_out_bonds > 1: self.construct_dephasing_krauss()

        self.list_num_nodes = [int(self.num_pixels / 2**(i+1)) for i in range(self.num_layers)]
        for i in range(self.num_layers):
            self.layers.append(Layer(self.list_num_nodes[i], i, self.num_anc, self.init_mean, self.init_std))

        if num_anc:
            self.ancilla = tf.constant([[1, 0], [0, 0]], dtype=tf.complex64)
            self.ancillas = tf.constant([[1, 0], [0, 0]], dtype=tf.complex64)
            for _ in range(self.num_anc-1):
                self.ancillas = tf.tensordot(self.ancillas, self.ancilla, axes=0)

        self.cce = tf.keras.losses.CategoricalCrossentropy()
        self.opt = tf.keras.optimizers.Adam()

    def get_network_output(self, input_batch):
        input_batch = tf.einsum('zna, znb -> znab', input_batch, input_batch)
        input_batch = tf.cast(input_batch, dtype=tf.complex64)
        if self.num_anc:
            input_batch = tf.tensordot(input_batch, self.ancillas, axes=0)
            input_batch = tf.transpose(
                input_batch,
                perm=[0, *list(range(1, 2*self.num_out_bonds+2, 2)), *list(range(2, 2*self.num_out_bonds+2, 2))])
        if self.deph_data: input_batch = self.dephase(input_batch)

        layer_out = self.layers[0].get_layer_output(input_batch)
        if self.deph_net: layer_out = self.dephase(layer_out)
        for i in range(1, self.num_layers-1):
            layer_out = self.layers[i].get_layer_output(layer_out)
            if self.deph_net: layer_out = self.dephase(layer_out)

        final_layer_out = self.layers[self.num_layers-1].get_layer_output(layer_out)[:, 0]
        # 'zabcd -> zab', final_layer_out
        # TODO: 'zabcd -> zab', final_layer_out

        output_probs = tf.math.abs(tf.linalg.diag_part(final_layer_out))
        return output_probs

    def update(self, input_batch, label_batch):
        self.input_batch = tf.constant(input_batch)
        self.label_batch = tf.constant(label_batch, dtype=tf.float32)
        self.opt.minimize(self.loss, var_list=[layer.param_var_lay for layer in self.layers])

    @tf.function
    def loss(self):
        pred_batch = self.get_network_output(self.input_batch)
        return self.cce(pred_batch, self.label_batch)

    def dephase(self, tensor):
        if self.num_anc:
            # left_contracted = 'kabcd, znbedf -> kznaecf', krauss, rho, when there is one ancilla (old)
            # left_contracted = 'kabcd, zncdef -> kznabef', krauss, rho, when there is one ancilla
            left_contracted = tf.tensordot(
                self.krauss_ops, tensor,
                axes=[list(range(1+self.num_out_bonds, 1+2*self.num_out_bonds)),
                      list(range(2, 2+self.num_out_bonds))])
            # TODO: check where the 'zn' dimensions got placed
            # dephased_tensor = 'kznaecf, kgehf (kegfh transposed) -> znagch', left_contracted, krauss (real so no conj) (old)
            # dephased_tensor = 'kznabef, kghef (kefgh transposed) -> znabgh', left_contracted, krauss (real so no conj)
            dephased_tensor = tf.tensordot(
                left_contracted, self.krauss_ops,
                axes=[[0]+list(range(5, 3+2*self.num_out_bonds)),
                      [0]+list(range(3, 1+2*self.num_out_bonds))])
            return dephased_tensor
        else:
            return (1 - self.deph_p) * tensor + self.deph_p * tf.linalg.diag(tf.linalg.diag_part(tensor))

    def construct_dephasing_krauss(self):
        m1 = tf.cast(tf.math.sqrt(1 - self.deph_p), tf.complex64) * tf.eye(2, dtype=tf.complex64)
        m2 = tf.cast(tf.math.sqrt(self.deph_p), tf.complex64) * tf.constant([[1, 0], [0, 0]], dtype=tf.complex64)
        m3 = tf.cast(tf.math.sqrt(self.deph_p), tf.complex64) * tf.constant([[0, 0], [0, 1]], dtype=tf.complex64)
        m = (m1, m2, m3)
        combinations = tf.reshape(tf.transpose(
                                tf.meshgrid(*[[0, 1, 2]]*self.num_out_bonds)
                            ), [-1, self.num_out_bonds])
        self.krauss_ops = []
        for combo in combinations:
            tensor_prod = m[combo[0]]
            for idx in combo[1:]: tensor_prod = tf.tensordot(tensor_prod, m[idx], axes=0)
            self.krauss_ops.append(tensor_prod)
        self.krauss_ops = tf.stack(self.krauss_ops)
        self.krauss_ops = tf.transpose(
            self.krauss_ops,
            perm=[0, *list(range(1, 1+2*self.num_out_bonds, 2)), *list(range(2, 1+2*self.num_out_bonds, 2))])


class Layer:
    def __init__(self, num_nodes, layer_idx, num_anc, init_mean, init_std):
        self.num_anc = num_anc
        self.layer_idx = layer_idx
        self.num_in_bonds = 2 * (self.num_anc + 1)
        self.num_diags = 2 ** self.num_in_bonds
        self.num_op_params = self.num_diags ** 2
        self.num_nodes = num_nodes
        self.init_mean, self.std = init_mean, init_std

        self.param_var_lay = tf.Variable(
            tf.random_normal_initializer(mean=init_mean, stddev=init_std)(
                shape=[self.num_op_params, num_nodes], dtype=tf.float32,
            ), name='param_var_lay_%s' % layer_idx, trainable=True)

    def get_unitary_tensor(self):
        num_off_diags = int(0.5 * (self.num_diags**2 - self.num_diags))
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
        unitary_tensor = tf.reshape(self.unitary_matrix, [self.num_nodes, *[2]*(2*self.num_in_bonds)])
        return unitary_tensor

    def get_layer_output(self, input):
        left_input, right_input = input[:, ::2], input[:, 1::2]
        unitary_tensor = self.get_unitary_tensor()
        # 'nabcdefgh, zniajb -> znijcdefgh', unitary_tensor, left_input, when there is one ancilla
        left_contracted = tf.tensordot(
            unitary_tensor, left_input,
            axes=[list(range(1, self.num_anc+2)),
                  list(range(3, self.num_in_bonds+2, 2))])
        # TODO: check where the 'zn' dimensions got placed
        # 'znijcdefgh, znkcld -> znijklefgh', left_contracted, right_input,
        input_contracted = tf.tensordot(
            left_contracted, right_input,
            axes=[list(range(self.num_anc+3, self.num_in_bonds+2)),
                  list(range(3, self.num_in_bonds+2, 2))])
        # 'znijklefgh, nijklefmo -> znmogh', input_contracted, conj(unitary_tensor)
        output = tf.tensordot(
            input_contracted, tf.math.conj(unitary_tensor),
            axes=[list(range(2, 2*self.num_in_bonds)),
                  list(range(1, 2*self.num_in_bonds-1))])
        # 'znmogh -> znmgoh'
        output = tf.transpose(
            output,
            perm=[0, 1, *list(range(2, 2+2*(self.num_anc+1), 2)), *list(range(3, 2+2*(self.num_anc+1), 2))])
        return output

