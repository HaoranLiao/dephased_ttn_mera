import tensorflow as tf
import numpy as np
import string


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
        print(f'Dephasing p: {deph_p:.1f}')
        if deph_p == 0: self.deph_data, self.deph_net = False, False
        self.layers = []

        self.num_out_qubits = self.num_anc + 1
        if self.num_anc and self.deph_p > 0: self.construct_dephasing_krauss()

        self.list_num_nodes = [int(self.num_pixels / 2**(i+1)) for i in range(self.num_layers)]
        for i in range(self.num_layers):
            self.layers.append(Layer(self.list_num_nodes[i], i, self.num_anc, self.init_mean, self.init_std))

        if num_anc:
            self.bond_dim = 2 ** (num_anc + 1)
            self.ancilla = tf.constant([[1, 0], [0, 0]], dtype=tf.complex64)
            self.ancillas = tf.constant([[1, 0], [0, 0]], dtype=tf.complex64)
            for _ in range(self.num_anc-1):
                self.ancillas = tf.experimental.numpy.kron(self.ancillas, self.ancilla)

        self.cce = tf.keras.losses.CategoricalCrossentropy()
        if not config['tree']['opt']['adam']['user_lr']: self.opt = tf.keras.optimizers.Adam()
        else: self.opt = tf.keras.optimizers.Adam(config['tree']['opt']['adam']['lr'])

        chars = string.ascii_lowercase
        self.trace_einsum = 'za' + chars[2:2+self.num_anc] + 'b' + chars[2:2+self.num_anc] + '-> zab'

    def get_network_output(self, input_batch):
        self.batch_size = len(input_batch)
        input_batch = tf.einsum('zna, znb -> znab', input_batch, input_batch)
        # input_batch = tf.cast(input_batch, dtype=tf.complex64)
        if self.num_anc:
            input_batch = tf.reshape(
                tf.einsum('znab, cd -> znacbd', input_batch, self.ancillas),
                [self.batch_size, 2*self.list_num_nodes[0], self.bond_dim, self.bond_dim])
        if self.deph_data: input_batch = self.dephase(input_batch)

        layer_out = self.layers[0].get_layer_output(input_batch)
        if self.deph_net: layer_out = self.dephase(layer_out)
        for i in range(1, self.num_layers-1):
            layer_out = self.layers[i].get_layer_output(layer_out)
            if self.deph_net: layer_out = self.dephase(layer_out)

        final_layer_out = tf.reshape(
            self.layers[self.num_layers-1].get_layer_output(layer_out)[:, 0],
            [self.batch_size, *[2]*(2*self.num_out_qubits)])
        final_layer_out = tf.einsum(self.trace_einsum, final_layer_out)

        output_probs = tf.math.abs(tf.linalg.diag_part(final_layer_out))
        return output_probs

    def update(self, input_batch, label_batch):
        # TODO: move it to the data level
        # self.input_batch = tf.constant(input_batch)
        # self.label_batch = tf.constant(label_batch, dtype=tf.float32)
        self.input_batch, self.label_batch = input_batch, label_batch
        self.opt.minimize(self.loss, var_list=[layer.param_var_lay for layer in self.layers])
        # self.get_network_output(self.input_batch)

    @tf.function
    def loss(self):
        pred_batch = self.get_network_output(self.input_batch)
        return self.cce(pred_batch, self.label_batch)

    def dephase(self, tensor):
        if self.num_anc: return tf.einsum('kab, znbc, kdc -> znad', self.krauss_ops, tensor, self.krauss_ops)
        else: return (1 - self.deph_p) * tensor + self.deph_p * tf.linalg.diag(tf.linalg.diag_part(tensor))

    def construct_dephasing_krauss(self):
        m1 = tf.cast(tf.math.sqrt(1 - self.deph_p), tf.complex64) * tf.eye(2, dtype=tf.complex64)
        m2 = tf.cast(tf.math.sqrt(self.deph_p), tf.complex64) * tf.constant([[1, 0], [0, 0]], dtype=tf.complex64)
        m3 = tf.cast(tf.math.sqrt(self.deph_p), tf.complex64) * tf.constant([[0, 0], [0, 1]], dtype=tf.complex64)
        m = (m1, m2, m3)
        combinations = tf.reshape(
            tf.transpose(tf.meshgrid(*[[0, 1, 2]]*self.num_out_qubits)),
            [-1, self.num_out_qubits])
        self.krauss_ops = []
        for combo in combinations:
            tensor_prod = m[combo[0]]
            for idx in combo[1:]: tensor_prod = tf.experimental.numpy.kron(tensor_prod, m[idx])
            self.krauss_ops.append(tensor_prod)
        self.krauss_ops = tf.stack(self.krauss_ops)

class Layer:
    def __init__(self, num_nodes, layer_idx, num_anc, init_mean, init_std):
        self.num_anc = num_anc
        self.layer_idx = layer_idx
        self.bond_dim = 2 ** (num_anc + 1)
        self.num_diags = 2 ** (2 * (num_anc + 1))
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
        unitary_tensor = tf.reshape(self.unitary_matrix, [self.num_nodes, *[self.bond_dim]*4])
        return unitary_tensor

    def get_layer_output(self, input):
        left_input, right_input = input[:, ::2], input[:, 1::2]
        unitary_tensor = self.get_unitary_tensor()
        left_contracted = tf.einsum('nabcd, znce, zndf -> znabef', unitary_tensor, left_input, right_input)
        output = tf.einsum('znabef, nahef -> znbh', left_contracted, tf.math.conj(unitary_tensor))
        return output


