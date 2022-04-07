import tensorflow as tf
import numpy as np
import string
import spsa


class Network:
    def __init__(self, num_pixels, deph_p, num_anc, init_std, lr, config):
        self.config = config
        self.num_anc = num_anc
        self.num_pixels = num_pixels
        self.num_layers = int(np.log2(num_pixels))
        self.init_std = init_std
        self.init_mean = config['tree']['param']['init_mean']
        self.deph_data = config['meta']['deph']['data']
        self.deph_net = config['meta']['deph']['network']
        self.deph_p = float(deph_p)
        if not deph_p: self.deph_data, self.deph_net = False, False

        self.num_out_qubits = self.num_anc + 1
        if self.num_anc and self.deph_p > 0: self.construct_dephasing_kraus()

        self.layers = []
        self.list_num_nodes = [int(self.num_pixels / 2**(i+1)) for i in range(self.num_layers)]
        for i in range(self.num_layers):
            self.layers.append(Layer(self.list_num_nodes[i], i, self.num_anc, self.init_mean, self.init_std))
        self.var_list = [layer.param_var_lay for layer in self.layers]

        if num_anc:
            self.bond_dim = 2 ** (num_anc + 1)
            self.ancilla = tf.constant([[1, 0], [0, 0]], dtype=tf.complex64)
            self.ancillas = tf.constant([[1, 0], [0, 0]], dtype=tf.complex64)
            for _ in range(self.num_anc-1):
                self.ancillas = tf.experimental.numpy.kron(self.ancillas, self.ancilla)

        self.cce = tf.keras.losses.CategoricalCrossentropy()
        if config['tree']['opt']['opt'] == 'adam':
            if not config['tree']['opt']['adam']['user_lr']: self.opt = tf.keras.optimizers.Adam()
            else: self.opt = tf.keras.optimizers.Adam(lr)
        elif config['tree']['opt']['opt'] == 'spsa':
            self.opt = spsa.Spsa(self, self.config['tree']['opt']['spsa'])
        else:
            raise NotImplemented

        if self.num_anc < 4:
            chars = string.ascii_lowercase
            self.trace_einsum = 'za' + chars[2:2+self.num_anc] + 'b' + chars[2:2+self.num_anc] + '-> zab'

        self.grads = None

    @tf.function
    def get_network_output(self, input_batch: tf.constant):
        batch_size = input_batch.shape[0]
        input_batch = tf.cast(input_batch, tf.complex64)
        input_batch = tf.einsum('zna, znb -> znab', input_batch, input_batch)   # omit conjugation since input is real
        if self.num_anc:
            input_batch = tf.reshape(
                tf.einsum('znab, cd -> znacbd', input_batch, self.ancillas),
                [batch_size, 2*self.list_num_nodes[0], self.bond_dim, self.bond_dim])
        if self.deph_data: input_batch = self.dephase(input_batch)

        layer_out = self.layers[0].get_layer_output(input_batch)
        if self.deph_net: layer_out = self.dephase(layer_out)
        for i in range(1, self.num_layers-1):
            layer_out = self.layers[i].get_layer_output(layer_out)
            if self.deph_net: layer_out = self.dephase(layer_out)

        final_layer_out = tf.reshape(
            self.layers[self.num_layers-1].get_layer_output(layer_out)[:, 0],
            [batch_size, *[2]*(2*self.num_out_qubits)])

        if self.num_anc < 4:
            final_layer_out = tf.einsum(self.trace_einsum, final_layer_out)
        elif self.num_anc == 4:
            final_layer_out = tf.transpose(final_layer_out, perm=[0, 1, 6, 2, 7, 3, 8, 4, 9, 5, 10])    # zabcdefghij -> zafbgchdiej
            for _ in range(4): final_layer_out = tf.linalg.trace(final_layer_out)
        else:
            raise NotImplemented

        output_probs = tf.math.abs(tf.linalg.diag_part(final_layer_out))
        return output_probs

    def update_no_processing(self, input_batch: np.ndarray, label_batch: np.ndarray):
        input_batch = tf.constant(input_batch, dtype=tf.complex64)
        label_batch = tf.constant(label_batch, dtype=tf.float32)
        self.opt.minimize(self.loss(input_batch, label_batch), var_list=self.var_list)

    def update(self, input_batch: np.ndarray, label_batch: np.ndarray, epoch, apply_grads=True, counter=1):
        input_batch = tf.constant(input_batch, dtype=tf.complex64)
        label_batch = tf.constant(label_batch, dtype=tf.float32)

        if self.opt._name == 'Adam':
            with tf.GradientTape() as tape:
                loss = self.loss(input_batch, label_batch)
            grads = tape.gradient(loss, self.var_list)
        elif self.opt._name == 'Spsa':
            grads = self.opt.get_update(epoch, input_batch, label_batch)

        if not self.grads:
            self.grads = grads
        else:
            for i in range(len(grads)): self.grads[i] += grads[i]

        if apply_grads:
            if counter > 1:
                for i in range(len(self.grads)): self.grads[i] /= counter
            self.opt.apply_gradients(zip(self.grads, self.var_list))
            self.grads = None

    @tf.function
    def loss(self, input_batch, label_batch):
        return self.cce(label_batch, self.get_network_output(input_batch))

    def dephase(self, tensor):
        if self.num_anc: return tf.einsum('kab, znbc, kdc -> znad', self.kraus_ops, tensor, self.kraus_ops)
        else: return (1 - self.deph_p) * tensor + self.deph_p * tf.linalg.diag(tf.linalg.diag_part(tensor))

    def construct_dephasing_kraus(self):
        m1 = tf.cast(tf.math.sqrt(1 - self.deph_p), tf.complex64) * tf.eye(2, dtype=tf.complex64)
        m2 = tf.cast(tf.math.sqrt(self.deph_p), tf.complex64) * tf.constant([[1, 0], [0, 0]], dtype=tf.complex64)
        m3 = tf.cast(tf.math.sqrt(self.deph_p), tf.complex64) * tf.constant([[0, 0], [0, 1]], dtype=tf.complex64)
        m = (m1, m2, m3)
        combinations = tf.reshape(
            tf.transpose(tf.meshgrid(*[[0, 1, 2]]*self.num_out_qubits)),
            [-1, self.num_out_qubits])
        self.kraus_ops = []
        for combo in combinations:
            tensor_prod = m[combo[0]]
            for idx in combo[1:]: tensor_prod = tf.experimental.numpy.kron(tensor_prod, m[idx])
            self.kraus_ops.append(tensor_prod)
        self.kraus_ops = tf.stack(self.kraus_ops)


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
        num_off_diags = int(0.5 * (self.num_op_params - self.num_diags))
        real_off_params = self.param_var_lay[:num_off_diags]
        imag_off_params = self.param_var_lay[num_off_diags:2 * num_off_diags]
        diag_params = self.param_var_lay[2 * num_off_diags:]

        herm_shape = (self.num_diags, self.num_diags, self.num_nodes)
        diag_part = tf.transpose(tf.linalg.diag(tf.transpose(diag_params)), perm=[1, 2, 0])
        off_diag_indices = [[i, j] for i in range(self.num_diags) for j in range(i + 1, self.num_diags)]
        real_off_diag_part = tf.scatter_nd(
            indices=off_diag_indices,
            updates=real_off_params,
            shape=herm_shape)
        imag_off_diag_part = tf.scatter_nd(
            indices=off_diag_indices,
            updates=imag_off_params,
            shape=herm_shape)
        imag_whole = imag_off_diag_part - tf.transpose(imag_off_diag_part, perm=[1, 0, 2])
        real_whole = diag_part + real_off_diag_part + tf.transpose(real_off_diag_part, perm=[1, 0, 2])
        herm_matrix = tf.transpose(tf.complex(real_whole, imag_whole), perm=[2, 0, 1])

        (eigenvalues, eigenvectors) = tf.linalg.eigh(herm_matrix)
        eig_exp = tf.exp(1.0j * eigenvalues)
        diag_exp_mat = tf.linalg.diag(eig_exp)
        unitary_matrix = tf.einsum('nab, nbc, ndc -> nad', eigenvectors, diag_exp_mat, tf.math.conj(eigenvectors))
        unitary_tensor = tf.reshape(unitary_matrix, [self.num_nodes, *[self.bond_dim]*4])
        return unitary_tensor

    def get_layer_output(self, input):
        left_input, right_input = input[:, ::2], input[:, 1::2]
        unitary_tensor = self.get_unitary_tensor()
        left_contracted = tf.einsum('nabcd, znce, zndf -> znabef', unitary_tensor, left_input, right_input)
        output = tf.einsum('znabef, nagef -> znbg', left_contracted, tf.math.conj(unitary_tensor))
        return output