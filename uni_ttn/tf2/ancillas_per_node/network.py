import tensorflow as tf
import numpy as np
import string, sys
sys.path.append('../')
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

        self.num_out_qubits = 1

        self.layers = []
        self.list_num_nodes = [int(self.num_pixels / 2**(i+1)) for i in range(self.num_layers)]
        for i in range(self.num_layers):
            self.layers.append(Layer(self.list_num_nodes[i], i, self.num_anc, self.init_mean, self.init_std))
        self.var_list = [layer.param_var_lay for layer in self.layers]

        self.cce = tf.keras.losses.CategoricalCrossentropy()
        if config['tree']['opt']['opt'] == 'adam':
            if not config['tree']['opt']['adam']['user_lr']: self.opt = tf.keras.optimizers.Adam()
            else: self.opt = tf.keras.optimizers.Adam(lr)
        elif config['tree']['opt']['opt'] == 'spsa':
            self.opt = spsa.Spsa(self, self.config['tree']['opt']['spsa'])
        else:
            raise NotImplementedError

        self.grads = None

    @tf.function
    def get_network_output(self, input_batch: tf.constant):
        batch_size = input_batch.shape[0]
        input_batch = tf.cast(input_batch, tf.complex64)
        input_batch = tf.einsum('zna, znb -> znab', input_batch, input_batch)   # omit conjugation since input is real
        if self.deph_data: input_batch = self.dephase(input_batch)

        layer_out = self.layers[0].get_layer_output(input_batch)
        if self.deph_net: layer_out = self.dephase(layer_out)
        for i in range(1, self.num_layers-1):
            layer_out = self.layers[i].get_layer_output(layer_out)
            if self.deph_net: layer_out = self.dephase(layer_out)

        final_layer_out = tf.reshape(
            self.layers[self.num_layers-1].get_layer_output(layer_out)[:, 0],
            [batch_size, *[2]*(2*self.num_out_qubits)])

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
        else:
            raise NotImplementedError

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
        return (1 - self.deph_p) * tensor + self.deph_p * tf.linalg.diag(tf.linalg.diag_part(tensor))


class Layer:
    def __init__(self, num_nodes, layer_idx, num_anc, init_mean, init_std):
        self.num_anc = num_anc
        self.layer_idx = layer_idx
        self.num_diags = 2 ** (num_anc + 2)
        self.num_op_params = self.num_diags ** 2
        self.num_nodes = num_nodes
        self.init_mean, self.std = init_mean, init_std
        self.num_bd = 2 * (num_anc + 2)

        self.param_var_lay = tf.Variable(
            tf.random_normal_initializer(mean=init_mean, stddev=init_std)(
                shape=[self.num_op_params, num_nodes], dtype=tf.float32,
            ), name='param_var_lay_%s' % layer_idx, trainable=True)

        self.cs = string.ascii_lowercase

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
        unitary_tensor = tf.reshape(unitary_matrix, [self.num_nodes, *[2]*self.num_bd])
        return unitary_tensor

    def get_layer_output(self, input):
        cs, num_bd = self.cs, self.num_bd

        left_input, right_input = input[:, ::2], input[:, 1::2]
        unitary_tensor = self.get_unitary_tensor()
        ancilla = tf.constant([[1, 0], [0, 0]], dtype=tf.complex64)
        input = tf.einsum('znab, zncd -> znabcd', left_input, right_input)
        for _ in range(self.num_anc):  input = tf.tensordot(input, ancilla, 0)
        perm = [0, 1, *list(range(2, 2+num_bd, 2)), *list(range(3, 3+num_bd, 2))]
        input = tf.transpose(input, perm=perm)

        # use y instead of n in the following strings
        # 'yabcdef, zyabcghi -> zydefghi'
        left_contract_str = 'y' + cs[:num_bd] + ',zy' + cs[:num_bd//2] + cs[num_bd:num_bd+num_bd//2] \
                             + '->zy' + cs[num_bd//2:num_bd] + cs[self.num_bd:num_bd+num_bd//2]
        left_contracted = tf.einsum(left_contract_str, unitary_tensor, input)
        # 'zydefghi, yghiabc -> zydefabc'
        right_contract_str = 'zy' + left_contract_str.split('->')[1].strip()[2:] \
                             + ',y' + cs[self.num_bd:num_bd+num_bd//2] + cs[:num_bd//2] \
                             + '->zy' + cs[num_bd//2:num_bd] + cs[:num_bd//2]
        contracted = tf.einsum(right_contract_str, left_contracted, tf.math.conj(unitary_tensor))

        if self.num_anc <= 1:
            # 'zyxabwab -> zyxw'
            trace_str = 'zyx' + cs[:num_bd//2-1] + 'w' + cs[:num_bd//2-1] + '->zyxw'
            output = tf.einsum(trace_str, contracted)
        elif self.num_anc == 2:
            # 'zyefghabcd -> zyeafbgchd'
            output = tf.transpose(contracted, perm=[0, 1, 2, 6, 3, 7, 4, 8, 5, 9])
            for _ in range(3): output = tf.linalg.trace(output)
        else:
            raise NotImplementedError

        return output