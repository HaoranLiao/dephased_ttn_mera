import tensorflow as tf
import numpy as np
import string, sys
sys.path.append('../../uni_ttn/tf2.7/')
import spsa


class Network:
    _lowercases, _uppercases = string.ascii_lowercase, string.ascii_uppercase

    def __init__(self, num_pixels, deph_p, num_anc, config, tune_config):
        self.config = config
        self.num_anc = num_anc
        self.num_pixels = num_pixels
        self.num_layers = int(np.log2(num_pixels)) * 2 - 1
        self.init_std = init_std
        self.init_mean = config['tree']['param']['init_mean']
        self.deph_data = config['meta']['deph']['data']
        self.deph_net = config['meta']['deph']['network']
        self.deph_p = float(deph_p)
        if not deph_p: self.deph_data, self.deph_net = False, False

        self.num_out_qubits = self.num_anc + 1
        self.kraus_ops_1_bd = self.construct_dephasing_multiqubit_kraus(self.num_out_qubits)
        self.kraus_ops_2_bd = self.construct_dephasing_multiqubit_kraus(2*self.num_out_qubits)
        self.kraus_ops_4_bd = self.construct_dephasing_multiqubit_kraus(4*self.num_out_qubits)

        self.layers = []
        self.list_num_nodes = [7, 8, 3, 4, 1, 2, 1]
        for i in range(0, self.num_layers-1, 2):
            self.layers.append(Ent_Layer(self.list_num_nodes[i], i+1, self.num_anc, self.init_mean, self.init_std))
            self.layers.append(Iso_Layer(self.list_num_nodes[i+1], i+2, self.num_anc, self.init_mean, self.init_std))
        self.layers.append(Iso_Layer(self.list_num_nodes[-1], self.num_layers, self.num_anc, self.init_mean, self.init_std))
        self.var_list = [layer.param_var_lay for layer in self.layers]

        self.bond_dim = 2 ** (num_anc + 1)
        if num_anc:
            self.ancilla = tf.constant([[1, 0], [0, 0]], dtype=tf.complex64)
            self.ancillas = tf.constant([[1, 0], [0, 0]], dtype=tf.complex64)
            for _ in range(self.num_anc-1):
                self.ancillas = tf.experimental.numpy.kron(self.ancillas, self.ancilla)

        self.cce = tf.keras.losses.CategoricalCrossentropy()
        if config['tree']['opt']['opt'] == 'adam':
            if not tune_config.get('tune_lr', False): self.opt = tf.keras.optimizers.Adam()
            else: self.opt = tf.keras.optimizers.Adam(tune_config['tune_lr'])
        elif config['tree']['opt']['opt'] == 'spsa':
            self.opt = spsa.Spsa(self, tune_config)
        else:
            raise NotImplementedError

        self.grads = None

    @tf.function
    def get_network_output(self, input_batch: tf.constant):
        batch_size = input_batch.shape[0]
        input_batch = tf.cast(input_batch, tf.complex64)
        input_batch = tf.einsum('zna, znb -> znab', input_batch, input_batch)   # omit conjugation since input is real
        if self.num_anc:
            input_batch = tf.reshape(
                tf.einsum('znab, cd -> znacbd', input_batch, self.ancillas),
                [batch_size, 16, self.bond_dim, self.bond_dim])
        if self.deph_data: input_batch = self.dephase(input_batch, num_bd=1)
        # :input_batch: tensors with canonical indices

        left_over = tf.gather(input_batch, [0, 15], axis=1)
        # No need to dephase :left_over: since all data qubits are dephased
        layer_out = self.layers[0].get_1st_ent_lay_out(input_batch[:, 1:15])
        if self.deph_net: layer_out = self.dephase(layer_out, num_bd=2)

        layer_out = self.layers[1].get_1st_iso_lay_out(layer_out, left_over)
        if self.deph_net: layer_out = self.dephase_1st_iso_lay_out(layer_out)

        layer_out = self.layers[2].get_2nd_ent_lay_out(layer_out)
        if self.deph_net: layer_out = self.dephase_2nd_ent_lay_out(layer_out)

        layer_out = self.layers[3].get_2nd_iso_lay_out(layer_out)
        if self.deph_net:
            layer_out = tf.expand_dims(layer_out, 1)
            layer_out = self.dephase(layer_out, num_bd=4)
            layer_out = layer_out[:, 0]

        layer_out = self.layers[4].get_3rd_ent_lay_out(layer_out)
        if self.deph_net: layer_out = self.dephase_3rd_ent_lay_out(layer_out)

        layer_out = self.layers[5].get_3rd_iso_lay_out(layer_out)
        if self.deph_net:
            layer_out = tf.expand_dims(layer_out, 1)
            layer_out = self.dephase(layer_out, num_bd=2)
            layer_out = layer_out[:, 0]

        final_layer_out = self.layers[6].get_4th_iso_lay_out(layer_out)

        output_probs = tf.math.abs(tf.linalg.diag_part(final_layer_out))
        return output_probs

    def update_no_processing(self, input_batch: np.ndarray, label_batch: np.ndarray):
        input_batch = tf.constant(input_batch, dtype=tf.complex64)
        label_batch = tf.constant(label_batch, dtype=tf.float32)
        with tf.GradientTape() as tape:
            loss = self.loss(input_batch, label_batch)
        grads = tape.gradient(loss, self.var_list)
        self.opt.apply_gradients(zip(grads, self.var_list))

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

    def dephase(self, tensors, num_bd=1):
        '''
        :param tensor: tensors with canonical indices
        :param num_bd: number of qubits per bond
        :return: tensors with the same shape and with canonical indices
        '''
        batch_size, num_nodes = tensors.shape[:2]
        matrices = tf.reshape(tensors, [batch_size, num_nodes, *[self.bond_dim**num_bd]*2])
        if num_bd == 1:     kraus_ops = self.kraus_ops_1_bd
        elif num_bd == 2:   kraus_ops = self.kraus_ops_2_bd
        elif num_bd == 4:   kraus_ops = self.kraus_ops_4_bd
        else: raise NotImplementedError
        dephased = tf.einsum('kab, znbc, kdc -> znad', kraus_ops, matrices, kraus_ops)
        return tf.reshape(dephased, [batch_size, num_nodes, *[self.bond_dim]*num_bd*2])

    def dephase_1st_iso_lay_out(self, tensor):
        '''
        :param tensor: single tensor with canonical indices
        :return: single tensor with canonical indices
        '''
        l, u = Network._lowercases, Network._uppercases
        for i in range(8):
            contract_str = 'X'+u[i]+l[i]+', Z'+l[:16]+', X'+u[8+i]+l[8+i]+' -> Z'+l[:i]+u[i]+l[i+1:8+i]+u[8+i]+l[9+i:16]
            tensor = tf.einsum(contract_str, self.kraus_ops_1_bd, tensor, self.kraus_ops_1_bd)
        return tensor

    def dephase_2nd_ent_lay_out(self, tensor):
        '''
        There are left-over bonds already dephased in this input tensor;
        The left-over bonds are the left-most and the right-most ones.
        :param tensor: single tensor with canonical indices
        :return: singel tensor with canonical indices
        '''
        l, u = Network._lowercases, Network._uppercases
        for i in range(6):
            # 'YXWV' are the left-over bonds that do not need to dephase again here
            contract_str = 'U'+u[i]+l[i]+', Z Y'+l[:6]+'XW'+l[6:12]+'V, U'+u[6+i]+l[6+i]+\
                           ' -> Z Y'+l[:i]+u[i]+l[i+1:6]+'XW'+l[6:6+i]+u[6+i]+l[7+i:12]+'V'
            tensor = tf.einsum(contract_str, self.kraus_ops_1_bd, tensor, self.kraus_ops_1_bd)
        return tensor

    def dephase_3rd_ent_lay_out(self, tensor):
        '''
        There are left-over bonds already dephased in this input tensor;
        The left-over bonds are the left-most and the right-most ones.
        :param tensor: single tensor with canonical indices
        :return: singel tensor with canonical indices
        '''
        l, u = Network._lowercases, Network._uppercases
        for i in range(2):
            # 'YXWV' are the left-over bonds that do not need to dephase again here
            contract_str = 'U'+u[i]+l[i]+', Z YabXWcdV, U'+u[2+i]+l[2+i]+\
                           ' -> Z Y'+l[:i]+u[i]+l[i+1:2]+'XW'+l[2:2+i]+u[2+i]+l[3+i:4]+'V'
            tensor = tf.einsum(contract_str, self.kraus_ops_1_bd, tensor, self.kraus_ops_1_bd)
        return tensor

    def construct_dephasing_multiqubit_kraus(self, num_out_qubits):
        m1 = tf.cast(tf.math.sqrt((2 - self.deph_p) / 2), tf.complex64) * tf.eye(2, dtype=tf.complex64)
        m2 = tf.cast(tf.math.sqrt(self.deph_p / 2), tf.complex64) * tf.constant([[1, 0], [0, -1]], dtype=tf.complex64)
        m = (m1, m2)
        combinations = tf.reshape(
            tf.transpose(tf.meshgrid(*[[0, 1]]*num_out_qubits)),
            [-1, num_out_qubits])
        kraus_ops = []
        for combo in combinations:
            tensor_prod = m[combo[0]]
            for idx in combo[1:]: tensor_prod = tf.experimental.numpy.kron(tensor_prod, m[idx])
            kraus_ops.append(tensor_prod)
        return tf.stack(kraus_ops)


class Ent_Layer:
    _name = 'entangler_layer'
    _lowercases, _uppercases = string.ascii_lowercase, string.ascii_uppercase

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
            ), name='param_var_ent_lay_%s' % layer_idx, trainable=True)

    def get_unitary_tensors(self):
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
        unitary_matrices = tf.einsum('nab, nbc, ndc -> nad', eigenvectors, diag_exp_mat, tf.math.conj(eigenvectors))
        unitary_tensors = tf.reshape(unitary_matrices, [self.num_nodes, *[self.bond_dim]*4])
        return unitary_tensors

    def get_1st_ent_lay_out(self, inputs):
        '''
        :param inputs: matrics
        :return: tensors with canonical indices
        '''
        first_half_inputs, second_half_inputs = inputs[:, ::2], inputs[:, 1::2]
        unitary_tensors = self.get_unitary_tensors()
        left_contracted = tf.einsum('nabcd, znce, zndf -> znabef', unitary_tensors, first_half_inputs, second_half_inputs)
        outputs = tf.einsum('znabef, nghef -> znabgh', left_contracted, tf.math.conj(unitary_tensors))
        return outputs

    def get_2nd_ent_lay_out(self, input):
        '''
        :param input: single tensor with canonical indices
        :return: single tensor with canonical indices
        '''
        l = Ent_Layer._lowercases
        unitary_tensors = self.get_unitary_tensors()
        contract_str = 'ZY'+l[:6]+'XW'+l[6:12]+'V, ABab, CDcd, EFef, GHgh, IJij, KLkl -> Z YABCDEFX WGHIJKLV'
        output = tf.einsum(contract_str, input, unitary_tensors[0], unitary_tensors[1], unitary_tensors[2],
                           tf.math.conj(unitary_tensors[0]), tf.math.conj(unitary_tensors[1]), tf.math.conj(unitary_tensors[2]))
        return output

    def get_3rd_ent_lay_out(self, input):
        '''
        :param input: single tensor with canonical indices
        :return: single tensor with canonical indices
        '''
        l = Ent_Layer._lowercases
        unitary_tensor = self.get_unitary_tensors()[0]
        contract_str = 'Z YabXWcdV, ABab, CDcd -> Z YABX WCDV'
        output = tf.einsum(contract_str, input, unitary_tensor, tf.math.conj(unitary_tensor))
        return output

class Iso_Layer(Ent_Layer):
    _name = 'isometry_layer'

    def __init__(self, num_nodes, layer_idx, num_anc, init_mean, init_std):
        super().__init__(num_nodes, layer_idx, num_anc, init_mean, init_std)

        self.param_var_lay = tf.Variable(
            tf.random_normal_initializer(mean=init_mean, stddev=init_std)(
                shape=[self.num_op_params, num_nodes], dtype=tf.float32,
            ), name='param_var_iso_lay_%s' % layer_idx, trainable=True)

    def get_1st_iso_lay_out(self, inputs, left_over_data_inputs):
        '''
        Bubbling contraction from left, starting with the first of the left_over_data_inputs,
        then the inputs, and ends with the second/last of the left_over_data_inputs
        :param input: tensors with canonical indices
        :param left_over_data_input: matrices
        :return: single tensor with canonical indices
        '''
        assert left_over_data_inputs.shape[1] == 2
        l = Iso_Layer._lowercases
        unitary_tensors = self.get_unitary_tensors()
        num_nodes = unitary_tensors.shape[0]

        contracted = tf.einsum('abcd, zce, zdfgh, ibeg -> zaifh',
                        unitary_tensors[0], left_over_data_inputs[:, 0], inputs[:, 0], tf.math.conj(unitary_tensors[0]))
        # :contracted: single tensor with alternating indices
        for i in range(1, num_nodes-1):
            contract_str_with_tracing = 'ABCD, Z'+l[:2*i]+'CE, ZDFGH, IBEG -> Z'+l[:2*i]+'AIFH'
            contracted = tf.einsum(contract_str_with_tracing,
                            unitary_tensors[i], contracted, inputs[:, i], tf.math.conj(unitary_tensors[i]))
            # :contracted: single tensor with alternating indices

        bond_inds, last = l[:2 * (num_nodes-1)], num_nodes - 1
        contract_str_with_tracing = 'ABCD, Z'+bond_inds+'CE, ZDF, GBEF -> Z'+bond_inds+'AG'
        output = tf.einsum(contract_str_with_tracing,
                    unitary_tensors[last], contracted, left_over_data_inputs[:, 1], tf.math.conj(unitary_tensors[last]))
        # :output: single tensor with alternating indices
        output = tf.transpose(output, perm=[0, *np.arange(1, 16, 2), *np.arange(2, 17, 2)])
        # :output: single tensor with canonical indices
        return output

    def get_2nd_iso_lay_out(self, input):
        '''
        :param input: single tensor with cononical indices
        :return: single tensor with cononical indices
        '''
        l = Iso_Layer._lowercases
        unitary_tensors = self.get_unitary_tensors()
        contract_str_with_tracing = 'ABab, CDcd, EFef, GHgh, Z'+l[:16]+', IBij, KDkl, MFmn, OHop -> ZACEGIKMO'
        output = tf.einsum(contract_str_with_tracing, unitary_tensors[0], unitary_tensors[1],
                           unitary_tensors[2], unitary_tensors[3], input,
                           tf.math.conj(unitary_tensors[0]), tf.math.conj(unitary_tensors[1]),
                           tf.math.conj(unitary_tensors[2]), tf.math.conj(unitary_tensors[3]))
        return output

    def get_3rd_iso_lay_out(self, input):
        '''
        :param input: single tensor with cononical indices
        :return: single tensor with cononical indices
        '''
        l = Iso_Layer._lowercases
        unitary_tensors = self.get_unitary_tensors()
        contract_str_with_tracing = 'ABab, CDcd, Z'+l[:8]+', EBef, GDgh -> ZACEG'
        output = tf.einsum(contract_str_with_tracing, unitary_tensors[0], unitary_tensors[1], input,
                           tf.math.conj(unitary_tensors[0]), tf.math.conj(unitary_tensors[1]))
        return output

    def get_4th_iso_lay_out(self, input):
        '''
        :param input: single tensor with cononical indices
        :return: single tensor with cononical indices
        '''
        unitary_tensor = self.get_unitary_tensors()[0]
        output = tf.einsum('ABab, Zabcd, CBcd -> ZAC', unitary_tensor, input, tf.math.conj(unitary_tensor))
        return output


