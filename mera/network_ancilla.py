'''
All tensors are batched with the first dimension being the batch axis. Except single tensor, tensors should have a
second axis for nodes. Canonical indices index all input dimensions before indexing all output dimension. Alternating
indices index one input dimension, one output dimension and so forth.
'''

import tensorflow as tf
import numpy as np
from mera import network


class Network(network.Network):
    def __init__(self, num_pixels, deph_p, num_anc, init_std, lr, config):
        super().__init__(num_pixels, deph_p, num_anc, init_std, lr, config)

        assert num_anc == 1
        self.kraus_ops_half_bd = self.construct_dephasing_multiqubit_kraus(1)

        self.layers = []
        self.list_num_nodes = [7, 8, 3, 4, 1, 2, 1]
        self.layers.append(network.Ent_Layer(self.list_num_nodes[0], 1, self.num_anc, self.init_mean, self.init_std))
        self.layers.append(Iso_Layer(self.list_num_nodes[1], 2, self.num_anc, self.init_mean, self.init_std))
        for i in range(2, self.num_layers-1, 2):
            self.layers.append(network.Ent_Layer(self.list_num_nodes[i], i+1, 0, self.init_mean, self.init_std))
            self.layers.append(Iso_Layer(self.list_num_nodes[i+1], i+2, 0, self.init_mean, self.init_std))
        self.layers.append(Iso_Layer(self.list_num_nodes[-1], self.num_layers, 0, self.init_mean, self.init_std))
        self.var_list = [layer.param_var_lay for layer in self.layers]

    # @tf.function
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

        layer_out = self.layers[1].get_1st_iso_lay_out_bottleneck(layer_out, left_over)
        if self.deph_net: layer_out = self.dephase_1st_iso_lay_out(layer_out)

        layer_out = self.layers[2].get_2nd_ent_lay_out(layer_out)
        if self.deph_net: layer_out = self.dephase_2nd_ent_lay_out(layer_out)

        layer_out = self.layers[3].get_2nd_iso_lay_out(layer_out)
        if self.deph_net:
            layer_out = tf.expand_dims(layer_out, 1)
            layer_out = self.dephase_after_bottleneck(layer_out, num_bd=4)
            layer_out = layer_out[:, 0]

        layer_out = self.layers[4].get_3rd_ent_lay_out(layer_out)
        if self.deph_net: layer_out = self.dephase_3rd_ent_lay_out(layer_out)

        layer_out = self.layers[5].get_3rd_iso_lay_out(layer_out)
        if self.deph_net:
            layer_out = tf.expand_dims(layer_out, 1)
            layer_out = self.dephase_after_bottleneck(layer_out, num_bd=2)
            layer_out = layer_out[:, 0]

        final_layer_out = self.layers[6].get_4th_iso_lay_out(layer_out)

        output_probs = tf.math.abs(tf.linalg.diag_part(final_layer_out))
        return output_probs

    def dephase_after_bottleneck(self, tensors, num_bd=1):
        batch_size, num_nodes = tensors.shape[:2]
        matrices = tf.reshape(tensors, [batch_size, num_nodes, *[2**num_bd]*2])
        if num_bd == 1:     kraus_ops = self.kraus_ops_half_bd
        elif num_bd == 2:   kraus_ops = self.kraus_ops_1_bd
        elif num_bd == 4:   kraus_ops = self.kraus_ops_2_bd
        else: raise NotImplementedError
        dephased = tf.einsum('kab, znbc, kdc -> znad', kraus_ops, matrices, kraus_ops)
        return tf.reshape(dephased, [batch_size, num_nodes, *[2]*num_bd*2])

    def dephase_1st_iso_lay_out(self, tensor):
        l, u = Network._lowercases, Network._uppercases
        for i in range(8):
            contract_str = 'X'+u[i]+l[i]+', Z'+l[:16]+', X'+u[8+i]+l[8+i]+' -> Z'+l[:i]+u[i]+l[i+1:8+i]+u[8+i]+l[9+i:16]
            tensor = tf.einsum(contract_str, self.kraus_ops_half_bd, tensor, self.kraus_ops_half_bd)
        return tensor

    def dephase_2nd_ent_lay_out(self, tensor):
        l, u = Network._lowercases, Network._uppercases
        for i in range(6):
            # 'YXWV' are the left-over bonds that do not need to dephase again here
            contract_str = 'U'+u[i]+l[i]+', Z Y'+l[:6]+'XW'+l[6:12]+'V, U'+u[6+i]+l[6+i]+\
                           ' -> Z Y'+l[:i]+u[i]+l[i+1:6]+'XW'+l[6:6+i]+u[6+i]+l[7+i:12]+'V'
            tensor = tf.einsum(contract_str, self.kraus_ops_half_bd, tensor, self.kraus_ops_half_bd)
        return tensor

    def dephase_3rd_ent_lay_out(self, tensor):
        l, u = Network._lowercases, Network._uppercases
        for i in range(2):
            # 'YXWV' are the left-over bonds that do not need to dephase again here
            contract_str = 'U'+u[i]+l[i]+', Z YabXWcdV, U'+u[2+i]+l[2+i]+\
                           ' -> Z Y'+l[:i]+u[i]+l[i+1:2]+'XW'+l[2:2+i]+u[2+i]+l[3+i:4]+'V'
            tensor = tf.einsum(contract_str, self.kraus_ops_half_bd, tensor, self.kraus_ops_half_bd)
        return tensor


class Iso_Layer(network.Iso_Layer):

    def __init__(self, num_nodes, layer_idx, num_anc, init_mean, init_std):
        super().__init__(num_nodes, layer_idx, num_anc, init_mean, init_std)

    def get_1st_iso_lay_out_bottleneck(self, inputs, left_over_data_inputs):
        assert left_over_data_inputs.shape[1] == 2
        l = Iso_Layer._lowercases
        unitary_tensors = self.get_unitary_tensors()
        num_nodes = unitary_tensors.shape[0]

        contracted = tf.einsum('abcd, zce, zdfgh, ibeg -> zaifh',
                               unitary_tensors[0], left_over_data_inputs[:, 0], inputs[:, 0], tf.math.conj(unitary_tensors[0]))
        # :contracted: single tensor with alternating indices

        batch_size, half_bd_dim = contracted.shape[0], self.bond_dim // 2
        shapes = [batch_size] + [half_bd_dim]*4 + [self.bond_dim]*2
        contracted = tf.einsum('zayiyfh -> zaifh', tf.reshape(contracted, shapes))

        for i in range(1, num_nodes-1):
            contract_str_with_tracing = 'ABCD, Z'+l[:2*i]+'CE, ZDFGH, IBEG -> Z'+l[:2*i]+'AIFH'
            contracted = tf.einsum(contract_str_with_tracing,
                                   unitary_tensors[i], contracted, inputs[:, i], tf.math.conj(unitary_tensors[i]))
            # :contracted: single tensor with alternating indices
            shapes = [batch_size] + [half_bd_dim]*(2*i) + [half_bd_dim]*4 + [self.bond_dim]*2
            # 'Z'+l[:2*i]+'AYIYFH -> Z'+l[:2*i]+'AIFH'
            contracted = tf.linalg.trace(tf.transpose(
                tf.reshape(contracted, shapes), perm=[*np.arange(1+2*i+1), 1+2*i+2, 1+2*i+4, 1+2*i+5, 1+2*i+1, 1+2*i+3]))

        bond_inds, last = l[:2 * (num_nodes-1)], num_nodes - 1
        contract_str_with_tracing = 'ABCD, Z'+bond_inds+'CE, ZDF, GBEF -> Z'+bond_inds+'AG'
        output = tf.einsum(contract_str_with_tracing,
                           unitary_tensors[last], contracted, left_over_data_inputs[:, 1], tf.math.conj(unitary_tensors[last]))
        # :output: single tensor with alternating indices

        shapes = [batch_size] + [half_bd_dim]*(2*last) + [half_bd_dim]*4
        # 'Z'+l[:2*last]+'AYGY -> Z'+l[:2*last]+'AG'
        output = tf.linalg.trace(tf.transpose(
                    tf.reshape(output, shapes), perm=[*np.arange(1+2*last+1), 1+2*last+2, 1+2*last+1, 1+2*last+3]))

        output = tf.transpose(output, perm=[0, *np.arange(1, 16, 2), *np.arange(2, 17, 2)])
        # :output: single tensor with canonical indices
        return output