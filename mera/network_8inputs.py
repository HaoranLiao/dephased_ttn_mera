import tensorflow as tf
import mera.network


class Network(mera.network.Network):

    def __init__(self, num_pixels, deph_p, num_anc, init_std, lr, config):
        super().__init__(num_pixels, deph_p, num_anc, init_std, lr, config)

        self.kraus_ops_4_bd = None

        self.layers = []
        self.list_num_nodes = [3, 4, 1, 2, 1]
        for i in range(0, self.num_layers-1, 2):
            self.layers.append(Ent_Layer(self.list_num_nodes[i], i+1, self.num_anc, self.init_mean, self.init_std))
            self.layers.append(Iso_Layer(self.list_num_nodes[i+1], i+2, self.num_anc, self.init_mean, self.init_std))
        self.layers.append(Iso_Layer(self.list_num_nodes[-1], self.num_layers, self.num_anc, self.init_mean, self.init_std))
        self.var_list = [layer.param_var_lay for layer in self.layers]

    @tf.function
    def get_network_output(self, input_batch: tf.constant):
        batch_size = input_batch.shape[0]
        input_batch = tf.cast(input_batch, tf.complex64)
        input_batch = tf.einsum('zna, znb -> znab', input_batch, input_batch)   # omit conjugation since input is real
        if self.num_anc:
            input_batch = tf.reshape(
                tf.einsum('znab, cd -> znacbd', input_batch, self.ancillas),
                [batch_size, 8, self.bond_dim, self.bond_dim])
        if self.deph_data: input_batch = self.dephase(input_batch, num_bd=1)
        # :input_batch: tensors with canonical indices

        left_over = tf.gather(input_batch, [0, 7], axis=1)
        # No need to dephase :left_over: since all data qubits are dephased
        layer_out = self.layers[0].get_1st_ent_lay_out(input_batch[:, 1:7])
        if self.deph_net: layer_out = self.dephase(layer_out, num_bd=2)

        layer_out = self.layers[1].get_1st_iso_lay_out(layer_out, left_over)
        if self.deph_net: layer_out = self.dephase_1st_iso_lay_out(layer_out)

        layer_out = self.layers[2].get_2nd_ent_lay_out(layer_out)
        if self.deph_net: layer_out = self.dephase_2nd_ent_lay_out(layer_out)

        layer_out = self.layers[3].get_2nd_iso_lay_out(layer_out)
        if self.deph_net:
            layer_out = tf.expand_dims(layer_out, 1)
            layer_out = self.dephase(layer_out, num_bd=2)
            layer_out = layer_out[:, 0]

        final_layer_out = self.layers[4].get_3rd_iso_lay_out(layer_out)

        if self.num_anc == 1:
            final_layer_out = tf.einsum('zabac -> zbc', tf.reshape(final_layer_out, [batch_size, *[2]*4]))

        output_probs = tf.math.abs(tf.linalg.diag_part(final_layer_out))
        return output_probs

    def dephase_1st_iso_lay_out(self, tensor):
        l, u = Network._lowercases, Network._uppercases
        for i in range(4):
            contract_str = 'X'+u[i]+l[i]+', Z'+l[:8]+', X'+u[4+i]+l[4+i]+' -> Z'+l[:i]+u[i]+l[i+1:4+i]+u[4+i]+l[5+i:8]
            tensor = tf.einsum(contract_str, self.kraus_ops_1_bd, tensor, self.kraus_ops_1_bd)
        return tensor

    def dephase_2nd_ent_lay_out(self, tensor):
        l, u = Network._lowercases, Network._uppercases
        for i in range(2):
            # 'YXWV' are the left-over bonds that do not need to dephase again here
            contract_str = 'U'+u[i]+l[i]+', Z Y'+l[:2]+'XW'+l[2:4]+'V, U'+u[2+i]+l[2+i]+\
                           ' -> Z Y'+l[:i]+u[i]+l[i+1:2]+'XW'+l[2:2+i]+u[2+i]+l[3+i:4]+'V'
            tensor = tf.einsum(contract_str, self.kraus_ops_1_bd, tensor, self.kraus_ops_1_bd)
        return tensor


class Ent_Layer(mera.network.Ent_Layer):
    def __init__(self, num_nodes, layer_idx, num_anc, init_mean, init_std):
        super().__init__(num_nodes, layer_idx, num_anc, init_mean, init_std)

    def get_2nd_ent_lay_out(self, input):
        '''
        :param input: single tensor with canonical indices
        :return: single tensor with canonical indices
        '''
        unitary_tensor = self.get_unitary_tensors()[0]
        contract_str = 'Z YabXWcdV, ABab, CDcd -> Z YABX WCDV'
        output = tf.einsum(contract_str, input, unitary_tensor, tf.math.conj(unitary_tensor))
        return output


class Iso_Layer(mera.network.Iso_Layer):
    def __init__(self, num_nodes, layer_idx, num_anc, init_mean, init_std):
        super().__init__(num_nodes, layer_idx, num_anc, init_mean, init_std)

    def get_2nd_iso_lay_out(self, input):
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

    def get_3rd_iso_lay_out(self, input):
        '''
        :param input: single tensor with cononical indices
        :return: single tensor with cononical indices
        '''
        unitary_tensor = self.get_unitary_tensors()[0]
        output = tf.einsum('ABab, Zabcd, CBcd -> ZAC', unitary_tensor, input, tf.math.conj(unitary_tensor))
        return output


if __name__ == '__main__':
    '''
    Test the contractions of the network by inputting 1/2 I. The output should be 1/2 I. 
    '''
    import yaml
    with open('config_example.yaml', 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    network = Network(8, 0.4, 1, 0.5, 0.005, config)
    identity_input = tf.tile(1/2*tf.eye(2, dtype=tf.complex64)[None, None, :], [1, 8, 1, 1])
    try: out = network.get_network_output(identity_input)
    except: raise Exception('Need to comment out the line to form density matrices from kets')
    print(out)