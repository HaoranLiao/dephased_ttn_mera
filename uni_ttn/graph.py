import numpy as np
import tensorflow.compat.v1 as tf
import itertools as itr
import sys
import model


class Graph():
    def __init__(self, num_pixels, bd_dims, deph_p,
                 deph_only_input, num_anc, embed_dim, batch_size, config):
        self.bd_dims = bd_dims
        self.num_anc = num_anc
        self.create_op_shapes()
        self.num_op_params = max(self.vir_bd_dim, self.data_bd_dim) ** (self.num_in_bd * 2)
        self.num_pixels = num_pixels
        self.deph_p = deph_p
        self.deph_only_input = deph_only_input
        self.embed_dim = embed_dim
        self.config = config
        self.batch_size = batch_size

        self.data_nodes = self.create_data_layer()
        self.num_layers = int(np.log2(self.num_pixels))
        if embed_dim:
            print('Embedded')
            assert self.num_in_bd == 2, 'not supported'
            embedding_config = self.config['tree']['param']['embedding']
            self.not_fir_lay = embedding_config['not_fir_lay']
            if embedding_config['id_lay']['enabled']:
                print('Has Identity Layer')
                self.id_nodes = self.create_identity_layer()
                self.has_id_lay = True
                self.fir_lay_op_shape = self.mid_lay_op_shape
                self.op_shapes[0] = self.mid_lay_op_shape
                self.op_layers = self.create_all_op_nodes(has_id_lay=True)
                self.create_params_var_embed(not_fir_lay=self.not_fir_lay)
            else:
                if self.not_fir_lay:
                    print('No Identity Lay, First Lay Not Embedded')
                else:
                    print('No Identity Layer, First Lay Op Padded Zeros')
                self.op_layers = self.create_all_op_nodes(has_id_lay=False)
                self.create_params_var_embed(not_fir_lay=self.not_fir_lay)
        else:
            print('Not Embedded')
            self.op_layers = self.create_all_op_nodes(has_id_lay=False)
            self.create_params_var()

        self.root_node = self.op_layers[self.num_layers-1][0]
        self.pred_batch = tf.real(tf.matrix_diag_part(self.root_node.output))
        self.label_batch = tf.placeholder(tf.float64, shape=(None, 2))
        self.optimizer()

        self.init = tf.global_variables_initializer()


    def create_op_shapes(self):
        (self.data_bd_dim, self.vir_bd_dim) = self.bd_dims
        self.num_in_bd = 2 + self.num_anc
        self.fir_lay_op_shape = [self.data_bd_dim]*self.num_in_bd + [self.vir_bd_dim]*self.num_in_bd
        self.mid_lay_op_shape = [self.vir_bd_dim]*(self.num_in_bd * 2)
        self.last_lay_op_shape = [self.vir_bd_dim]*self.num_in_bd + [2]*self.num_in_bd
        self.op_shapes = [self.fir_lay_op_shape, self.mid_lay_op_shape, self.last_lay_op_shape]


    def create_data_layer(self):
        data_nodes = []
        for pixel in range(self.num_pixels):
            node = Data_Node(self.batch_size, self.data_bd_dim, self.config)
            data_nodes.append(node)

        return data_nodes


    def create_all_op_nodes(self, has_id_lay=None):
        op_layers = {}
        if has_id_lay:
            op_layers[0] = self.create_op_layer(self.id_nodes, 0, is_prev_id_lay=True)
        else:
            op_layers[0] = self.create_op_layer(self.data_nodes, 0)

        for lay_ind in range(1, self.num_layers):
            prev_layer = op_layers[lay_ind - 1]
            op_layers[lay_ind] = self.create_op_layer(prev_layer, lay_ind)

        return op_layers


    def create_op_layer(self, prev_layer, lay_ind, is_prev_id_lay=False):
        op_layer = []
        if is_prev_id_lay:
            for i in range(0, len(prev_layer)):
                input_id_node = prev_layer[i]
                op_node = Op_Node(input_id_node, lay_ind, self.num_layers, self.op_shapes, self.deph_p,
                                  self.deph_only_input, self.num_in_bd, self.num_anc, self.config,
                                  is_prev_id_lay=True)
                op_layer.append(op_node)
        else:
            for i in range(0, len(prev_layer), 2):
                input_nodes = (prev_layer[i], prev_layer[i+1])
                op_node = Op_Node(input_nodes, lay_ind, self.num_layers, self.op_shapes, self.deph_p,
                                  self.deph_only_input, self.num_in_bd, self.num_anc, self.config)
                op_layer.append(op_node)

        return op_layer


    def create_identity_layer(self):
        id_nodes = []
        for i in range(0, len(self.data_nodes), 2):
            input_nodes = (self.data_nodes[i], self.data_nodes[i+1])
            id_node = Id_Node(input_nodes, self.bd_dims, self.fir_lay_op_shape, self.batch_size, self.config)
            id_nodes.append(id_node)

        return id_nodes


    def create_params_var(self):
        op_layers = self.op_layers.values()
        self.op_nodes = list(itr.chain.from_iterable(op_layers))

        self.opt_config = self.config['tree']['opt']
        self.init_mean = self.config['tree']['param']['init_mean']
        self.init_std = self.config['tree']['param']['init_std']
        if self.opt_config['opt'] == 'sweep':
            for (ind, op_node) in enumerate(self.op_nodes):
                param_var_name = 'param_var_%s'%ind
                self.param_var = tf.get_variable(
                    param_var_name,
                    shape=[self.num_op_params],
                    dtype=tf.float64,
                    initializer=tf.random_normal_initializer(
                        mean=self.init_mean, stddev=self.init_std),
                    trainable=True)
                op_node.create_node_tensor(ind, self.param_var)
        else:
            self.param_var_all = tf.get_variable(
                'param_var_all',
                shape=[self.num_op_params * len(self.op_nodes)],
                dtype=tf.float64,
                initializer=tf.random_normal_initializer(
                    mean=self.init_mean, stddev=self.init_std),
                trainable=True)
            for (ind, op_node) in enumerate(self.op_nodes):
                op_node.create_node_tensor(ind, self.param_var_all)


    def create_params_var_embed(self, not_fir_lay=None):
        assert self.embed_dim < self.vir_bd_dim
        self.embed_op_size = self.embed_dim ** self.num_in_bd
        self.rest_op_size = int(np.sqrt(self.num_op_params) - self.embed_op_size)
        self.num_op_params_embed = self.embed_op_size ** 2
        self.num_op_params_rest = self.rest_op_size ** 2

        op_layers = self.op_layers.values()
        self.op_nodes = list(itr.chain.from_iterable(op_layers))

        self.opt_config = self.config['tree']['opt']
        self.init_mean = self.config['tree']['param']['init_mean']
        self.init_std = self.config['tree']['param']['init_std']
        if self.opt_config['opt'] == 'sweep':
            for (ind, op_node) in enumerate(self.op_nodes):
                param_var_name = 'param_var_%s'%ind
                self.param_var = tf.get_variable(
                    param_var_name,
                    shape=[self.num_op_params],
                    dtype=tf.float64,
                    initializer=tf.random_normal_initializer(
                        mean=self.init_mean, stddev=self.init_std),
                    trainable=True)

            for (ind, op_node) in enumerate(self.op_nodes):
                param_var_embed_name = 'param_var_embed_%s' % ind
                param_var_embed = tf.get_variable(
                    param_var_embed_name,
                    shape=[self.num_op_params_embed],
                    dtype=tf.float64,
                    initializer=tf.random_normal_initializer(
                        mean=self.init_mean, stddev=self.init_std),
                    trainable=True)
                param_var_rest_name = 'param_var_embed_%s' % ind
                param_var_rest = tf.get_variable(
                    param_var_rest_name,
                    shape=[self.num_op_params_rest],
                    dtype=tf.float64,
                    initializer=tf.random_normal_initializer(
                        mean=self.init_mean, stddev=self.init_std),
                    trainable=True)
                op_node.create_embedded_node_tensor(ind, param_var_embed, param_var_rest,
                                                    self.embed_op_size, self.rest_op_size,
                                                    self.param_var,
                                                    not_fir_lay=not_fir_lay)
        else:
            self.param_var_all = tf.get_variable(
                'param_var_all',
                shape=[self.num_op_params * len(self.op_nodes)],
                dtype=tf.float64,
                initializer=tf.random_normal_initializer(
                    mean=self.init_mean, stddev=self.init_std),
                trainable=True)

            self.param_var_embed_all = tf.get_variable(
                'param_var_embed_all',
                shape=[self.num_op_params_embed * len(self.op_nodes)],
                dtype=tf.float64,
                initializer=tf.random_normal_initializer(
                    mean=self.init_mean, stddev=self.init_std),
                trainable=True)
            self.param_var_rest_all = tf.get_variable(
                'param_var_rest_all',
                shape=[self.num_op_params_rest * len(self.op_nodes)],
                dtype=tf.float64,
                initializer=tf.random_normal_initializer(
                    mean=self.init_mean, stddev=self.init_std),
                trainable=True)
            for (ind, op_node) in enumerate(self.op_nodes):
                op_node.create_embedded_node_tensor(ind, self.param_var_embed_all,
                                                    self.param_var_rest_all, self.embed_op_size,
                                                    self.rest_op_size, self.param_var_all,
                                                    not_fir_lay=not_fir_lay)


    def optimizer(self):
        self.loss_config = self.config['tree']['loss']
        if self.loss_config == 'l2':
            self.loss = tf.reduce_sum(tf.square(self.pred_batch - self.label_batch))
            print('L2 Loss')
        elif self.loss_config == 'l1':
            self.loss = tf.losses.absolute_difference(self.label_batch, self.pred_batch)
            print('L1 Loss')
        elif self.loss_config == 'log':
            self.loss = tf.losses.log_loss(self.label_batch, self.pred_batch)
            print('Log Loss')
        else:
            raise Exception('Invalid Loss')

        if self.opt_config['opt'] == 'adam':
            opt = tf.train.AdamOptimizer()
            self.grad_var = opt.compute_gradients(self.loss)
            self.train_op = opt.apply_gradients(self.grad_var)
            print('Adam Optimizer')
        elif self.opt_config['opt'] == 'sgd':
            step_size = self.opt_config['sgd']['step_size']
            self.train_op = tf.train.GradientDescentOptimizer(step_size).minimize(self.loss)
            print('SGD Optimizer')
        elif self.opt_config['opt'] == 'sweep':
            self.train_ops = [0]*len(self.op_nodes)
            for (ind, op_node) in enumerate(self.op_nodes):
                self.train_ops[ind] = tf.train.AdamOptimizer().minimize(self.loss, var_list=[op_node.param_var])
            print('Sweep with Adam Optimizer')
        elif self.opt_config['opt'] == 'rmsprop':
            learning_rate = self.opt_config['rmsprop']['learning_rate']
            self.train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
            print('RMSProp Optimizer')
        else:
            raise Exception('Invalid Optimizer')

        sys.stdout.flush()


    def run_graph(self, sess, image_batch):
        fd_dict = self.create_pixel_dict(image_batch)
        pred_batch = sess.run(self.pred_batch, feed_dict=fd_dict)
        return pred_batch


    def create_pixel_dict(self, image_batch):
        if self.config['data']['data_im_size'] == [8, 8]:
            pixel_dict = {}
            for (index, node) in enumerate(self.data_nodes):
                quad = index // 16
                quad_quad = (index % 16) // 4
                pos = index % 4
                row = (pos // 2) + 2 * (quad_quad // 2) + 4 * (quad // 2)
                col = (pos % 2) + 2 * (quad_quad % 2) + 4 * (quad % 2)
                pixel = col + 8 * row
                pixel_dict[node.pixel_batch] = image_batch[:, pixel, :]
        else:
            pixel_dict = {
                node.pixel_batch: image_batch[:, pixel, :] for (pixel, node) in enumerate(self.data_nodes)}
            
        return pixel_dict


    def train(self, sess, image_batch, label_batch):
        fd_dict = self.create_pixel_dict(image_batch)
        fd_dict.update({self.label_batch: label_batch})

        self.avg_grad, self.std_grad = None, None
        if self.opt_config['opt'] == 'sweep':
            for (ind, __) in enumerate(self.op_nodes):
                sess.run(self.train_ops[ind], feed_dict=fd_dict)
                if self.opt_config['sweep']['inspection']:
                    self.run_graph(sess, image_batch)
                    pred_batch = self.run_graph(sess, image_batch)
                    print('%s/%s-%.3f'%(ind, len(self.op_nodes), model.get_accuracy(pred_batch, label_batch)))
                    sys.stdout.flush()
                else:
                    pass
        else:
            sess.run(self.train_op, feed_dict=fd_dict)
            if self.opt_config['adam']['show_grad']:
                for gv in self.grad_var:
                    self.avg_grad = round(float(np.mean(sess.run(gv[0], feed_dict=fd_dict))), 3)
                    self.std_grad = round(float(np.std(sess.run(gv[0], feed_dict=fd_dict))), 3)
            else:
                pass

            if self.opt_config['adam']['show_hess']:
                real_off_params = []
                for (ind, op_node) in enumerate(self.op_nodes):
                    if ind == 0:
                        real_off_params = op_node.real_off_params

                self.hess = tf.hessians(self.loss, real_off_params)
                self.hess_val = sess.run(self.hess, feed_dict=fd_dict)
                pass
            else:
                pass


class Data_Node():
    def __init__(self, batch_size, data_bd_dim, config):
        self.data_bd_dim = data_bd_dim
        self.batch_size = batch_size
        if config['tree']['param']['embedding']['id_lay']['id_lay_kron']:
            assert batch_size == config['data']['sample_size']
            self.pixel_batch = tf.placeholder(tf.complex128, shape=(self.batch_size, self.data_bd_dim))
        else:
            self.pixel_batch = tf.placeholder(tf.complex128, shape=(None, self.data_bd_dim))

        conj_pixel_batch = tf.conj(self.pixel_batch)
        self.output = tf.einsum('za, zb -> zab', self.pixel_batch, conj_pixel_batch)


class Id_Node():
    def __init__(self, input_nodes, bd_dims, fir_lay_op_shape, batch_size, config):
        self.batch_size = batch_size
        (self.data_bd_dim, self.vir_bd_dim) = bd_dims
        (self.left_data_node, self.right_data_node) = input_nodes
        self.left_in_bat = self.left_data_node.output
        self.right_in_bat = self.right_data_node.output
        concat_num = int((self.vir_bd_dim ** 2) / (self.data_bd_dim ** 2))
        self.id_matrix = tf.concat([tf.eye(self.data_bd_dim ** 2, dtype=tf.complex128)]*concat_num, 0)

        if config['tree']['param']['embedding']['id_lay']['id_lay_kron']:
            self.id_matrix = tf.cast(self.id_matrix, dtype=tf.float64)
            self.output = []
            for j in range(self.batch_size):
                a = self.left_in_bat[j, :, :]
                b = self.right_in_bat[j, :, :]
                i, k, s = int(a.shape[0]), int(b.shape[0]), int(b.shape[0])
                o = s * (i - 1) + k
                a_tf = tf.cast(tf.reshape(a, [1, i, i, 1]), dtype=tf.float64)
                b_tf = tf.cast(tf.reshape(b, [k, k, 1, 1]), dtype=tf.float64)
                self.input_mat = tf.squeeze(
                    tf.nn.conv2d_transpose(a_tf, b_tf, (1, o, o, 1), [1, s, s, 1], "VALID")
                )
                self.contract = tf.einsum('ab, ca, db -> cd', self.input_mat, self.id_matrix, self.id_matrix)
                self.output.append(tf.cast(
                    tf.reshape(self.contract, [self.vir_bd_dim] * 4),
                    dtype=tf.complex128))

            self.output = tf.convert_to_tensor(self.output)
        else:
            self.id_tensor = tf.reshape(tf.transpose(self.id_matrix), fir_lay_op_shape)
            self.contract_left = tf.einsum('zab, acde -> zbcde', self.left_in_bat, self.id_tensor)
            self.contract_right = tf.einsum('zbcde, zcf -> zbfde', self.contract_left, self.right_in_bat)
            self.output = tf.einsum('zbfde, bfgh -> zdegh', self.contract_right, tf.conj(self.id_tensor))



class Op_Node():
    def __init__(self, input_nodes, lay_ind, num_layers,
                 op_shapes, deph_p, deph_only_input, num_in_bd, num_anc, config, is_prev_id_lay=False):
        self.input_nodes = input_nodes
        self.lay_ind = lay_ind
        self.num_layers = num_layers
        (self.fir_lay_op_shape, self.mid_lay_op_shape, self.last_lay_op_shape) = op_shapes
        self.fir_lay_mat_shape = self.get_mat_shapes(self.fir_lay_op_shape)
        self.last_lay_mat_shape = self.get_mat_shapes(self.last_lay_op_shape)
        if self.fir_lay_op_shape[0] > self.mid_lay_op_shape[0]:
            self.max_op_size = self.fir_lay_op_shape[0] ** num_in_bd
            if lay_ind != 0:
                self.op_size = self.mid_lay_op_shape[0] ** num_in_bd
            else:
                assert lay_ind == 0
                self.op_size = self.fir_lay_op_shape[0] ** num_in_bd
        else:
            self.op_size = self.mid_lay_op_shape[0] ** num_in_bd
            self.max_op_size = self.op_size

        self.deph_p = deph_p
        self.deph_only_input = deph_only_input
        self.num_anc = num_anc
        self.config = config
        self.is_prev_id_lay = is_prev_id_lay


    def get_mat_shapes(self, op_shape):
        num_input = len(op_shape)/2
        mat_shape = [int(op_shape[-1] ** num_input),
                     int(op_shape[0] ** num_input)]
        return mat_shape


    def create_node_tensor(self, index, param_var):
        self.index = index
        with tf.variable_scope(str(index)):
            self.set_matrix_params(param_var, self.op_size, self.max_op_size)
            self.create_hermitian_matrix(self.op_size)
            unitary_matrix_raw = self.create_unitary_matrix()
            self.create_unitary_tensor(unitary_matrix_raw)
            self.create_contractions()


    def create_embedded_node_tensor(self, index, param_var_embed, param_var_rest,
                                    embed_op_size, rest_op_size, param_var,
                                    not_fir_lay=None):
        self.index = index
        with tf.variable_scope(str(index)):
            self.set_matrix_params(param_var_embed, embed_op_size, embed_op_size)
            self.create_hermitian_matrix(embed_op_size)
            uni_mat_embed = self.create_unitary_matrix()
            try:
                if not_fir_lay:
                    if self.lay_ind == 0:
                        self.create_node_tensor(index, param_var)
                        return
                    else:
                        pad = tf.constant([[0, rest_op_size], [0, 0]])
                        uni_mat_embed_pad = tf.pad(uni_mat_embed, pad, 'CONSTANT')
                else:
                    pad = tf.constant([[0, rest_op_size], [0, 0]])
                    uni_mat_embed_pad = tf.pad(uni_mat_embed, pad, 'CONSTANT')
            except ValueError:
                print('Check if Rest Block is of Size 1')

            self.set_matrix_params(param_var_rest, rest_op_size, rest_op_size)
            self.create_hermitian_matrix(rest_op_size)
            uni_mat_rest = self.create_unitary_matrix()
            try:
                pad = tf.constant([[embed_op_size, 0], [0, 0]])
                uni_mat_rest_pad = tf.pad(uni_mat_rest, pad, 'CONSTANT')
            except ValueError:
                print('Check if Rest Block is of Size 1')

            unitary_matrix_raw = tf.concat([uni_mat_embed_pad, uni_mat_rest_pad], 1)
            self.create_unitary_tensor(unitary_matrix_raw)
            self.create_contractions()


    def set_matrix_params(self, param_var, op_size, max_op_size):
        num_diags = op_size
        num_off_diags = int(0.5 * op_size * (op_size - 1))
        max_total_params = max_op_size ** 2
        if self.config['tree']['opt']['opt'] == 'sweep':
            start_slice = 0
        else:
            start_slice = self.index * max_total_params
        diag_end = start_slice + num_diags
        real_end = diag_end + num_off_diags
        self.diag_params = tf.slice(param_var, [start_slice], [num_diags])
        self.real_off_params = tf.slice(param_var, [diag_end], [num_off_diags])
        self.imag_off_params = tf.slice(param_var, [real_end], [num_off_diags])


    def create_hermitian_matrix(self, op_size):
        herm_shape = (op_size, op_size)
        diag_part = tf.diag(self.diag_params)
        off_diag_indices = [(i, j) for i in range(op_size) for j in range(i+1, op_size)]
        real_off_diag_part = tf.scatter_nd(
            indices=off_diag_indices,
            updates=self.real_off_params,
            shape=herm_shape)
        imag_off_diag_part = tf.scatter_nd(
            indices=off_diag_indices,
            updates=self.imag_off_params,
            shape=herm_shape)
        imag_whole = imag_off_diag_part - tf.transpose(imag_off_diag_part)
        real_whole = diag_part + real_off_diag_part + tf.transpose(real_off_diag_part)
        self.herm_matrix = tf.complex(real_whole, imag_whole)


    def create_unitary_matrix(self):
        (eigenvalues, eigenvectors) = tf.linalg.eigh(self.herm_matrix)
        eig_exp = tf.exp(1j * eigenvalues)
        diag_exp_mat = tf.diag(eig_exp)
        unitary_matrix_raw = tf.einsum(
            'ab, bc, dc -> ad',
            eigenvectors,
            diag_exp_mat,
            tf.conj(eigenvectors))

        return unitary_matrix_raw


    def create_unitary_tensor(self, unitary_matrix_raw):
        if self.lay_ind == 0:   # if is first op layer
            if self.fir_lay_op_shape[0] <= self.mid_lay_op_shape[0]:
                # if data bd dim smaller or equal to vir bd dim
                self.unitary_tensor = tf.reshape(
                    tf.transpose(
                        tf.slice(
                            unitary_matrix_raw, [0, 0], self.fir_lay_mat_shape)
                    ),
                    self.fir_lay_op_shape
                )
                self.op_shape = self.fir_lay_op_shape
            else:
                self.unitary_tensor = tf.reshape(
                    tf.slice(unitary_matrix_raw, [0, 0], list(reversed(self.fir_lay_mat_shape))),
                self.fir_lay_op_shape
                )
                self.op_shape = self.fir_lay_op_shape
        elif self.lay_ind == (self.num_layers-1):
            self.unitary_tensor = tf.reshape(
                tf.transpose(
                    tf.slice(
                        unitary_matrix_raw, [0, 0], self.last_lay_mat_shape)
                ),
                self.last_lay_op_shape
            )
            self.op_shape = self.last_lay_op_shape
        else:
            assert self.lay_ind > 0 and self.lay_ind < (self.num_layers-1)
            self.unitary_tensor = tf.reshape(unitary_matrix_raw, self.mid_lay_op_shape)
            self.op_shape = self.mid_lay_op_shape


    def create_contractions(self):
        if self.is_prev_id_lay:
            assert self.deph_p == 0, 'not supported'
            assert self.num_anc == 0, 'not supported'
            self.input_id_node = self.input_nodes
            self.input = self.input_id_node.output
            self.contract_with_id = tf.einsum('zabcd, abef -> zcdef', self.input, self.unitary_tensor)
            self.output = tf.einsum(
                'zcdef, cdgf -> zeg', self.contract_with_id, tf.conj(self.unitary_tensor))
        else:
            (left_node, right_node) = self.input_nodes
            if self.deph_only_input:
                if self.lay_ind == 0:
                    left_input = self.dephase(left_node.output, self.deph_p)
                    right_input = self.dephase(right_node.output, self.deph_p)
                else:
                    assert self.lay_ind > 0
                    left_input = self.dephase(left_node.output, 0)
                    right_input = self.dephase(right_node.output, 0)

            else:
                left_input = self.dephase(left_node.output, self.deph_p)
                right_input = self.dephase(right_node.output, self.deph_p)

            if self.num_anc == 0:
                contract_left = tf.einsum('abcd, zea -> zebcd', self.unitary_tensor, left_input)
                contract_right = tf.einsum('zebcd, zfb -> zefcd', contract_left, right_input)
                self.output = tf.einsum('zefcd, efcg -> zdg', contract_right, tf.conj(self.unitary_tensor))
            elif self.num_anc == 1:
                indices = [-1] * self.op_shape[2]
                indices[0] = 1
                ancilla = tf.one_hot(
                    indices=indices,
                    depth=self.op_shape[2],
                    dtype=tf.complex128)
                uni_and_anc = tf.einsum('abcdef, ag -> gbcdef', self.unitary_tensor, ancilla)
                contract_left = tf.einsum('gbcdef, zhb -> zghcdef', uni_and_anc, left_input)
                contract_right = tf.einsum('zghcdef, zic -> zghidef', contract_left, right_input)
                self.output = tf.einsum('zghidef, ghidej -> zfj', contract_right, tf.conj(uni_and_anc))
            else:
                raise Exception('Invalid Ancilla Number')


    def dephase(self, rho, p=1):
        # this is only true for complete dephasing.
        dephased_rho = (1 - p) * rho + p * tf.matrix_diag(tf.matrix_diag_part(rho))
        return dephased_rho
