import tensorflow as tf

class Spsa():
    _name = 'Spsa'

    def __init__(self, network, hparams):
        self.a = hparams['a']
        self.b = hparams['b']
        self.A = hparams['A']
        self.s = hparams['s']
        self.t = hparams['t']
        self.gamma = hparams['gamma']
        self.network = network
        self.perturbs = [tf.zeros(lay_var.shape) for lay_var in network.var_list]
        self.old_update = [tf.zeros(lay_var.shape) for lay_var in network.var_list]

    def anneal_hparams(self, epoch):
        a_anneal = (epoch + 1 + self.A) ** self.s
        b_anneal = (epoch + 1) ** self.t
        alpha = self.a / a_anneal
        beta = self.b / b_anneal
        return alpha, beta

    def get_update(self, epoch, input_batch, label_batch):
        alpha, beta = self.anneal_hparams(epoch)
        bern_perturbs = [2 * tf.round(tf.random.uniform(pert.shape)) - 1 for pert in self.perturbs]
        perturbs = [alpha * bern_pert for bern_pert in bern_perturbs]
        num_layers = len(bern_perturbs)
        for i in range(num_layers): self.network.var_list[i].assign_add(perturbs[i])
        plus_loss = self.network.loss(input_batch, label_batch)
        for i in range(num_layers): self.network.var_list[i].assign_add(-2 * perturbs[i])
        minus_loss = self.network.loss(input_batch, label_batch)
        for i in range(num_layers): self.network.var_list[i].assign_add(perturbs[i])
        g = [(plus_loss - minus_loss) / (2 * alpha * bern_pert) for bern_pert in bern_perturbs]
        new_update = [self.gamma * self.old_update[i] - beta * g[i] for i in range(num_layers)]
        self.old_update = new_update
        return new_update

    def apply_gradients(self, zipped):
        for grads, vars in zipped: vars.assign_add(grads)

