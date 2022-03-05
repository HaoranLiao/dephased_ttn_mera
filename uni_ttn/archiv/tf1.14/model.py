import tensorflow as tf
import numpy as np
import sys, data, os, time, yaml, json
import graph
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def invoke(config):
    global data_path
    global val_split
    global num_repeat
    global num_anc

    data_path = config['data']['path']
    val_split = config['data']['val_split']
    num_repeat = config['meta']['num_repeat']
    num_anc = config['tree']['num_anc']

    global list_digits
    global list_vir_bd_dim
    global list_deph
    global list_batch_sizes
    global list_epochs
    global list_data_bd_dim
    global list_length
    global list_embed_dims

    list_digits = config['data']['list_digits']
    list_vir_bd_dim = config['tree']['list_vir_bd_dim']
    list_deph = config['tree']['deph']['list_deph']
    list_batch_sizes = config['data']['list_batch_sizes']
    list_epochs = config['meta']['list_epochs']
    list_data_bd_dim = config['data']['list_data_bd_dim']

    list_length = max(len(list_digits), len(list_vir_bd_dim), len(list_deph),
                      len(list_batch_sizes), len(list_epochs), len(list_data_bd_dim))

    main(config)


def main(config):
    start = time.time()
    try:
        for i in range(list_length):
            run_all(list_batch_sizes, i, config)
    finally:
        print_results(start)


def print_results(start):
    end = time.time()
    print('All Avg Test Accs:\n', diff_avg_test_acc)
    print('All Avg Train/Val Accs:\n', diff_avg_train_acc)
    print('All Std Test Accs:\n', diff_std_test_acc)
    print('All Std Train/Val Accs:\n', diff_std_train_acc)
    print('Time: %.1f' % (end - start))
    sys.stdout.flush()


def variable_or_uniform(input, i):
    if len(input) > 1:
        assert list_length > 1; return input[i]
    else:
        return input[0]


diff_avg_test_acc, diff_avg_train_acc = [], []
diff_std_test_acc, diff_std_train_acc = [], []


def run_all(list_batch_sizes, i, config):
    digits = variable_or_uniform(list_digits, i)
    deph = variable_or_uniform(list_deph, i)
    vir_bd_dim = variable_or_uniform(list_vir_bd_dim, i)
    data_bd_dim = variable_or_uniform(list_data_bd_dim, i)
    if data_bd_dim != 2: assert config['data']['load_from_file'] is False
    bd_dims = (data_bd_dim, vir_bd_dim)
    epochs = variable_or_uniform(list_epochs, i)
    batch_size = variable_or_uniform(list_batch_sizes, i)

    auto_epochs = config['meta']['auto_epochs']['enabled']
    deph_only_input = config['tree']['deph']['only_input']

    test_accs, train_accs = [], []
    for j in range(num_repeat):
        start = time.time()
        print('\nRepeat: %s/%s' % (j + 1, num_repeat))
        print('Digits:\t', digits)

        if deph:
            assert deph == 1
            print('Dephasing: %.2f' % deph)
            if deph_only_input: print('Dephase only input')
            else: print('Dephase all')
        else:
            print('Dephase none')

        print('Vir Bond Dim: %s' % vir_bd_dim)
        print('Data Bond Dim: %s' % data_bd_dim)
        print('Num Anc: %s' % num_anc)
        print('Auto Epochs', auto_epochs)
        print('Batch Size: %s' % batch_size)
        sys.stdout.flush()

        model = Model(data_path, digits, val_split, bd_dims,
                      deph, deph_only_input, num_anc, batch_size, config)
        (test_acc, train_acc) = model.train_network(epochs, batch_size, auto_epochs)

        test_accs.append(test_acc)
        train_accs.append(train_acc)
        end = time.time()
        print('Time: %.1f' % (end - start))
        sys.stdout.flush()

    print('\nTrain Accs:\n', train_accs)
    print('Avg Train Acc: %.3f' % np.mean(train_accs))
    print('Std Train Acc: %.3f' % np.std(train_accs))
    print('Test Accs:\n', test_accs)
    print('Avg Test Acc: %.3f' % np.mean(test_accs))
    print('Std Test Acc: %.3f' % np.std(test_accs))
    sys.stdout.flush()

    diff_avg_test_acc.append(round(float(np.mean(test_accs)), 4))
    diff_avg_train_acc.append(round(float(np.mean(train_accs)), 4))
    diff_std_test_acc.append(round(float(np.std(test_accs)), 4))
    diff_std_train_acc.append(round(float(np.std(train_accs)), 4))


class Model:
    def __init__(self, data_path, digits, val_split, bd_dims,
                 deph, deph_only_input, num_anc, batch_size, config):
        sample_size = config['data']['sample_size']
        data_im_size = config['data']['data_im_size']
        if config['data']['load_from_file']:
            assert data_im_size == [8, 8]
            (train_data, val_data, test_data) = data.get_data_file(
                data_path, digits, val_split, sample_size=sample_size)
        else:
            (data_bd_dim, __) = bd_dims
            (train_data, val_data, test_data) = data.get_data_web(
                digits, val_split, data_im_size, data_bd_dim, sample_size=sample_size)

        (self.train_images, self.train_labels) = train_data
        print('Sample Size: %s' % self.train_images.shape[0])

        if val_data is not None:
            print('Validation Split: %.2f' % val_split)
            (self.val_images, self.val_labels) = val_data
        else:
            assert config['data']['val_split'] == 0
            print('No Validation')
        sys.stdout.flush()

        (self.test_images, self.test_labels) = test_data
        num_pixels = self.train_images.shape[1]
        self.config = config

        # self.train_images = np.vstack([self.train_images[0, 0] for _ in range(4)])[None, :, :]
        # self.test_images = np.vstack([self.test_images[0, 0] for _ in range(4)])[None, :, :]

        tf.reset_default_graph()
        self.graph = graph.Graph(num_pixels, bd_dims, deph,
                                 deph_only_input, num_anc, batch_size, config)

    def train_network(self, epochs, batch_size, auto_epochs):
        settings = tf.ConfigProto(log_device_placement=self.config['meta']['log_device_placement'])
        if self.config['meta']['gpu']:
            settings.gpu_options.allow_growth = True
            sess = tf.Session(config=settings)
        else:
            sess = tf.Session()

        if self.config['meta']['list_devices']:
            sess.list_devices()

        sess.run(self.graph.init)

        self.temp_acc = []
        for epoch in range(epochs):
            sys.stdout.flush()
            accuracy = self.run_epoch(sess, batch_size)
            if self.graph.opt_config['adam']['show_grad']:
                last_bat_avg_grad = self.graph.avg_grad
                last_bat_std_grad = self.graph.std_grad
                print('%s/%s : %.3f\t\t%s\t%s' %
                      (epoch + 1, epochs, accuracy, last_bat_avg_grad, last_bat_std_grad))
                print(f'loss: {self.graph.loss_}')
                print(f'pred_batch: {self.graph.pred_batch_}')
                print(f'label_batch: {self.graph.label_batch_}')
                print(f'output_density_mat: {self.graph.root_node.output_}')
                print(f'all_grads: {self.graph.all_grads}')
            else:
                print('%s/%s : %.3f' % (epoch + 1, epochs, accuracy))
            sys.stdout.flush()
            self.temp_acc.append(accuracy)

            if auto_epochs:
                trigger = self.config['meta']['auto_epochs']['trigger']
                assert trigger < epochs
                if epoch >= trigger and self.check_val_acc_satified(accuracy):
                    train_or_val_accuracy = accuracy; break
                elif epoch == epochs - 1: train_or_val_accuracy = accuracy
                else: continue

            else:
                if epoch == epochs - 1: train_or_val_accuracy = accuracy
                else: continue

        if not val_split: print('Train Accuracy: %.3f' % train_or_val_accuracy)
        else: print('Validation Accuracy: %.3f' % train_or_val_accuracy)

        test_accuracy = self.run_test_data(sess)
        print('Test Accuracy : {:.3f}'.format(test_accuracy))
        sys.stdout.flush()
        sess.close()

        return test_accuracy, train_or_val_accuracy

    def check_val_acc_satified(self, accuracy):
        criterion = self.config['meta']['auto_epochs']['criterion']
        for i in range(self.config['meta']['auto_epochs']['num_match']):
            if abs(accuracy - self.temp_acc[-(i + 2)]) <= criterion: continue
            else: return False

        return True

    def run_epoch(self, sess, batch_size):
        batch_iter = data.batch_generator(self.train_images, self.train_labels, batch_size)
        for (train_image_batch, train_label_batch) in batch_iter:
            self.graph.train(sess, train_image_batch, train_label_batch)

        if val_split:
            assert self.config['data']['val_split'] > 0
            pred_probs = self.graph.run_graph(sess, self.val_images)
            val_accuracy = get_accuracy(pred_probs, self.val_labels)
            return val_accuracy
        else:
            pred_probs = self.graph.run_graph(sess, self.train_images)
            train_accuracy = get_accuracy(pred_probs, self.train_labels)
            return train_accuracy

    def run_test_data(self, sess):
        test_results = self.graph.run_graph(sess, self.test_images)
        test_accuracy = get_accuracy(test_results, self.test_labels)
        return test_accuracy


def get_accuracy(guesses, labels):
    guess_index = np.argmax(guesses, axis=1)
    label_index = np.argmax(labels, axis=1)
    compare = guess_index - label_index
    num_correct = float(np.sum(compare == 0))
    total = float(guesses.shape[0])
    accuracy = num_correct / total
    return accuracy


if __name__ == "__main__":
    with open('config_example.yaml', 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
        print(json.dumps(config, indent=2))
        sys.stdout.flush()

    np.random.seed(45)
    tf.random.set_random_seed(45)

    invoke(config)
