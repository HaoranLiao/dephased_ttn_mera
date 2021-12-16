import tensorflow as tf
import numpy as np
import sys, os, time, yaml, json
from tqdm import tqdm
import network
import data
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
TQDM_DISABLED = False


def print_results(start_time):
    print('All Avg Test Accs:\n', avg_repeated_test_acc)
    print('All Avg Train/Val Accs:\n', avg_repeated_train_acc)
    print('All Std Test Accs:\n', std_repeated_test_acc)
    print('All Std Train/Val Accs:\n', std_repeated_train_acc)
    print('Time: %.1f' % (time.time() - start_time))
    sys.stdout.flush()


def variable_or_uniform(input, i):
    if len(input) > 1: return input[i]
    else: return input[0]


def run_all(i):
    digits = variable_or_uniform(list_digits, i)
    epochs = variable_or_uniform(list_epochs, i)
    batch_size = variable_or_uniform(list_batch_sizes, i)
    deph_p = variable_or_uniform(list_deph_p, i)
    num_anc = variable_or_uniform(list_num_anc, i)

    auto_epochs = config['meta']['auto_epochs']['enabled']
    test_accs, train_accs = [], []
    for j in range(num_repeat):
        start_time = time.time()
        print('\nRepeat: %s/%s' % (j + 1, num_repeat))
        print('Digits:\t', digits)
        print('Dephasing data', config['meta']['deph']['data'])
        print('Dephasing network', config['meta']['deph']['network'])
        print('Dephasing rate %.2f' % deph_p)
        print('Auto Epochs', auto_epochs)
        print('Batch Size: %s' % batch_size)
        print('Number of Ancillas: %s' % num_anc)
        sys.stdout.flush()

        model = Model(data_path, digits, val_split, deph_p, num_anc, config)
        test_acc, train_acc = model.train_network(epochs, batch_size, auto_epochs)

        test_accs.append(round(test_acc, 4))
        train_accs.append(round(train_acc, 4))
        print('Time: %.1f' % (time.time() - start_time)); sys.stdout.flush()

    print(f'\nSetting {i} Train Accs: {train_accs}\t')
    print('Setting %d Avg Train Acc: %.3f' % (i, np.mean(train_accs)))
    print('Setting %d Std Train Acc: %.3f' % (i, np.std(train_accs)))
    print(f'Setting {i} Test Accs: {test_accs}\t')
    print('Setting %d Avg Test Acc: %.3f' % (i, np.mean(test_accs)))
    print('Setting %d Std Test Acc: %.3f' % (i, np.std(test_accs)))
    sys.stdout.flush()

    avg_repeated_test_acc.append(round(float(np.mean(test_accs)), 3))
    avg_repeated_train_acc.append(round(float(np.mean(train_accs)), 3))
    std_repeated_test_acc.append(round(float(np.std(test_accs)), 3))
    std_repeated_train_acc.append(round(float(np.std(train_accs)), 3))


class Model:
    def __init__(self, data_path, digits, val_split, deph_p, num_anc, config):
        sample_size = config['data']['sample_size']
        data_im_size = config['data']['data_im_size']
        feature_dim = config['data']['feature_dim']
        if config['data']['load_from_file']:
            assert data_im_size == [8, 8] and feature_dim == 2
            train_data, val_data, test_data = data.get_data_file(
                data_path, digits, val_split, sample_size=sample_size)
        else:
            train_data, val_data, test_data = data.get_data_web(
                digits, val_split, data_im_size, feature_dim, sample_size=sample_size)

        self.train_images, self.train_labels = train_data
        # self.train_images = tf.constant(self.train_images, dtype=tf.complex64)
        # self.train_labels = tf.constant(self.train_labels, dtype=tf.float32)
        print('Sample Size: %s' % self.train_images.shape[0])

        if val_data is not None:
            print('Validation Split: %.2f' % val_split)
            self.val_images, self.val_labels = val_data
            # self.val_images = tf.constant(self.val_images, dtype=tf.complex64)
            # self.val_labels = tf.constant(self.val_labels, dtype=tf.float32)
        else:
            assert config['data']['val_split'] == 0
            print('No Validation')
        sys.stdout.flush()

        self.test_images, self.test_labels = test_data
        # self.test_images = tf.constant(self.test_images, dtype=tf.complex64)
        # self.test_labels = tf.constant(self.test_labels, dtype=tf.float32)

        num_pixels = self.train_images.shape[1]
        self.config = config
        self.network = network.Network(num_pixels, deph_p, num_anc, config)

    def train_network(self, epochs, batch_size, auto_epochs):
        if self.config['meta']['list_devices']: tf.config.list_physical_devices()
        sys.stdout.flush()

        self.epoch_acc = []
        for epoch in range(epochs):
            accuracy = self.run_epoch(batch_size)
            print('Epoch %d: %.5f accuracy' % (epoch, accuracy)); sys.stdout.flush()

            if epoch%5 == 0:
                test_accuracy = self.run_network(self.test_images, self.test_labels, batch_size)
                print('Test Accuracy : {:.3f}'.format(test_accuracy)); sys.stdout.flush()

            self.epoch_acc.append(accuracy)
            if auto_epochs:
                trigger = self.config['meta']['auto_epochs']['trigger']
                assert trigger < epochs
                if epoch >= trigger and self.check_acc_satified(accuracy): break

        train_or_val_accuracy = accuracy
        if not val_split: print('Train Accuracy: %.3f' % train_or_val_accuracy)
        else: print('Validation Accuracy: %.3f' % train_or_val_accuracy)
        sys.stdout.flush()

        test_accuracy = self.run_network(self.test_images, self.test_labels, batch_size)
        print('Test Accuracy : {:.3f}'.format(test_accuracy)); sys.stdout.flush()
        return test_accuracy, train_or_val_accuracy

    def run_network(self, images, labels, batch_size):
        num_correct = 0
        batch_iter = data.batch_generator_np(images, labels, batch_size)
        for (image_batch, label_batch) in tqdm(batch_iter, leave=False, disable=TQDM_DISABLED):
            pred_probs = self.network.get_network_output(image_batch)
            num_correct += get_accuracy(pred_probs, label_batch)[1]
        accuracy = num_correct / images.shape[0]
        return accuracy

    def check_acc_satified(self, accuracy):
        criterion = self.config['meta']['auto_epochs']['criterion']
        for i in range(self.config['meta']['auto_epochs']['num_match']):
            if abs(accuracy - self.epoch_acc[-(i + 2)]) <= criterion: continue
            else: return False
        return True

    def run_epoch(self, batch_size):
        batch_iter = data.batch_generator_np(self.train_images, self.train_labels, batch_size)
        for (train_image_batch, train_label_batch) in tqdm(batch_iter, leave=False, disable=TQDM_DISABLED):
            self.network.update(train_image_batch, train_label_batch)

        if val_split:
            assert self.config['data']['val_split'] > 0
            val_accuracy = self.run_network(self.val_images, self.val_labels, batch_size)
            return val_accuracy
        else:
            train_accuracy = self.run_network(self.train_images, self.train_labels, batch_size)
            return train_accuracy


def get_accuracy(guesses, labels):
    guess_index = np.argmax(guesses, axis=1)
    label_index = np.argmax(labels, axis=1)
    compare = guess_index - label_index
    num_correct = float(np.sum(compare == 0))
    total = float(guesses.shape[0])
    accuracy = num_correct / total
    return accuracy, num_correct


if __name__ == "__main__":
    with open('config_example.yaml', 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
        print(json.dumps(config, indent=1)); sys.stdout.flush()

    data_path = config['data']['path']
    val_split = config['data']['val_split']
    list_batch_sizes = config['data']['list_batch_sizes']
    list_digits = config['data']['list_digits']
    num_repeat = config['meta']['num_repeat']
    list_epochs = config['meta']['list_epochs']
    list_deph_p = config['meta']['deph']['p']
    list_num_anc = config['meta']['list_num_anc']

    num_settings = max(len(list_digits), len(list_num_anc),
                       len(list_batch_sizes), len(list_epochs), len(list_deph_p))

    avg_repeated_test_acc, avg_repeated_train_acc = [], []
    std_repeated_test_acc, std_repeated_train_acc = [], []

    start_time = time.time()
    try:
        for i in range(num_settings): run_all(i)
    finally:
        print_results(start_time)
