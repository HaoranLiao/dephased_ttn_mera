import tensorflow as tf
import numpy as np
import sys, data, os, time, yaml, json
import network
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    start_time = time.time()
    try:
        for i in range(num_settings): run_all(i)
    finally:
        print_results(start_time)


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


avg_repeated_test_acc, avg_repeated_train_acc = [], []
std_repeated_test_acc, std_repeated_train_acc = [], []


def run_all(i):
    digits = variable_or_uniform(list_digits, i)
    bd_dim = variable_or_uniform(list_bd_dims, i)
    if bd_dim != 2: assert config['data']['load_from_file'] is False
    epochs = variable_or_uniform(list_epochs, i)
    batch_size = variable_or_uniform(list_batch_sizes, i)

    auto_epochs = config['meta']['auto_epochs']['enabled']
    deph_data = config['meta']['deph']['data']
    deph_net = config['meta']['deph']['network']
    val_split = config['data']['val_split']

    test_accs, train_accs = [], []
    for j in range(num_repeat):
        start_time = time.time()
        print('\nRepeat: %s/%s' % (j + 1, num_repeat))
        print('Digits:\t', digits)

        print('Dephasing data', deph_data)
        print('Dephasing network', deph_net)

        print('Bond Dim: %s' % bd_dim)
        print('Auto Epochs', auto_epochs)
        print('Batch Size: %s' % batch_size)
        sys.stdout.flush()

        model = Model(data_path, digits, val_split, bd_dim, config)
        (test_acc, train_acc) = model.train_network(epochs, batch_size, auto_epochs)

        test_accs.append(test_acc)
        train_accs.append(train_acc)
        print('Time: %.1f' % (time.time() - start_time)); sys.stdout.flush()

    print('\nTrain Accs:\n', train_accs)
    print('Avg Train Acc: %.3f' % np.mean(train_accs))
    print('Std Train Acc: %.3f' % np.std(train_accs))
    print('Test Accs:\n', test_accs)
    print('Avg Test Acc: %.3f' % np.mean(test_accs))
    print('Std Test Acc: %.3f' % np.std(test_accs))
    sys.stdout.flush()

    avg_repeated_test_acc.append(round(float(np.mean(test_accs)), 4))
    avg_repeated_train_acc.append(round(float(np.mean(train_accs)), 4))
    std_repeated_test_acc.append(round(float(np.std(test_accs)), 4))
    std_repeated_train_acc.append(round(float(np.std(train_accs)), 4))


class Model:
    def __init__(self, data_path, digits, val_split, bd_dim, config):
        sample_size = config['data']['sample_size']
        data_im_size = config['data']['data_im_size']
        deph_data = config['meta']['deph']['data']
        deph_net = config['meta']['deph']['network']
        feature_dim = config['data']['feature_dim']
        if config['data']['load_from_file']:
            assert data_im_size == [8, 8] and bd_dim == feature_dim == 2
            (train_data, val_data, test_data) = data.get_data_file(
                data_path, digits, val_split, sample_size=sample_size, deph_data=deph_data)
        else:
            (train_data, val_data, test_data) = data.get_data_web(
                digits, val_split, data_im_size, bd_dim, sample_size=sample_size, deph_data=deph_data)

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

        self.network = network.Network(num_pixels, bd_dim, config, deph_net)

    def train_network(self, epochs, batch_size, auto_epochs):
        tf.debugging.set_log_device_placement(self.config['meta']['log_device_placement'])
        if self.config['meta']['list_devices']: tf.config.list_physical_devices()

        self.epoch_acc = []
        for epoch in range(epochs):
            sys.stdout.flush()
            accuracy = self.run_epoch(batch_size)

            self.epoch_acc.append(accuracy)
            if auto_epochs:
                trigger = self.config['meta']['auto_epochs']['trigger']
                assert trigger < epochs
                if epoch >= trigger and self.check_acc_satified(accuracy): break

        train_or_val_accuracy = accuracy
        if not val_split: print('Train Accuracy: %.3f' % train_or_val_accuracy)
        else: print('Validation Accuracy: %.3f' % train_or_val_accuracy)

        test_accuracy = self.test_network()
        return test_accuracy, train_or_val_accuracy

    def test_network(self):
        test_accuracy = self.run_test_data()
        print('Test Accuracy : {:.3f}'.format(test_accuracy)); sys.stdout.flush()
        return test_accuracy

    def check_acc_satified(self, accuracy):
        criterion = self.config['meta']['auto_epochs']['criterion']
        for i in range(self.config['meta']['auto_epochs']['num_match']):
            if abs(accuracy - self.epoch_acc[-(i + 2)]) <= criterion: continue
            else: return False
        return True

    def run_epoch(self, batch_size):
        batch_iter = data.batch_generator(self.train_images, self.train_labels, batch_size)
        for (train_image_batch, train_label_batch) in batch_iter:
            self.network.train(train_image_batch, train_label_batch)

        if val_split:
            assert self.config['data']['val_split'] > 0
            pred_probs = self.network.get_network_output(self.val_images)
            val_accuracy = get_accuracy(pred_probs, self.val_labels)
            return val_accuracy
        else:
            pred_probs = self.network.get_network_output(self.train_images)
            train_accuracy = get_accuracy(pred_probs, self.train_labels)
            return train_accuracy

    def run_test_data(self):
        test_results = self.network.get_network_output(self.test_images)
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
        print(json.dumps(config, indent=1)); sys.stdout.flush()

    data_path = config['data']['path']
    val_split = config['data']['val_split']
    list_batch_sizes = config['data']['list_batch_sizes']
    list_digits = config['data']['list_digits']
    list_bd_dims = config['meta']['list_bd_dims']
    num_repeat = config['meta']['num_repeat']
    list_epochs = config['meta']['list_epochs']

    num_settings = max(len(list_digits), len(list_bd_dims), len(list_batch_sizes), len(list_epochs))

    main()
