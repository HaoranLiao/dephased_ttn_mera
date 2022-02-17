import tensorflow as tf
import numpy as np
import sys, os, time, yaml, json
from tqdm import tqdm
import network_distributed as network_dist
sys.path.append('../')
import data
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
TQDM_DISABLED = False
TQDM_DICT = {'leave': False, 'disable': TQDM_DISABLED, 'position': 0}


def print_results(start_time):
    print('All Avg Test Accs:\n', avg_repeated_test_acc)
    print('All Avg Train/Val Accs:\n', avg_repeated_train_acc)
    print('All Std Test Accs:\n', std_repeated_test_acc)
    print('All Std Train/Val Accs:\n', std_repeated_train_acc)
    print('Time (hr): %.4f' % ((time.time()-start_time)/3600))
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

    early_stop = config['meta']['early_stop']['enabled']
    test_accs, train_accs = [], []
    for j in tqdm(range(num_repeat), total=num_repeat, leave=True):
        start_time = time.time()
        print('\nRepeat: %s/%s' % (j + 1, num_repeat))
        print('Digits:\t', digits)
        print('Dephasing data', config['meta']['deph']['data'])
        print('Dephasing network', config['meta']['deph']['network'])
        print('Dephasing rate %.2f' % deph_p)
        print('Auto Epochs', early_stop)
        print('Batch Size: %s' % batch_size)
        print('Sub Batch Size: %s' % config['data']['sub_batch_size'])
        print('Distributed:', config['data']['distributed'])
        print('Number of Ancillas: %s' % num_anc)
        print('Random Seed:', config['meta']['random_seed'])
        sys.stdout.flush()

        assert epochs; assert batch_size
        model = Model(data_path, digits, val_split, deph_p, num_anc, config)
        test_acc, train_acc = model.train_network(epochs, batch_size, early_stop)

        test_accs.append(round(test_acc, 4))
        train_accs.append(round(train_acc, 4))
        print('Time (hr): %.1f' % ((time.time()-start_time)/3600)); sys.stdout.flush()

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

        if config['meta']['list_devices']: tf.config.list_physical_devices(); sys.stdout.flush()
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, config['meta']['set_memory_growth'])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs", flush=True)

        self.strategy = tf.distribute.MirroredStrategy()
        self.options = tf.data.Options()
        self.options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

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
        print('Sample Size: %s' % self.train_images.shape[0])

        if val_data is not None:
            print('Validation Split: %.2f' % val_split, flush=True)
            self.val_images, self.val_labels = val_data
        else:
            assert config['data']['val_split'] == 0
            print('No Validation', flush=True)

        self.test_images, self.test_labels = test_data

        num_pixels = self.train_images.shape[1]
        self.config = config
        self.network = network_dist.Network(num_pixels, deph_p, num_anc, config, self.strategy)

        self.b_factor = self.config['data']['eval_batch_size_factor']

    def train_network(self, epochs, batch_size, early_stop):
        self.epoch_acc = []
        for epoch in range(epochs):
            accuracy = self.run_epoch(batch_size)
            print('Epoch %d: %.5f accuracy' % (epoch, accuracy), flush=True)

            if not epoch%5:
                test_accuracy = self.run_network(self.test_images, self.test_labels, batch_size*self.b_factor)
                print(f'Test Accuracy : {test_accuracy:.3f}', flush=True)

            self.epoch_acc.append(accuracy)
            if early_stop:
                trigger = self.config['meta']['early_stop']['trigger']
                assert trigger < epochs
                if epoch >= trigger and self.check_acc_satified(accuracy): break

        train_or_val_accuracy = accuracy
        if not val_split: print('Train Accuracy: %.3f' % train_or_val_accuracy, flush=True)
        else: print('Validation Accuracy: %.3f' % train_or_val_accuracy, flush=True)

        test_accuracy = self.run_network(self.test_images, self.test_labels, batch_size*self.b_factor)
        print(f'Test Accuracy : {test_accuracy:.3f}', flush=True)
        return test_accuracy, train_or_val_accuracy

    def check_acc_satified(self, accuracy):
        criterion = self.config['meta']['early_stop']['criterion']
        for i in range(self.config['meta']['early_stop']['num_match']):
            if abs(accuracy - self.epoch_acc[-(i + 2)]) <= criterion: continue
            else: return False
        return True

    @tf.function
    def distributed_eval_step(self, image_batch: tf.constant, label_batch: tf.constant):
        per_replica_num_corrects = self.strategy.run(self.network.get_network_output, args=(image_batch, label_batch))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_corrects, axis=None)

    def run_network(self, images: np.ndarray, labels: np.ndarray, batch_size):
        batch_iter = tf.data.Dataset.from_tensor_slices((images, labels))
        batch_iter = batch_iter.with_options(self.options)
        batch_iter = batch_iter.batch(batch_size)
        num_correct = 0
        if self.config['data']['distributed']:
            batch_iter = self.strategy.experimental_distribute_dataset(batch_iter)
            for (image_batch, label_batch) in tqdm(batch_iter, total=len(images)//batch_size, **TQDM_DICT):
                num_correct += self.distributed_eval_step(image_batch, label_batch)
        else:
            for (image_batch, label_batch) in tqdm(batch_iter, total=len(images)//batch_size, **TQDM_DICT):
                pred_probs = self.network.get_network_output(image_batch)
                num_correct += get_num_correct(pred_probs, label_batch)

        accuracy = num_correct / len(images)
        return float(accuracy)

    @tf.function
    def distributed_train_step(self, train_image_batch, train_label_batch):
        self.strategy.run(self.network.update_distributed, args=(train_image_batch, train_label_batch))

    def run_epoch(self, batch_size):
        batch_iter = tf.data.Dataset.from_tensor_slices((self.train_images, self.train_labels))
        batch_iter = batch_iter.with_options(self.options)
        batch_iter = batch_iter.shuffle(len(self.train_images))
        if self.config['data']['distributed']:
            batch_iter = batch_iter.batch(batch_size)
            batch_iter = self.strategy.experimental_distribute_dataset(batch_iter)
            for (train_image_batch, train_label_batch) in tqdm(batch_iter, total=len(self.train_images)//batch_size, **TQDM_DICT):
                self.distributed_train_step(train_image_batch, train_label_batch)
        else:
            sub_batch_size = self.config['data']['sub_batch_size']
            counter = batch_size // sub_batch_size
            assert not batch_size % sub_batch_size, 'batch_size not divisible by exec_batch_size'
            batch_iter = batch_iter.batch(sub_batch_size)
            for (train_image_batch, train_label_batch) in tqdm(batch_iter, total=len(self.train_images)//sub_batch_size, **TQDM_DICT):
                if counter > 1:
                    counter -= 1
                    self.network.update(train_image_batch, train_label_batch, apply_grads=False)
                else:
                    counter = batch_size // sub_batch_size
                    self.network.update(train_image_batch, train_label_batch, apply_grads=True, counter=counter)

        if val_split:
            assert self.config['data']['val_split'] > 0
            val_accuracy = self.run_network(self.val_images, self.val_labels, batch_size*self.b_factor)
            return val_accuracy
        else:
            train_accuracy = self.run_network(self.train_images, self.train_labels, batch_size*self.b_factor)
            return train_accuracy


def get_num_correct(guesses: np.ndarray, labels: np.ndarray):
    guess_index = np.argmax(guesses, axis=1)
    label_index = np.argmax(labels, axis=1)
    compare = guess_index - label_index
    num_correct = float(np.sum(compare == 0))
    return num_correct

def get_num_correct_tf(guesses: tf.constant, labels: tf.constant):
    guess_index = tf.math.argmax(guesses, axis=1)
    label_index = tf.math.argmax(labels, axis=1)
    compare = guess_index - label_index
    num_correct = tf.reduce_sum(tf.cast(compare == 0, tf.int32))
    return num_correct

if __name__ == "__main__":
    with open('config_example_distributed.yaml', 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
        print(json.dumps(config, indent=1)); sys.stdout.flush()

    np.random.seed(config['meta']['random_seed'])
    tf.random.set_seed(config['meta']['random_seed'])

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
        for i in tqdm(range(num_settings), total=num_settings, leave=True): run_all(i)
    finally:
        print_results(start_time)
