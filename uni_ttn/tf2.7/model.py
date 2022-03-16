import tensorflow as tf
import numpy as np
import sys, os, time, yaml, json
from tqdm import tqdm
import network
import data
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
TQDM_DISABLED = False
TQDM_DICT = {'leave': False, 'disable': TQDM_DISABLED, 'position': 0}


def print_results(start_time):
    print('All Settings Avg Test Accs:\n', avg_repeated_test_acc)
    print('All Settings Avg Train/Val Accs:\n', avg_repeated_train_acc)
    print('All Settings Std Test Accs:\n', std_repeated_test_acc)
    print('All Settings Std Train/Val Accs:\n', std_repeated_train_acc)
    print('Time (hr): %.4f' % ((time.time()-start_time)/3600))
    sys.stdout.flush()

def variable_or_uniform(input, i):
    return input[i] if len(input) > 1 else input[0]

def run_all(i):
    digits = variable_or_uniform(list_digits, i)
    epochs = variable_or_uniform(list_epochs, i)
    batch_size = variable_or_uniform(list_batch_sizes, i)
    deph_p = variable_or_uniform(list_deph_p, i)
    num_anc = variable_or_uniform(list_num_anc, i)
    init_std = variable_or_uniform(list_init_std, i)
    lr = variable_or_uniform(list_lr, i)

    auto_epochs = config['meta']['auto_epochs']['enabled']
    test_accs, train_accs = [], []
    for j in tqdm(range(num_repeat), total=num_repeat, leave=True):
        start_time = time.time()
        print('\nRepeat: %s/%s' % (j + 1, num_repeat))
        print('Digits:\t', digits)
        print('Dephasing data', config['meta']['deph']['data'])
        print('Dephasing network', config['meta']['deph']['network'])
        print('Dephasing rate %.2f' % deph_p)
        print('Auto Epochs', auto_epochs)
        print('Batch Size: %s' % batch_size)
        print('Exec Batch Size: %s' % config['data']['execute_batch_size'])
        print('Number of Ancillas: %s' % num_anc)
        print('Random Seed:', config['meta']['random_seed'])
        print(f'Init Std: {init_std}')
        print(f'Adam Learning Rate: {lr}')
        sys.stdout.flush()

        model = Model(data_path, digits, val_split, deph_p, num_anc, init_std, lr, config)
        test_acc, train_acc = model.train_network(epochs, batch_size, auto_epochs)

        test_accs.append(round(test_acc, 4))
        train_accs.append(round(train_acc, 4))
        print('Time (hr): %.4f' % ((time.time()-start_time)/3600), flush=True)

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
    def __init__(self, data_path, digits, val_split, deph_p, num_anc, init_std, lr, config):

        if config['meta']['list_devices']: tf.config.list_physical_devices(); sys.stdout.flush()
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, config['meta']['set_memory_growth'])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print('Physical GPUs:', len(gpus), 'Logical GPUs:', len(logical_gpus), flush=True)

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

        if data_im_size == [8, 8] and config['data']['use_8by8_pixel_dict']:
            print('Using 8x8 Pixel Dict', flush=True)
            self.create_pixel_dict()
            self.train_images = self.train_images[:, self.pixel_dict]
            self.test_images = self.test_images[:, self.pixel_dict]
            if val_data: self.val_images = self.val_images[:, self.pixel_dict]

        num_pixels = self.train_images.shape[1]
        self.config = config
        self.num_anc = num_anc
        self.network = network.Network(num_pixels, deph_p, num_anc, init_std, lr, config)

        self.b_factor = self.config['data']['eval_batch_size_factor']

    def create_pixel_dict(self):
        self.pixel_dict = []
        for index in range(64):
            quad = index // 16
            quad_quad = (index % 16) // 4
            pos = index % 4
            row = (pos // 2) + 2 * (quad_quad // 2) + 4 * (quad // 2)
            col = (pos % 2) + 2 * (quad_quad % 2) + 4 * (quad % 2)
            pixel = col + 8 * row
            self.pixel_dict.append(pixel)

    def train_network(self, epochs, batch_size, auto_epochs):
        self.epoch_acc, self.history_val_acc = [], [-1]
        for epoch in range(epochs):
            train_accuracy = self.run_epoch(batch_size)

            if not epoch % 2:
                val_accuracy = self.run_network(self.val_images, self.val_labels, batch_size*self.b_factor)
                print('Epoch {0:2} Train Accuracy: {1:.4}\tValid Accuracy: {2:.4}'
                      .format(epoch, train_accuracy, val_accuracy), flush=True)
                if val_accuracy >= max(self.history_val_acc):
                    checkpoint = {f'param_var_lay_{i}':
                                      tf.identity(layer.param_var_lay) for i, layer in enumerate(self.network.layers)}
                    checkpoint['epoch'] = epoch
                    print('Checkpoint saved...', flush=True)
            else:
                print('Epoch {0:2} Train Accuracy: {1:.4}'.format(epoch, train_accuracy), flush=True)

            self.epoch_acc.append(train_accuracy)
            if auto_epochs:
                trigger = self.config['meta']['auto_epochs']['trigger']
                assert trigger < epochs
                if epoch >= trigger and self.check_acc_satified(train_accuracy): break

        for i in range(len(self.network.layers)):
            self.network.layers[i].param_var_lay = checkpoint[f'param_var_lay_{i}']
        print('Restored from epoch %d' % (checkpoint['epoch']), flush=True)

        tf.config.run_functions_eagerly(True)
        test_accuracy = self.run_network(self.test_images, self.test_labels, batch_size*self.b_factor)
        print(f'Test Accuracy : {test_accuracy:.3f}', flush=True)
        return test_accuracy, train_accuracy

    def run_network(self, images, labels, batch_size):
        num_correct = 0
        batch_iter = data.batch_generator_np(images, labels, batch_size)
        for (image_batch, label_batch) in tqdm(batch_iter, total=len(images)//batch_size, **TQDM_DICT):
            image_batch = tf.constant(image_batch, dtype=tf.float32)
            pred_probs = self.network.get_network_output(image_batch)
            num_correct += get_num_correct(pred_probs, label_batch)
        accuracy = num_correct / len(images)
        return accuracy

    def check_acc_satified(self, accuracy):
        criterion = self.config['meta']['auto_epochs']['criterion']
        for i in range(self.config['meta']['auto_epochs']['num_match']):
            if abs(accuracy - self.epoch_acc[-(i + 2)]) <= criterion: continue
            else: return False
        return True

    def run_epoch(self, batch_size, grad_accumulation=True):
        if not grad_accumulation:
            batch_iter = data.batch_generator_np(self.train_images, self.train_labels, batch_size)
            for (train_image_batch, train_label_batch) in tqdm(batch_iter, total=len(self.train_images)//batch_size, **TQDM_DICT):
                self.network.update_no_processing(train_image_batch, train_label_batch)
        else:
            exec_batch_size = self.config['data']['execute_batch_size']
            counter = batch_size // exec_batch_size
            assert not batch_size % exec_batch_size, 'batch_size not divisible by exec_batch_size'
            batch_iter = data.batch_generator_np(self.train_images, self.train_labels, exec_batch_size)
            for (train_image_batch, train_label_batch) in tqdm(batch_iter, total=len(self.train_images)//exec_batch_size, **TQDM_DICT):
                if counter > 1:
                    counter -= 1
                    self.network.update(train_image_batch, train_label_batch, apply_grads=False)
                else:
                    counter = batch_size // exec_batch_size
                    self.network.update(train_image_batch, train_label_batch, apply_grads=True, counter=counter)

        # if val_split:
        #     assert self.config['data']['val_split'] > 0
        #     val_accuracy = self.run_network(self.val_images, self.val_labels, batch_size*self.b_factor)
        #     return val_accuracy
        # else:
        train_accuracy = self.run_network(self.train_images, self.train_labels, batch_size*self.b_factor)
        return train_accuracy


def get_num_correct(guesses, labels):
    guess_index = np.argmax(guesses, axis=1)
    label_index = np.argmax(labels, axis=1)
    compare = guess_index - label_index
    num_correct = float(np.sum(compare == 0))
    return num_correct


if __name__ == "__main__":
    with open('config_example.yaml', 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
        print(json.dumps(config, indent=1), flush=True)

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
    list_init_std = config['tree']['param']['init_std']
    list_lr = config['tree']['opt']['adam']['lr']

    num_settings = max(len(list_digits), len(list_num_anc), len(list_init_std),
                       len(list_batch_sizes), len(list_epochs), len(list_deph_p),
                       len(list_lr))

    avg_repeated_test_acc, avg_repeated_train_acc = [], []
    std_repeated_test_acc, std_repeated_train_acc = [], []

    start_time = time.time()
    try:
        for i in tqdm(range(num_settings), total=num_settings, leave=True): run_all(i)
    finally:
        print_results(start_time)
