'''
https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/tf_mnist_example.py
'''

import tensorflow as tf
import numpy as np
import sys, os, time, yaml, json
from tqdm import tqdm
import tuning_network
sys.path.append('../../uni_ttn/tf2.7/')
import data
sys.path.append('../')
import model
from model import variable_or_uniform
from ray import tune
try: from ray.tune.suggest.ax import AxSearch
except ImportError: pass
from filelock import FileLock

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
TQDM_DISABLED = True
TQDM_DICT = {'leave': False, 'disable': TQDM_DISABLED, 'position': 0}


class Model(model.Model):
    def __init__(self, data_path, digits, val_split, deph_p, num_anc, config, tune_config):

        if config['meta']['list_devices']: tf.config.list_physical_devices(); sys.stdout.flush()
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, config['meta']['set_memory_growth'])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print('Physical GPUs:', len(gpus), 'Logical GPUs:', len(logical_gpus), flush=True)

        sample_size = config['data']['sample_size']
        data_im_size = config['data']['data_im_size']
        feature_dim = config['data']['feature_dim']

        with FileLock(os.path.expanduser("~/.tune.lock")):
            if config['data']['load_from_file']:
                assert data_im_size == [4, 4] and feature_dim == 2
                train_data, val_data, test_data = data.get_data_file(
                    data_path, digits, val_split, sample_size=sample_size)
            else:
                train_data, val_data, test_data = data.get_data_web(
                    digits, val_split, data_im_size, feature_dim, sample_size=sample_size)

        self.train_images, self.train_labels = train_data
        print('Train Sample Size: %s' % len(self.train_images), flush=True)

        if val_data is not None:
            self.val_images, self.val_labels = val_data
            print('Validation Split: %.2f\t Size: %d' % (val_split, len(self.val_images)), flush=True)
        else:
            self.val_images = None
            assert not config['data']['val_split']; print('No Validation', flush=True)

        self.test_images, self.test_labels = test_data
        print('Test Sample Size: %s' % len(self.test_images), flush=True)

        if data_im_size == [4, 4] and config['data']['use_4by4_pixel_dict']:
            print('Using 4x4 Pixel Dict', flush=True)
            self.create_pixel_dict()
            self.train_images = self.train_images[:, self.pixel_dict]
            self.test_images = self.test_images[:, self.pixel_dict]
            if val_data: self.val_images = self.val_images[:, self.pixel_dict]

        num_pixels = self.train_images.shape[1]
        self.config = config
        self.network = tuning_network.Network(num_pixels, deph_p, num_anc, config, tune_config)

        self.b_factor = self.config['data']['eval_batch_size_factor']

    def train_network(self, epochs, batch_size, auto_epochs):
        self.epoch_acc = []
        for epoch in range(epochs):
            accuracy = self.run_epoch(batch_size)
            print('Epoch %d: %.5f accuracy' % (epoch, accuracy), flush=True)

            if not epoch % 2:
                test_accuracy = self.run_network(self.test_images, self.test_labels, batch_size*self.b_factor)
                print(f'Test Accuracy : {test_accuracy:.3f}', flush=True)

            self.epoch_acc.append(accuracy)
            if auto_epochs:
                trigger = self.config['meta']['auto_epochs']['trigger']
                assert trigger < epochs
                if epoch >= trigger and self.check_acc_satified(accuracy): break

        train_or_val_accuracy = accuracy
        if not val_split: print('Train Accuracy: %.3f' % train_or_val_accuracy, flush=True)
        else: print('Validation Accuracy: %.3f' % train_or_val_accuracy, flush=True)

        test_accuracy = self.run_network(self.test_images, self.test_labels, batch_size*self.b_factor)
        print(f'Test Accuracy : {test_accuracy:.3f}', flush=True)
        return test_accuracy, train_or_val_accuracy


    def run_epoch(self, batch_size, epoch, grad_accumulation=True):
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
                    self.network.update(train_image_batch, train_label_batch, epoch, apply_grads=False)
                else:
                    counter = batch_size // exec_batch_size
                    self.network.update(train_image_batch, train_label_batch, epoch, apply_grads=True, counter=counter)


class MERA(tune.Trainable):
    def setup(self, tune_config):
        import tensorflow as tf     # required by ray tune
        print(os.getcwd())  # the cwd may not be the current file path

        self.tune_config = tune_config
        digits = variable_or_uniform(list_digits, 0)
        deph_p = variable_or_uniform(list_deph_p, 0)
        num_anc = variable_or_uniform(list_num_anc, 0)
        self.model = Model(data_path, digits, val_split, deph_p, num_anc, config, tune_config)

    def step(self):
        self.model.run_epoch(batch_size, self.iteration, grad_accumulation=config['data']['grad_accumulation'])
        test_accuracy = self.model.run_network(self.model.test_images, self.model.test_labels, batch_size*self.model.b_factor)
        return {
            "epoch": self.iteration,
            "test_accuracy": test_accuracy * 100
        }


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

    deph_p = variable_or_uniform(list_deph_p, 0)
    num_anc = variable_or_uniform(list_num_anc, 0)
    batch_size = variable_or_uniform(list_batch_sizes, 0)

    asha_scheduler = tune.schedulers.ASHAScheduler(
        time_attr='training_iteration',
        max_t=80,
        grace_period=30
    )

    #ax_search = AxSearch(metric="score")

    analysis = tune.run(
        MERA,
        metric='test_accuracy',
        mode='max',
        verbose=3,
        num_samples=1,
        config={'num_anc': num_anc,
                'deph_p': deph_p,
                'tune_lr': tune.grid_search([0.005, 0.025]), #0, # not used in spsa
                'tune_init_std': tune.grid_search([0.5, 0.1, 0.05, 0.01, 0.005, 0.001]), #0.1,
                # 'a': tune.uniform(1, 50),
                # 'b': tune.uniform(1, 50),
                # 'A': tune.uniform(1, 10),
                # 's': tune.uniform(0, 5),
                # 't': tune.uniform(0, 3),
                # 'gamma': tune.uniform(0, 1)
                },
        local_dir='~/dephased_ttn_project/mera/ray_results/',
        resources_per_trial={'cpu': 12, 'gpu': 1},
        scheduler=asha_scheduler,
        progress_reporter=tune.CLIReporter(max_progress_rows=100),
        #search_alg=ax_search,
        log_to_file=True,
        name='anc%.0f_deph%.0f' % (num_anc, deph_p)
    )

    print("Best hyperparameters found were: ", analysis.best_config)

    # num_settings = max(len(list_digits), len(list_num_anc),
    #                    len(list_batch_sizes), len(list_epochs), len(list_deph_p))
    #
    # avg_repeated_test_acc, avg_repeated_train_acc = [], []
    # std_repeated_test_acc, std_repeated_train_acc = [], []
    #
    # start_time = time.time()
    # try:
    #     for i in tqdm(range(num_settings), total=num_settings, leave=True): run_all(i)
    # finally:
    #     print_results(start_time)
