import tensorflow as tf
import numpy as np
import sys, os, time, yaml, json, gc
from tqdm import tqdm
import mera.network, mera.network_ancilla, mera.network_8inputs
import uni_ttn.tf2.model
from uni_ttn.tf2.model import var_or_const

TQDM_DISABLED = False if __name__ == '__main__' else True
TQDM_DICT = {'leave': False, 'disable': TQDM_DISABLED, 'position': 0}

def print_results(start_time):
    print('All Settings Avg Test Accs:\n', avg_repeated_test_acc)
    print('All Settings Avg Train/Val Accs:\n', avg_repeated_train_acc)
    print('All Settings Std Test Accs:\n', std_repeated_test_acc)
    print('All Settings Std Train/Val Accs:\n', std_repeated_train_acc)
    print('Time (hr): %.4f' % ((time.time()-start_time)/3600))
    sys.stdout.flush()

def run_all(i):
    digits = var_or_const(list_digits, i)
    epochs = var_or_const(list_epochs, i)
    batch_size = var_or_const(list_batch_sizes, i)
    deph_p = var_or_const(list_deph_p, i)
    num_anc = var_or_const(list_num_anc, i)
    init_std = var_or_const(list_init_std, i)
    lr = var_or_const(list_lr, i)

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
        print(f'Initialized {model.model_type}')
        test_acc, train_acc = model.train_network(epochs, batch_size, auto_epochs)

        test_accs.append(round(test_acc, 5))
        train_accs.append(round(train_acc, 5))
        print('Time (hr): %.4f' % ((time.time()-start_time)/3600), flush=True)
        gc.collect()

    print(f'\nSetting {i} Train Accs: {train_accs}\t')
    print('Setting %d Avg Train Acc: %.4f' % (i, float(np.mean(train_accs))))
    print('Setting %d Std Train Acc: %.4f' % (i, float(np.std(train_accs))))
    print(f'Setting {i} Test Accs: {test_accs}\t')
    print('Setting %d Avg Test Acc: %.4f' % (i, float(np.mean(test_accs))))
    print('Setting %d Std Test Acc: %.4f' % (i, float(np.std(test_accs))))
    sys.stdout.flush()

    avg_repeated_test_acc.append(round(float(np.mean(test_accs)), 4))
    avg_repeated_train_acc.append(round(float(np.mean(train_accs)), 4))
    std_repeated_test_acc.append(round(float(np.std(test_accs)), 4))
    std_repeated_train_acc.append(round(float(np.std(train_accs)), 4))

class Model(uni_ttn.tf2.model.Model):
    def __init__(self, data_path, digits, val_split, deph_p, num_anc, init_std, lr, config):
        super().__init__(data_path, digits, val_split, deph_p, num_anc, init_std, lr, config)

        self.model_type = 'MERA'
        data_im_size = config['data']['data_im_size']

        if data_im_size == [4, 4] and config['data']['use_4x4_pixel_dict']:
            print('Using 4x4 Pixel Dict', flush=True)
            self.create_pixel_dict()
            self.train_images = self.train_images[:, self.pixel_dict]
            self.test_images = self.test_images[:, self.pixel_dict]
            if self.val_images: self.val_images = self.val_images[:, self.pixel_dict]

        num_pixels = self.train_images.shape[1]
        self.config, self.num_anc = config, num_anc
        if data_im_size == [4, 4]:
            self.network = mera.network.Network(num_pixels, deph_p, num_anc, init_std, lr, config) if not num_anc \
                        else mera.network_ancilla.Network(num_pixels, deph_p, num_anc, init_std, lr, config)
        elif data_im_size == [8, 1]:
            self.network = mera.network_8inputs.Network(num_pixels, deph_p, num_anc, init_std, lr, config)

    def create_pixel_dict(self):
        self.pixel_dict = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]


if __name__ == "__main__":
    with open('config_example.yaml', 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
        print(json.dumps(config, indent=1), flush=True)

    if config['meta']['set_visible_gpus']:
        os.environ["CUDA_VISIBLE_DEVICES"] = config['meta']['visible_gpus']

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
