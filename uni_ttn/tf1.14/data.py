import pickle as pk
import numpy as np
import tensorflow as tf
import skimage.transform
import math


class DataGenerator:
    def __init__(self):
        mnist_data = tf.keras.datasets.mnist.load_data()
        self.train_images = mnist_data[0][0]
        self.train_labels = mnist_data[0][1]
        self.test_images = mnist_data[1][0]
        self.test_labels = mnist_data[1][1]

    def shrink_images(self, new_shape):
        self.train_images = resize_images(self.train_images, new_shape)
        self.test_images = resize_images(self.test_images, new_shape)

    def featurize(self, dim=2):
        self.train_images = trig_featurize(self.train_images, dim)
        self.test_images = trig_featurize(self.test_images, dim)
        
    def featurize_org(self):
        self.train_images = trig_featurize_org(self.train_images)
        self.test_images = trig_featurize_org(self.test_images)

    def export(self, path):
        train_dest = path + '_train'
        test_dest = path + '_test'
        save_data(self.train_images, self.train_labels, train_dest)
        save_data(self.test_images, self.test_labels, test_dest)


def select_digits(images, labels, digits):
    cumulative_test = (labels == digits[0])
    for digit in digits[1:]:
        digit_test = (labels == digit)
        cumulative_test = np.logical_or(digit_test, cumulative_test)

    valid_images = images[cumulative_test]
    valid_labels = labels[cumulative_test]
    return valid_images, valid_labels


def resize_images(images, shape):
    num_images = images.shape[0]
    new_images_shape = (num_images, shape[0], shape[1])
    new_images = skimage.transform.resize(
        images,
        new_images_shape,
        anti_aliasing=True,
        mode='constant')
    return new_images


def batch_generator(images, labels, batch_size):
    num_images = images.shape[0]
    random_perm = np.random.permutation(num_images)
    randomized_images = images[random_perm]
    randomized_labels = labels[random_perm]
    
    for i in range(0, num_images, batch_size):
        batch_images = randomized_images[i : i+batch_size]
        batch_labels = randomized_labels[i : i+batch_size]
        yield batch_images, batch_labels


def flatten_images(images):
    num_images = images.shape[0]
    flattened_image = np.reshape(images, [num_images, -1])
    return flattened_image


def trig_featurize_org(images):
    flat_images = flatten_images(images)
    (num_images, num_pixels) = flat_images.shape
    prep_axes = np.reshape(flat_images, (num_images, num_pixels, 1))
    pix_copy = np.tile(prep_axes, [1, 1, 2])
    pix_copy[:, :, 0] = np.cos(pix_copy[:, :, 0] * np.pi/2)
    pix_copy[:, :, 1] = np.sin(pix_copy[:, :, 1] * np.pi/2)
    return pix_copy
    
def trig_featurize(images, dim):
    flat_images = flatten_images(images)
    (num_images, num_pixels) = flat_images.shape
    prep_axes = np.reshape(flat_images, (num_images, num_pixels, 1))
    pix_copy = np.tile(prep_axes, [1, 1, dim])
    
    d = dim
    for s in range(dim):
        s += 1
        pix_copy[:, :, s-1] = np.sqrt(
                float(math.factorial(d-1)) / \
                float(math.factorial(s-1) * math.factorial(d-s)) \
                ) \
                * np.cos(pix_copy[:, :, s-1] * np.pi/2) ** (d-s) \
                * np.sin(pix_copy[:, :, s-1] * np.pi/2) ** (s-1)
        
    return pix_copy


def split_data(images, labels, split):
    num_images = images.shape[0]
    split_point = int(split * num_images)
    true_train_data = (images[:split_point], labels[:split_point])
    val_data = (images[split_point:], labels[split_point:])
    return true_train_data, val_data


def binary_labels(labels):
    max_digit = np.amax(labels)
    binary_values = np.floor_divide(labels, max_digit)
    binary_labels = one_hot(binary_values)
    return binary_labels


def one_hot(bin_labels):
    length = np.amax(bin_labels) + 1
    blank = np.zeros(bin_labels.size * length)
    multiples = np.arange(bin_labels.size)
    index_shift = length * multiples
    new_indices = bin_labels + index_shift
    blank[new_indices] = 1
    matrix_blank = np.reshape(blank, [bin_labels.size, length])
    return matrix_blank


def get_data_file(data_path, digits, val_split, sample_size=None):
    print('Load Data From File')
    (train_raw_im, train_raw_lab) = load_data(data_path + '_train')
    (test_raw_im, test_raw_lab) = load_data(data_path + '_test')
    
    (train_images, train_labels_int) = select_digits(train_raw_im, train_raw_lab, digits)
    (test_images, test_labels_int) = select_digits(test_raw_im, test_raw_lab, digits)
    if sample_size is not None:
        assert sample_size > 0
        train_images, train_labels_int = train_images[0:sample_size], train_labels_int[0:sample_size]
        test_images, test_labels_int = test_images[0:sample_size], test_labels_int[0:sample_size]

    train_labels = binary_labels(train_labels_int)
    test_labels = binary_labels(test_labels_int)
    
    if val_split:
        assert val_split > 0
        (true_train_data, val_data) = split_data(train_images, train_labels, val_split)
        return true_train_data, val_data, (test_images, test_labels)
    else:
        assert val_split == 0
        return (train_images, train_labels), None, (test_images, test_labels)
    
    
def get_data_web(digits, val_split, size, dim, sample_size=None):
    print('Fetch Data From Web')
    data = DataGenerator()
    data.shrink_images(size)
    data.featurize(dim=dim)
    train_raw_im, train_raw_lab = data.train_images, data.train_labels
    test_raw_im, test_raw_lab = data.test_images, data.test_labels

    (train_images, train_labels_int) = select_digits(train_raw_im, train_raw_lab, digits)
    (test_images, test_labels_int) = select_digits(test_raw_im, test_raw_lab, digits)
    if sample_size is not None:
        assert sample_size > 0
        train_images, train_labels_int = train_images[0:sample_size], train_labels_int[0:sample_size]
        test_images, test_labels_int = test_images[0:sample_size], test_labels_int[0:sample_size]
    
    train_labels = binary_labels(train_labels_int)
    test_labels = binary_labels(test_labels_int)
    
    if val_split:
        assert val_split > 0
        (true_train_data, val_data) = split_data(train_images, train_labels, val_split)
        return true_train_data, val_data, (test_images, test_labels)
    else:
        assert val_split == 0
        return (train_images, train_labels), None, (test_images, test_labels)


def save_data(images, labels, path):
    dest = open(path, 'wb')
    data = (images, labels)
    pk.dump(data, dest, protocol=2)


def load_data(path):
    dest = open(path, 'rb')
    data = pk.load(dest)
    return data


if __name__ == '__main__':
    data1 = DataGenerator()
    data1.shrink_images([8, 8])
    dim = 2
    data1.featurize(dim=dim)
    
    data2 = DataGenerator()
    data2.shrink_images([8, 8])
    data2.featurize_org()