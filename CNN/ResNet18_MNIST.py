'''
Using ResNet18 to give a baseline.
Codes mainly base on the cnn_adver_mnist repo.
Accuracy: 0.928
Other reference materials:
sTTN_Deph_MNIST.conv_net_mnist.conv_net_mnist.py
pytroch.vision.references.classification.train.py
https://discuss.pytorch.org/t/modify-resnet-or-vgg-for-single-channel-grayscale/22762
https://nextjournal.com/gkoehler/pytorch-mnist
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py (official implementation of resnet)
'''
import torch
import torchvision.models as models
import torch.optim as optim

import numpy as np
import torch.nn as nn
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './uni_ttn/tf2.7/')
import data

# n_epochs = 3
# batch_size_train = 50
# batch_size_test = 1000
# learning_rate = 0.01
# momentum = 0.5
# log_interval = 10

# random_seed = 1
# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)

def load_data(digits, sample_size, val_split=0):
	datagen = data.DataGenerator()
	datagen.shrink_images([8, 8])

	(train_images, train_labels), _, (test_images, test_labels) = data.process(
		datagen.train_images, datagen.train_labels, 
		datagen.test_images, datagen.test_labels,
		 digits, val_split, sample_size=sample_size 
	)
				
	train_images = np.reshape(train_images, [-1, 1, 8, 8])
	train_images = torch.from_numpy(train_images).to(dtype=torch.float32)
	train_labels = torch.from_numpy(train_labels).to(dtype=torch.float32)
		 
	test_images = np.reshape(test_images, [-1, 1, 8, 8])
	test_images = torch.from_numpy(test_images).to(dtype=torch.float32)
	test_labels = torch.from_numpy(test_labels).to(dtype=torch.float32)

	return (train_images, train_labels, test_images, test_labels)
		
class resnet18_mod(models.resnet.ResNet):
#Modifying resnet18 for 1-channel grayscale MNIST
#https://discuss.pytorch.org/t/modify-resnet-or-vgg-for-single-channel-grayscale/22762
	def __init__(self, block, layers, num_classes=2):
		super(resnet18_mod, self).__init__(block, layers, num_classes)
		self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

		self.loss_func = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.parameters())


	def train(self, train_images, train_labels, batch_size):
		batch_iter_train = data.batch_generator_np(train_images, train_labels, batch_size)
		for images, labels in batch_iter_train:
			self.optimizer.zero_grad()
			pred_labels = self(images)
			loss = self.loss_func(pred_labels, labels)
			loss.backward()
			self.optimizer.step()
		
	def test(self, test_images, test_labels, batch_size):
		correct = 0
		with torch.no_grad():
			batch_iter_test = data.batch_generator_np(test_images, test_labels, batch_size)
			for images, labels in batch_iter_test:
				pred_labels = self(images)
				correct += get_accuracy(pred_labels, labels)[1]

		test_accuracy = correct / test_images.shape[0]
		return test_accuracy

def get_accuracy(output, target):
	output_index = np.argmax(output, axis=1)
	target_index = np.argmax(target, axis=1)
	compare = output_index - target_index
	compare = compare.numpy()
	num_correct = float(np.sum(compare == 0))
	total = float(output_index.shape[0])
	accuracy = num_correct / total
	return accuracy, num_correct

def main():
	digits = [3, 5]
	sample_size = 5000
	(train_images, train_labels, test_images, test_labels) = load_data(digits, sample_size)
	network = resnet18_mod(block=models.resnet.Bottleneck, layers=[2, 2, 2, 2])
	num_epochs = 100
	train_batch_size = 250
	
	for epoch in range(num_epochs):
		print('Epoch: %s'%epoch)
		network.train(train_images, train_labels, train_batch_size)
		test_accuracy = network.test(test_images, test_labels, 1000000)
		print('Test Accuracy: %.3f'%test_accuracy)
	
	torch.save(network.state_dict(), '../trained_models/samp1000_size8.pth')
	print('Model saved')
	
if __name__ == "__main__":
	main()