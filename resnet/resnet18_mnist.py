'''
Using ResNet18 to give a baseline.
accuracy: 0.990 ([3,5] for 11000(full) training samples)
accuracy: 0.987 ([3,5] for 5000 training samples)
Reference materials:
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
from uni_ttn.tf2 import data
import torch.nn.functional as F
from copy import deepcopy


def load_data(digits, sample_size):
	'''load not-quantum-featurized data'''
	# data_gen = DataGenerator(dataset='Fashion_MNIST')
	data_gen = data.DataGenerator(dataset='CIFAR')
	shrunk_img_size = 8
	data_gen.shrink_images([shrunk_img_size] * 2)
	(train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = data.process(
		data_gen.train_images, data_gen.train_labels,
		data_gen.test_images, data_gen.test_labels,
		digits, 0.2, sample_size=sample_size
	)

	train_images = train_images[:, None]
	train_images = torch.from_numpy(train_images).to(dtype=torch.float32, device=device)
	train_labels = torch.from_numpy(np.argmax(train_labels, axis=1)).to(dtype=torch.long, device=device)

	valid_images = valid_images[:, None]
	valid_images = torch.from_numpy(valid_images).to(dtype=torch.float32, device=device)
	valid_labels = torch.from_numpy(np.argmax(valid_labels, axis=1)).to(dtype=torch.long, device=device)

	test_images = test_images[:, None]
	test_images = torch.from_numpy(test_images).to(dtype=torch.float32, device=device)
	test_labels = torch.from_numpy(np.argmax(test_labels, axis=1)).to(dtype=torch.long, device=device)

	return train_images, train_labels, valid_images, valid_labels, test_images, test_labels


class Resnet18_Mod(models.resnet.ResNet):
	'''
	Modifying resnet18 for 1-channel grayscale MNIST
	https://discuss.pytorch.org/t/modify-resnet-or-vgg-for-single-channel-grayscale/22762
	'''

	def __init__(self, block, layers, num_classes=2):
		super().__init__(block, layers, num_classes)
		self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

		self.loss_func = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.parameters(), lr=0.005)

		self.to(device=torch.device("cuda"))

	def train_network(self, train_images, train_labels, batch_size):
		batch_iter = data.batch_generator_np(train_images, train_labels, batch_size)
		for images, labels in batch_iter:
			self.optimizer.zero_grad()
			pred_labels = self(images)
			loss = self.loss_func(pred_labels, labels)
			loss.backward()
			self.optimizer.step()

		return self.run_network(train_images, train_labels)

	def run_network(self, images, labels, batch_size=50000):
		num_correct = 0
		with torch.no_grad():
			batch_iter = data.batch_generator_np(images, labels, batch_size)
			for image_batch, label_batch in batch_iter:
				pred_labels = self(image_batch)
				num_correct += get_accuracy_torch(pred_labels, label_batch)

		accuracy = num_correct / len(images)
		return accuracy


class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 20, kernel_size=2)
		self.bn1 = nn.BatchNorm2d(20)
		self.conv1_drop = nn.Dropout2d()
		self.conv2 = nn.Conv2d(20, 40, kernel_size=2)
		self.bn2 = nn.BatchNorm2d(40)
		self.conv2_drop = nn.Dropout2d()
		self.conv3 = nn.Conv2d(40, 80, kernel_size=2)
		self.bn3 = nn.BatchNorm2d(80)
		self.conv3_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 80)
		self.fc2 = nn.Linear(80, 10)
		self.fc3 = nn.Linear(10, 2)
		self.sigmoid = nn.Sigmoid()

		self.loss_func = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.parameters(), lr=0.005)

		self.to(device=torch.device("cuda"))

	def forward(self, x):  # [samp, 1, 8, 8]
		x = self.conv1(x)  # [samp, 20, 7, 7]
		self.bn1(x)
		self.conv1_drop(x)
		x = self.conv2(x)  # [samp, 40, 6, 6]
		self.bn2(x)
		self.conv2_drop(x)
		x = F.relu(F.max_pool2d(x, 2))  # [samp, 80, 3, 3]
		x = self.conv3(x)  # [samp, 80, 2, 2]
		self.bn3(x)
		self.conv3_drop(x)  # [samp, 80, 2, 2]
		x = F.relu(x)
		x = x.view(-1, 320)  # [samp, 320]
		x = self.fc1(x)  # [samp, 80]
		x = F.dropout(x, training=self.training)
		x = F.relu(self.fc2(x))  # [samp, 10]
		x = F.dropout(x, training=self.training)
		x = self.fc3(x)  # [samp, 2]
		x = self.sigmoid(x)
		return x

	def train_network(self, train_images, train_labels, batch_size):
		batch_iter = data.batch_generator_np(train_images, train_labels, batch_size)
		for images, labels in batch_iter:
			self.optimizer.zero_grad()
			pred_labels = self(images)
			loss = self.loss_func(pred_labels, labels)
			loss.backward()
			self.optimizer.step()

		return self.run_network(train_images, train_labels)

	def run_network(self, images, labels, batch_size=50000):
		num_correct = 0
		with torch.no_grad():
			batch_iter = data.batch_generator_np(images, labels, batch_size)
			for image_batch, label_batch in batch_iter:
				pred_labels = self(image_batch)
				num_correct += get_accuracy_torch(pred_labels, label_batch)

		accuracy = num_correct / len(images)
		return accuracy


def get_accuracy_torch(output, target_index):
	output_index = torch.argmax(output, dim=1)
	compare = output_index - target_index
	compare = compare
	num_correct = torch.sum(compare == 0).float()
	return num_correct


def main():
	digits = [[2, 3, 4, 5, 6, 7], [0, 1, 8, 9]]
	# digits = ['even', 'odd']
	# digits = [6, 4]
	sample_size = 8000000
	num_epochs = 120
	train_batch_size = 250

	train_images, train_labels, valid_images, valid_labels, test_images, test_labels = load_data(digits, sample_size)
	print('Train image sample size:', len(train_images))
	print('Valid image sample size:', len(valid_images))
	print('Test image sample size:', len(test_images))

	train_images = torch.tensor(train_images, device=torch.device("cuda"))
	train_labels = torch.tensor(train_labels, device=torch.device("cuda"))
	valid_images = torch.tensor(valid_images, device=torch.device("cuda"))
	valid_labels = torch.tensor(valid_labels, device=torch.device("cuda"))
	test_images = torch.tensor(test_images, device=torch.device("cuda"))
	test_labels = torch.tensor(test_labels, device=torch.device("cuda"))

	network = Resnet18_Mod(block=models.resnet.Bottleneck, layers=[2, 2, 2, 2])
	# network = CNN()

	best_epoch_valid_acc = 0
	for epoch in range(num_epochs):
		training_accuracy = network.train_network(train_images, train_labels, train_batch_size)
		print(f'Epoch: {epoch}: {training_accuracy:.3f}', flush=True)
		if not epoch % 2:
			valid_accuracy = network.run_network(valid_images, valid_labels)
			print('Valid Accuracy: %.3f' % valid_accuracy, flush=True)
			if valid_accuracy >= best_epoch_valid_acc:
				best_epoch_valid_acc = valid_accuracy
				weights = deepcopy(network.state_dict())
				print('Checkpoint saved...')

	network.load_state_dict(weights)
	print('Load last Checkpoint...')
	train_accuracy = network.run_network(train_images, train_labels)
	print('Train Accuracy: %.3f' % train_accuracy, flush=True)
	test_accuracy = network.run_network(test_images, test_labels)
	print('Test Accuracy: %.3f' % test_accuracy, flush=True)


if __name__ == "__main__":
	np.random.seed(42)
	torch.manual_seed(42)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print('Is cuda available:', torch.cuda.is_available(), flush=True)

	main()