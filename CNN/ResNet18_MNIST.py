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
sys.path.insert(1, '../uni_ttn/tf2.7/')
import data as datagen

# n_epochs = 3
# batch_size_train = 50
# batch_size_test = 1000
# learning_rate = 0.01
# momentum = 0.5
# log_interval = 10

# random_seed = 1
# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)

def load_train_images():
	(train_data, val_data, test_data) = datagen.get_data_web([3,5], 0, [8,8], 2, sample_size=1000)
				
	(train_images, train_labels) = train_data
	train_images = np.reshape(train_images, [1000, 1, 8, 8])
	train_images = torch.from_numpy(train_images).to(dtype=torch.float32, )
	train_labels = torch.from_numpy(train_labels).to(dtype=torch.int64)
		 
	(test_images, test_labels) = test_data
	test_images = np.reshape(test_images, [1000, 1, 8, 8])
	test_images = torch.from_numpy(test_images).to(dtype=torch.float32)
	test_labels = torch.from_numpy(test_labels).to(dtype=torch.int64)
	
	return (train_images, train_labels, test_images, test_labels)
		
class resnet18_mod(models.resnet.ResNet):
#Modifying resnet18 for 1-channel grayscale MNIST
#https://discuss.pytorch.org/t/modify-resnet-or-vgg-for-single-channel-grayscale/22762
	def __init__(self, block, layers, num_classes=2):
		self.inplanes = 64
		super(resnet18_mod, self).__init__(block, layers, num_classes)
		self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)

# train_loader = torch.utils.data.DataLoader(
#   	torchvision.datasets.MNIST('~/QTTN/dephased_ttn_project/mnist8by8/', train=True, download=True,
# 							 transform=torchvision.transforms.Compose([
# 							   torchvision.transforms.ToTensor(),
# 							   torchvision.transforms.Normalize(
# 								 (0.1307,), (0.3081,))
# 							 ])),
#   	batch_size=batch_size_train, shuffle=True)

# test_loader = torch.utils.data.DataLoader(
#   	torchvision.datasets.MNIST('~/QTTN/dephased_ttn_project/mnist8by8/', train=False, download=True,
# 							 transform=torchvision.transforms.Compose([
# 							   torchvision.transforms.ToTensor(),
# 							   torchvision.transforms.Normalize(
# 								 (0.1307,), (0.3081,))
# 							 ])),
#   	batch_size=batch_size_test, shuffle=True)

#Repeating each pixel three times to fit in ResNet. See https://discuss.pytorch.org/t/grayscale-to-rgb-transform/18315/4 
# train_loader = torch.utils.data.DataLoader(
#   	torchvision.datasets.MNIST('~/QTTN/dephased_ttn_project/mnist8by8/', train=True, download=True,
# 							 transform=torchvision.transforms.Compose([
# 							   torchvision.transforms.ToTensor(),
# 							   torchvision.transforms.Normalize(
# 								 (0.1307,), (0.3081,)),
# 							   torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) )
# 							 ])),
#   batch_size=batch_size_train, shuffle=True)

# test_loader = torch.utils.data.DataLoader(
#   	torchvision.datasets.MNIST('~/QTTN/dephased_ttn_project/mnist8by8/', train=False, download=True,
# 							 transform=torchvision.transforms.Compose([
# 							   torchvision.transforms.ToTensor(),
# 							   torchvision.transforms.Normalize(
# 								 (0.1307,), (0.3081,)),
# 								torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) )
# 							 ])),
#   batch_size=batch_size_test, shuffle=True)

# network = resnet18_mod(block = models.resnet.Bottleneck, layers = [2, 2, 2, 2] )

# optimizer = optim.SGD(network.parameters(), lr=learning_rate,
# 					  momentum=momentum)

# train_losses = []
# train_counter = []
# test_losses = []
# test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(network, optimizer, train_images, train_labels):
	network.train()
	batch_idx = 0
	batch_iter_train = datagen.batch_generator(train_images, train_labels, 100)
	for (data, target) in batch_iter_train:
		batch_idx += 1
		optimizer.zero_grad()
		output = network(data)
		# loss_func = nn.CrossEntropyLoss()
		# loss = loss_func(output,target)
		loss = ((output - target)**2).mean()
		loss.backward()
		optimizer.step()
	  
def test(network, test_images, test_labels):
	network.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		batch_iter_test = datagen.batch_generator(test_images, test_labels, 1000)
		for data, target in batch_iter_test:
			output = network(data)
			# loss_func = nn.CrossEntropyLoss()
			# test_loss += loss_func(output,target)
			test_loss += ((output - target)**2).mean()
			correct += get_accuracy(output, target)

	return correct

def get_accuracy(output, target):
	output_index = np.argmax(output, axis=1)
	target_index = np.argmax(target, axis=1)
	compare = output_index - target_index
	compare = compare.numpy()
	num_correct = float(np.sum(compare == 0))
	total = float(output_index.shape[0])
	accuracy = num_correct / total
	return accuracy

def main():
	(train_images, train_labels, test_images, test_labels) = load_train_images()
	network = resnet18_mod(block = models.resnet.Bottleneck, layers = [2, 2, 2, 2] )
	optimizer = optim.Adam(network.parameters())
	
	print('Test Accuracy: %.3f'%test(network, test_images, test_labels))
	for epoch in range(1, 101):
		print('Epoch: %s'%epoch)
		train(network, optimizer, train_images, train_labels)
		print('Test Accuracy: %.3f'%test(network, test_images, test_labels))
	
	torch.save(network.state_dict(), '../trained_models/samp1000_size8.pth')
	print('Model saved')
	
if __name__ == "__main__":
	main()