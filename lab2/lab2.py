import os
import sys
import torch
import argparse
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer

# Needed for C1
class KaggleDataset(Dataset):
	def __init__(self, csv_file, root_dir, transform=None):
		self.root_dir = root_dir
		self.data_frame = pd.read_csv(csv_file)
		self.mlb = MultiLabelBinarizer()
		self.all_labels = torch.from_numpy(self.mlb.fit_transform(self.data_frame.iloc[:, 1].str.split().apply(lambda x: [float (i) for i in x])).astype(np.float32))
		self.transform = transform

	def __len__(self):
		return len(self.data_frame)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0] + ".jpg")
		image = Image.open(img_name)
		image = image.convert("RGB")
		if self.transform:
			image = self.transform(image)
		labels = self.all_labels[idx]
		sample = {'image': image, 'labels': labels}
		return sample

# Needed for C1
class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 32, 3),
			nn.MaxPool2d(2),
			nn.ReLU())

		self.layer2 = nn.Sequential(
			nn.Conv2d(32, 64, 3),
			nn.Dropout2d(),
			nn.MaxPool2d(2),
			nn.ReLU())

		self.layer3 = nn.Sequential(
			nn.Linear(2304, 256),
			nn.Dropout(),
			nn.ReLU())

		self.layer4 = nn.Sequential(
			nn.Linear(256, 17),
			nn.Sigmoid())

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.view(out.size(0), -1)
		out = self.layer3(out)
		out = self.layer4(out)
		return out

# Needed for C5, C1
def precision_at_k(k, output, label, cuda_enabled):
	vals, indices = output.topk(k)
	
	count = 0.0
	for i, idx in enumerate(indices):
		index = idx
		if cuda_enabled:
			index = idx.cpu()
		corresponding = label[i].index_select(0, index.data)
		count += corresponding.sum()

	precision = count / (k * output.size(0))
	return precision.item()


def main(): 
	
	parser = argparse.ArgumentParser(description='Lab 2: Training in PyTorch')
	parser.add_argument("csv_path", help = "Path to the csv file.")
	parser.add_argument("dataset_directory", help = "Path to the dataset directory.")
	parser.add_argument("--n_workers", type = int, default = 1, help = "The number of workers to use for the dataloader.") # Needed for C3
	parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA.', default = False) # Needed for C5
	parser.add_argument("--optimiser", default = "sgd", choices = ['sgd', 'sgdwithnesterov', 'adagrad', 'adadelta', 'adam'], help = "The optimiser to use.") # Needed for C5

	args = parser.parse_args()
	args.device = None
	
	cuda_enabled = False
	print("Torch version " + torch.__version__)
	print("Python version " + sys.version)
	if args.enable_cuda and torch.cuda.is_available():
		print("Running on GPU")
		args.device = torch.device("cuda:0") # Needed for C5
		print("Number of GPUs: " + str(torch.cuda.device_count()))
		cuda_enabled = True
	else:
		if (args.enable_cuda):
			print("--cuda-enable specified but torch.cuda.is_available() = false.")
		print("Running on CPU.")
		args.device = torch.device("cpu")
	
	print(args.device)

	
	print("Number of workers: " + str(args.n_workers) + "; Optimiser: " + args.optimiser)

	if (cuda_enabled):
		n = NeuralNetwork().cuda() # Needed for C5
	else:
		n = NeuralNetwork()

	optimiser = None
	opt = args.optimiser.lower()
	# Needed for C5.
	if (opt == "sgd"):
		optimiser = optim.SGD(n.parameters(), lr=0.01, momentum=0.9)
	elif (opt == "sgdwithnesterov"):
		optimiser = optim.SGD(n.parameters(), lr=0.01, momentum=0.9, nesterov = True)
	elif (opt == "adagrad"):
		optimiser = optim.Adagrad(n.parameters())
	elif (opt == "adadelta"):
		optimiser = optim.Adadelta(n.parameters())
	elif (opt == "adam"):
		optimiser = optim.Adam(n.parameters())		
	
	composed = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
	kd = KaggleDataset(args.csv_path, args.dataset_directory, composed)
	dataloader = DataLoader(kd, batch_size = 250, num_workers = args.n_workers)

	criterion = nn.BCELoss()

	n_epochs = 5
	# Aggregate time for each epoch
	total_time = 0.0

	# Overall total time spent waiting for the Dataloader (Needed for C3)
	overall_total_load_time = 0.0
	print("Epoch\tAvg Load Time\t\tAvg Minibatch Comp Time\t\tAvg Loss\t\t\t\tAvg p@1\t\t\t\tAvg p@3")

	for epoch in range(n_epochs):
		n.train()

		# Aggregate time spent waiting to load the batchfrom the DataLoader during the training
		total_load_time = 0.0
		# Aggregate time for a mini-batch computation (dataloading and NN forward/backward)
		total_minibatch_computation_time = 0.0
		epoch_loss = 0.0
		epoch_p_at_1 = 0.0
		epoch_p_at_3 = 0.0

		load_start = time.monotonic()
		for i, data in enumerate(dataloader, 0):
			load_end = time.monotonic()
			inputs = Variable(data["image"])
			labels = Variable(data["labels"])

			if cuda_enabled:
				inputs = inputs.cuda()
				labels = labels.cuda()
	
			optimiser.zero_grad()
			outputs = n.forward(inputs)

			loss = criterion(outputs, labels)
			loss.backward()
			optimiser.step()
			total_minibatch_computation_time += (time.monotonic() - load_start)
		
			total_load_time += (load_end - load_start)
			# For each minibatch calculate the loss value, the precision@1 and precision@3 of the predictions.	# Needed for C5
			precision_at_3 = precision_at_k(3, outputs, data["labels"], cuda_enabled)
			precision_at_1 = precision_at_k(1, outputs, data["labels"], cuda_enabled)
			epoch_p_at_1 += precision_at_1
			epoch_p_at_3 += precision_at_3
			loss_val = loss.item()

			epoch_loss += loss_val
			load_start = time.monotonic()
		
		overall_total_load_time += total_load_time
		avg_batch_load_time = total_load_time / len(dataloader) # Needed for C2
		avg_loss_this_epoch = epoch_loss / len(dataloader)
		avg_p_at_1_this_epoch = epoch_p_at_1 / len(dataloader)
		avg_p_at_3_this_epoch = epoch_p_at_3 / len(dataloader)
		avg_minibatch_computation_time = total_minibatch_computation_time / len(dataloader) # Needed for C2
	
		print(str(epoch + 1) + "\t\t" + str(avg_batch_load_time) + "\t" + str(avg_minibatch_computation_time) + "\t\t\t" + str(avg_loss_this_epoch) + "\t\t" + str(avg_p_at_1_this_epoch) + "\t\t" + str(avg_p_at_3_this_epoch))
		
		total_time += total_minibatch_computation_time
	
	avg_time_per_epoch = total_time / n_epochs # Needed for C2, C5
	print("Average time per epoch: " + str(avg_time_per_epoch))
	print("Overall total load time: " + str(overall_total_load_time))
	print("\n")


if __name__ == "__main__":
	main()
