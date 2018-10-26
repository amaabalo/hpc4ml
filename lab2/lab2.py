import os
import torch
import argparse
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer

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


def precision_at_k(k, output, label):
	vals, indices = output.topk(k)
	
	count = 0.0
	for i, idx in enumerate(indices):
		corresponding = label[i].index_select(0, idx.data)
		count += corresponding.sum()

	precision = count / (k * output.size(0))
	return precision

	

def main(): 
	'''
	parser = argparse.ArgumentParser(description='Training in PyTorch')
	parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
	args = parser.parse_args()
	args.device = None
	print (torch.cuda.is_available())
	if not args.disable_cuda and torch.cuda.is_available():
		args.device = torch.device('cuda')
		print("Running on GPU")
	else:
		args.device = torch.device('cpu')
		print("Running on CPU.")
	'''
	composed = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
	kd = KaggleDataset("/scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train.csv", "/scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/train-jpg", composed)
	dataloader = DataLoader(kd, batch_size = 250, num_workers = 1)
	n = NeuralNetwork()
	criterion = nn.BCELoss()
	optimiser = optim.SGD(n.parameters(), lr=0.01, momentum=0.9)

	for epoch in range(5):
		n.train()
		for i, data in enumerate(dataloader, 0):
			optimiser.zero_grad()
			outputs = n.forward(Variable(data["image"]))

			loss = criterion(outputs, Variable(data["labels"]))
			loss.backward()
			optimiser.step()
			loss_val = loss.data[0]
		
			# For each minibatch calculate the loss value, the precision@1 and precision@3 of the predictions.	
			print(precision_at_k(3, outputs, data["labels"]))
			exit()


if __name__ == "__main__":
	main()
