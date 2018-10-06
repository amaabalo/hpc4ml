import time
import numpy as np

# Compute AB
def matmul(A, B):
	# matrices must be correct size
	if (len(A[0]) != len(B)):
		raise ValueError ("Cannot multiply matrices with dimensions.")

	rows_a = len(A)
	cols_a = len(A[0])
	rows_b = len(B)
	cols_b = len(B[0])
	res = [[0 for x in range(cols_b)] for y in range (rows_a)]

	for r_a in range(rows_a):
		t = [0 for x in range(cols_b)]
		for c in range (cols_b):
			sm = 0
			for r in range(0, rows_b):
				sm += (A[r_a][r] * B[r][c])
			res[r_a][c] = sm 

	return res
		 
a = [[1, 2], [3, 4]]
b = [[0, 1], [1, 0]]
c = matmul(b, b)

# Return a zero matrix with given dimensions, 
# or a matrix initialized according to the function f(r, c)
def mat(rows, cols, f = None):
	if f == None:
		return [[0 for c in range(cols)] for r in range(rows)]

	return [[f(r, c) for c in range(cols)] for r in range(rows)]

def init_func(i, j):
	return (0.4 + ((i+j) % 40 - 20) / 40.0)

class NeuralNetwork:
	def __init__(self, input_rows, layer_dims):
		self.W = []
		self.n_layers = len(layer_dims)
		self.Z = []
		# Initialise the input
		self.inpt = mat(input_rows, 1, init_func)

		# Initialise the weights at each layer
		n_cols = input_rows
		for i in range(self.n_layers):
			W = mat(layer_dims[i], n_cols, init_func)
			self.W.append(W)
			n_cols = layer_dims[i]

	def get_checksum(self):
		if (self.Z == []):
			return None

		checksum = 0;
		for v in self.Z[len(self.Z) - 1]:
			checksum += v[0]
		return checksum

	def feed_forward(self):
		self.Z = []
		start = time.monotonic()
		for l in range(self.n_layers):
			if (l == 0):
				x = self.inpt
			else:
				x = self.Z[l - 1]

			Z = matmul(self.W[l], x)


			rows = len(self.W[l])

			#RELU activation
			for r in range(rows):
				if (Z[r][0]) < 0:
					Z[r][0] = 0 

			self.Z.append(Z)
		end = time.monotonic()
		return (end - start, self.get_checksum())

	def feed_forward_w_np(self):
		self.Z = []
		start = time.monotonic()
		for l in range(self.n_layers):
			if (l == 0):
				x = self.inpt
			else:
				x = self.Z[l - 1]


			Z = np.dot(self.W[l], x)

			rows = len(self.W[l])

			#RELU activation
			np.clip(Z, 0, None)

			self.Z.append(Z)
		end = time.monotonic()
		return (end - start, self.get_checksum())

def main():
	nn = NeuralNetwork(256 * 256, [4000, 1000])
	##nn = NeuralNetwork(10 * 10, [40, 10])
	exec_time, checksum = nn.feed_forward()
	print("*C3*")
	print("Time: " + str(exec_time) + " secs")
	print("Checksum: " + str(checksum))

	print()

	exec_time2, checksum = nn.feed_forward_w_np()
	print("*C4*")
	print("Time: " + str(exec_time2) + " secs")
	print("Checksum: " + str(checksum))
	print("Speedup w.r.t. C3: " + str(exec_time / exec_time2))

	print()

if __name__ == "__main__":
	main()
