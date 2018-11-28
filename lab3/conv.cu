#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cudnn.h>

#define CUDNN_CALL(x)																			\
{																								\
	cudnnStatus_t status = (x);																	\
	if (status != CUDNN_STATUS_SUCCESS) {														\
		fprintf(stderr, "%s:%d ERROR: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status));	\
		exit(-1);																				\
	}																							\
}																								\

#define CUDA_CALL(x)\
{\
	cudaError_t err = (x);\
	if (err != cudaSuccess) {\
		fprintf(stderr, "%s:%d ERROR: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
		exit(-1);\
	}\
}\

double checksum (double *arr, int count) {
	double sum = 0;
	for (int i = 0; i < count; i++) {
		sum += arr[i];
	}
	return sum;
}

// C1 Convolution without tiling, shared memory
__global__ void C1 (double *I0, double *F, double *O, int C, int W, int H, int W0, int H0, int K, int FW, int FH) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	if (x < H && y < W) {
		int o_k_stride = W * H;
		int f_k_stride = C * FW * FH;
		int c_stride = W0 * H0;	
		int f_c_stride = FW * FH;
		int xy_linear = x * W + y;
		for (int k = 0; k < K; k++) {
			double val = 0.0;
			int A_f = k * f_k_stride;
			for (int c = 0; c < C; c++) {
				int A = c * c_stride;
				int B_f = A_f + c * f_c_stride;
				for (int j = 0; j < FH; j++) {
					int B = A + y + j;
					int C_f = B_f + (FH - 1 - j);
					for (int i = 0; i < FW; i++) {
						int I0_index = B + (x + i) * W0;
						int F_index = C_f + (FW - 1 - i) * FW;
						val += F[F_index] * I0[I0_index];
					}
				}
			}
			int O_index = k * o_k_stride + xy_linear;
			O[O_index] = val;
		}
	}
}

// C2 Convolution with tiling, shared memory
__global__ void C2 (double *I0, double *F, double *O, int C, int W, int H, int W0, int H0, int K, int FW, int FH) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	if (x < H && y < W) {
		int o_k_stride = W * H;
		int f_k_stride = C * FW * FH;
		int c_stride = W0 * H0;	
		int f_c_stride = FW * FH;
		int xy_linear = x * W + y;

		int TH = FW + blockDim.x - 1;
		int TW = FH + blockDim.y - 1;
		// Allocate and initialise shared memory
		extern __shared__ double tile[];
		for (int c = 0; c < C; c++) {
			for (int i = 0; i < FW + blockDim.x - 1; i += FW) {
				int tx = threadIdx.x + i;
				for (int j = 0; j < FH + blockDim.y - 1; j += FH) {
					int ty = threadIdx.y + j;
					if (tx < TH && ty < TW){
						int x1 = x + i;
						int y1 = y + j;
						int l = c * c_stride + x1 * W0 + y1;
						int l1 = c * TH * TW + TW * tx + ty;
						tile[l1] = I0[l];  
					}
				}
			}
		}
		__syncthreads();

		for (int k = 0; k < K; k++) {
			double val = 0.0;
			for (int c = 0; c < C; c++) {
				for (int j = 0; j < FH; j++) {
					for (int i = 0; i < FW; i++) {
						int tile_index = c * TH * TW + (threadIdx.x + i) * TW + (threadIdx.y + j);
						int F_index = k * f_k_stride + c * f_c_stride + (FW - 1 - i) * FW + (FH - 1 - j);
						val += F[F_index] * tile[tile_index];
					}
				}
			}
			int O_index = k * o_k_stride + xy_linear;
			O[O_index] = val;
		}
	}
}

// C3 Convolution with CUDNN
void C3 (double *I, double *F, double *O, int C, int W, int H, int K, int FW, int FH, struct timespec *start, struct timespec *end) {
	// Create the handle
	cudnnHandle_t cudnn;
	CUDNN_CALL(cudnnCreate(&cudnn));

	// Create the input/output tensor, filter, and convolution descriptors
	cudnnTensorDescriptor_t I_desc;
	CUDNN_CALL(cudnnCreateTensorDescriptor(&I_desc));
	CUDNN_CALL(cudnnSetTensor4dDescriptor(I_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W));
	
	cudnnTensorDescriptor_t O_desc;
	CUDNN_CALL(cudnnCreateTensorDescriptor(&O_desc));
	CUDNN_CALL(cudnnSetTensor4dDescriptor(O_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, H, W));
	
	cudnnFilterDescriptor_t F_desc;
	CUDNN_CALL(cudnnCreateFilterDescriptor(&F_desc));
	CUDNN_CALL(cudnnSetFilter4dDescriptor(F_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW));

	cudnnConvolutionDescriptor_t conv_desc;
	CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
	CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

	// Select the convolution algorithm
	cudnnConvolutionFwdAlgo_t conv_algo;
	CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(cudnn, I_desc, F_desc, conv_desc, O_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 10000, &conv_algo));

	// Determine the memory required by the algorithm
	size_t size_in_bytes;
	CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn, I_desc, F_desc, conv_desc, O_desc, conv_algo, &size_in_bytes));

	// Allocate memory for the workspace
	void *workspace;
	CUDA_CALL(cudaMalloc(&workspace, size_in_bytes));

	// Convolve
	double alpha = 1, beta = 0;
	clock_gettime(CLOCK_MONOTONIC, start);
	CUDNN_CALL(cudnnConvolutionForward(cudnn, &alpha, I_desc, I, F_desc, F, conv_desc, conv_algo, workspace, size_in_bytes, &beta, O_desc, O));
	CUDA_CALL (cudaDeviceSynchronize());
	clock_gettime(CLOCK_MONOTONIC, end);

	// Destroy handles and descriptors
	CUDNN_CALL(cudnnDestroy(cudnn));
	CUDNN_CALL(cudnnDestroyTensorDescriptor(I_desc));
	CUDNN_CALL(cudnnDestroyTensorDescriptor(O_desc));
	CUDNN_CALL(cudnnDestroyFilterDescriptor(F_desc));
	CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));

	// Free workspace memory
	CUDA_CALL(cudaFree(workspace));
}

void run_version (int version, int block_size, double *I_dev, double *I0_dev, double *F_dev, double *O_dev, double *O_host, int C, int W, int H, int W0, int H0, int K, int FW, int FH) {
	struct timespec start;
	struct timespec end;
	double time_usec;
	
	if (version == 1 || version == 2) {
		dim3 block_dim(block_size, block_size);
		int grid_x = W / block_size;
		int grid_y = W / block_size;
		dim3 grid_dim(grid_x, grid_y);	
		if (version == 1) {
			clock_gettime(CLOCK_MONOTONIC, &start);
			C1<<<grid_dim, block_dim>>>(I0_dev, F_dev, O_dev, C, W, H, W0, H0, K, FW, FH);
		} else {
			int tile_height = FW + block_size - 1;
			int tile_width = FH + block_size - 1;
			clock_gettime(CLOCK_MONOTONIC, &start);
			C2<<<grid_dim, block_dim, C * tile_height * tile_width * sizeof(double) >>>(I0_dev, F_dev, O_dev, C, W, H, W0, H0, K, FW, FH);
		}
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &end);
	} else if (version == 3) {
		C3 (I_dev, F_dev, O_dev, C, W, H, K, FW, FH, &start, &end);
	}
	int O_sz = K * W * H * sizeof(double);
	CUDA_CALL(cudaMemcpy(O_host, O_dev, O_sz, cudaMemcpyDeviceToHost));
	double cksum = checksum(O_host, K * W * H); 
	time_usec = ((double)end.tv_sec *1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000);
	printf("%.2lf, %4.6lf\n", cksum, time_usec/1000);
}

int main(int argc, char *argv[]) {
	
	// Tensor and filter dimensions
	int C = 3;
	int W = 4096;
	int H = 4096;
	int FW = 3;
	int FH = 3;
	int K = 10;
	int P = 1;

	// Initialise the input tensors I, I0
	int I_sz = C * W * H * sizeof(double);
	double *I = (double *) malloc (I_sz);
	int W0 = W + 2 * P;
	int H0 = H + 2 * P;
	int I0_sz = C * W0 * H0 * sizeof(double);
	double *I0 = (double *) malloc (I0_sz);
	int zs = W0 * H0;
	int i_max = H0 - P - 1;
	int j_max = W0 - P - 1;

	for (int c = 0; c < C; c++) {
		for (int i = 0; i < H0; i++) {
			for (int j = 0; j < W0; j++) {
				int l = c * zs + (W0 * i) + j;
				if (i < P || j < P || i > i_max || j > j_max) {
					I0[l] = 0;
				} else {
					int x = i - P;	
					int y = j - P;
					I0[l] = double(c * (x + y));
					int l2 = (c * W * H) + (x * W) + y;
					I[l2] = I0[l];
				}
			}
		}
	}
	
	// Initialise the filters
	int F_sz = K * C * FW * FH * sizeof(double);
	double *F = (double *) malloc(F_sz);

	for (int k = 0; k < K; k++) {
		for (int c = 0; c < C; c++) {
			for (int i = 0; i < FH; i++) {
				for (int j = 0; j < FW; j++) {
					int l = (k * C * FW * FH) + (c * FW * FH) + (i * FW) + j;
					F[l] = (c + k) * (i + j);
				}
			}
		}
	}

	// Allocate space for output on host
	int O_sz = K * W * H * sizeof(double);
	double *O = (double *) malloc(O_sz);

	// Allocate space for tensor, filters, and output on device
	double *I_dev;
	double *I0_dev;
	double *F_dev;
	double *O_dev;
	CUDA_CALL(cudaMalloc((void **)&I0_dev, I0_sz));
	CUDA_CALL(cudaMalloc((void **)&F_dev, F_sz));
	CUDA_CALL(cudaMalloc((void **)&O_dev, O_sz));

	// Copy tensor and filters to the device
	CUDA_CALL(cudaMemcpy(I0_dev, I0, I0_sz, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(F_dev, F, F_sz, cudaMemcpyHostToDevice));
	
	// C1, C2, C3
	run_version (1, 8, NULL, I0_dev, F_dev, O_dev, O, C, W, H, W0, H0, K, FW, FH);
	run_version (2, 8, NULL, I0_dev, F_dev, O_dev, O, C, W, H, W0, H0, K, FW, FH);
	CUDA_CALL(cudaFree(I0_dev));
	CUDA_CALL(cudaMalloc((void **)&I_dev, I_sz));
	CUDA_CALL(cudaMemcpy(I_dev, I, I_sz, cudaMemcpyHostToDevice));
	run_version (3, 0, I_dev, NULL, F_dev, O_dev, O, C, W, H, W0, H0, K, FW, FH);

	// Free all memory on host
	free(I);
	free(I0);
	free(F);
	free(O);

	// Free all memory on device
	CUDA_CALL(cudaFree(I_dev));
	CUDA_CALL(cudaFree(F_dev));
	CUDA_CALL(cudaFree(O_dev));
	return 0;
}
