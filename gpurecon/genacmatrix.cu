#include"headerfiles.h"
__global__ void genacmatrix(float* attenuation_matrix) {
	//do nothing
	////miu=0
	for (int i = threadIdx.x; i < Nx * Ny * Nz; i += blockDim.x) {
		attenuation_matrix[i] = 1e-8;
	}
}