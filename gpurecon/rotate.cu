#include"headerfiles.h"
__global__ void Frotate(float *back_image, float *back_imagetemp)
{
	for (int i=0;i<Nx;i++){
		for (int j = threadIdx.x+blockIdx.x*blockDim.x;j<Nz*Ny;j += blockDim.x * gridDim.x){
		back_imagetemp[j+Ny*Nz*i] = back_image[(j%Ny)*Nz*Nx + (j/Ny)*Nx + i];
		}	
	}
}
//backimg==planeXZ
//backtempimg==planeYZ
//[Y][Z][X]->[X][Z][Y]
//j%Ny=y j/Ny=z
__global__ void Brotate(float* back_image, float* back_imagetemp)
{
	for (int i = 0; i < Ny; i++) {
		for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < Nz * Nx; j += blockDim.x * gridDim.x) {
			back_imagetemp[j + Nz * Nx * i] = back_image[(j % Nx) * Nz * Ny + (j / Nx) * Ny + i];
		}
	}
}
//backimg==planeXZ
//backtempimg==planeYZ
//[X][Z][Y]->[Y][Z][X]
//j%Nx=x j/Nx=z
__global__ void Rrotate(float* imageYZX, float* imageZYX) {
	for (int i = 0; i < Nz; i++) {
		for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < Ny * Nx; j += blockDim.x * gridDim.x) {
			imageZYX[j + Ny * Nx * i] = imageYZX[j % Nx + i * Nx + (j/Nx)*Nz*Nx];
		}
	}
}
//planeXZ->planeXY
//imageYZX==imageZYX
//[Y][Z][X]->[Z][Y][X]
//i=z j%Nx=x j/Nx=y
//j=x+Nx*y
//