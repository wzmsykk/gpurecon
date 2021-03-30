#include"headerfiles.h"
__global__ void Frotate(float * dst_image, float * src_image)
{
	for (int i=0;i<Nx;i++){
		for (int j = threadIdx.x+blockIdx.x*blockDim.x;j<Nz*Ny;j += blockDim.x * gridDim.x){
			dst_image[j+Ny*Nz*i] = src_image[(j%Ny)*Nz*Nx + (j/Ny)*Nx + i];
		}	
	}
}
//backimg==planeXZ
//backtempimg==planeYZ
//SRC[Y][Z][X]->DST[X][Z][Y]
//j%Ny=y j/Ny=z
//j=z*N_y+y
__global__ void Brotate(float* dst_image, float* src_image)
{
	for (int i = 0; i < Ny; i++) {
		for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < Nz * Nx; j += blockDim.x * gridDim.x) {
			dst_image[j + Nz * Nx * i] = src_image[(j % Nx) * Nz * Ny + (j / Nx) * Ny + i];
		}
	}
}
//backimg==planeXZ
//backtempimg==planeYZ
//SRC[X][Z][Y]->DST[Y][Z][X]
//j%Nx=x j/Nx=z
//j=z*Nx+x
__global__ void Rrotate(float* dst_image, float* src_image) {
	for (int i = 0; i < Nz; i++) {
		for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < Ny * Nx; j += blockDim.x * gridDim.x) {
			dst_image[j + Ny * Nx * i] = src_image[j % Nx + i * Nx + (j/Nx)*Nz*Nx];
		}
	}
}
//planeXZ->planeXY
//imageYZX==imageZYX
//SRC[Y][Z][X]->DST[Z][Y][X]
//i=z j%Nx=x j/Nx=y
//j=y*Nx+x
//