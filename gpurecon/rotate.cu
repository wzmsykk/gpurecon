#include"headerfiles.h"
__global__ void Frotate(float *back_image, float *back_imagetemp)
{
	for (int i=0;i<Nx;i++){
		for (int j = threadIdx.x+blockIdx.x*blockDim.x;j<Ny*Nz;j += blockDim.x * gridDim.x){
		back_imagetemp[j+Ny*Nz*i] = back_image[(j%Ny)*Ny*Nz + (j/Ny)*Ny + i];
		}	
	}
}
