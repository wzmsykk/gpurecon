#include"headerfiles.h"
__global__ void convertolorxz(short *dev_lor_data_array, int *dev_indexymax,float *lines, int nlines, int noffset)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nlines; i += blockDim.x * gridDim.x) { 

	short rsectorID1, rsectorID2, moduleID1, moduleID2, crystalID1, crystalID2;
	rsectorID1= *(dev_lor_data_array+6*dev_indexymax[(i+noffset)]+0);
	moduleID1 = *(dev_lor_data_array+6*dev_indexymax[(i+noffset)]+1);
	crystalID1 = *(dev_lor_data_array+6*dev_indexymax[(i+noffset)]+2);
	rsectorID2 = *(dev_lor_data_array+6*dev_indexymax[(i+noffset)]+3);
	moduleID2 = *(dev_lor_data_array+6*dev_indexymax[(i+noffset)]+4);
	crystalID2 = *(dev_lor_data_array+6*dev_indexymax[(i+noffset)]+5);
		
	Lorposition lor = CalcLorPositionFull(rsectorID1,rsectorID2,moduleID1, moduleID2, crystalID1, crystalID2);
	int totalparams = CUDAlor_size;

	*(lines + 0 + totalparams * i) = lor.x0/pixel_size + (Nx-1)/2.0f;
	*(lines + 1 + totalparams * i) = lor.x1 / pixel_size + (Nx - 1) / 2.0f;
	*(lines + 2 + totalparams * i) = (lor.x1-lor.x0)/sqrt((lor.x1-lor.x0)*(lor.x1-lor.x0)+(lor.y1-lor.y0)*(lor.y1-lor.y0)+(lor.z1-lor.z0)*(lor.z1-lor.z0));
	*(lines + 3 + totalparams * i) = lor.y0/pixel_size + (Ny-1)/2.0f;
	*(lines + 4 + totalparams * i) = lor.y1 / pixel_size + (Ny - 1) / 2.0f;
	*(lines + 5 + totalparams * i) = (lor.y1-lor.y0)/sqrt((lor.x1-lor.x0)*(lor.x1-lor.x0)+(lor.y1-lor.y0)*(lor.y1-lor.y0)+(lor.z1-lor.z0)*(lor.z1-lor.z0));
	*(lines + 6 + totalparams * i) = lor.z0/pixel_size + (Nz-1)/2.0f;
	*(lines + 7 + totalparams * i) = lor.z1 / pixel_size + (Nz - 1) / 2.0f;
	*(lines + 8 + totalparams * i) = (lor.z1-lor.z0)/sqrt((lor.x1-lor.x0)*(lor.x1-lor.x0)+(lor.y1-lor.y0)*(lor.y1-lor.y0)+(lor.z1-lor.z0)*(lor.z1-lor.z0));
	*(lines + 9 + totalparams * i) = 0.0;
	*(lines + 10 + totalparams * i) = 1.0;
	/************************************************
	lines的结构
	lines[x][0]=lor.x0所处像素位置
	lines[x][1]=lor.x1所处像素位置
	lines[x][2]=lor.x方向导数
	lines[x][3]=lor.y0所处像素位置
	lines[x][4]=lor.y1所处像素位置
	lines[x][5]=lor.y方向导数
	lines[x][6]=lor.z0所处像素位置
	lines[x][7]=lor.z1所处像素位置
	lines[x][8]=lor.z方向导数
	lines[x][9]=0.0 前向投影value
	lines[x][10]=1.0 衰减校正值
	************************************************/
	}
__syncthreads();
}

