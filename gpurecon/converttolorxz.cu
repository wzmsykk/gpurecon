#include"headerfiles.h"
__global__ void convertolorxz(short* dev_lor_data_array, int* dev_indexymax, CUDAlor* lines, int nlines, int noffset)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nlines; i += blockDim.x * gridDim.x) {
		


		short rsectorID1, rsectorID2, moduleID1, moduleID2, crystalID1, crystalID2;
		rsectorID1 = *(dev_lor_data_array + 6 * dev_indexymax[(i + noffset)] + 0);
		moduleID1 = *(dev_lor_data_array + 6 * dev_indexymax[(i + noffset)] + 1);
		crystalID1 = *(dev_lor_data_array + 6 * dev_indexymax[(i + noffset)] + 2);
		rsectorID2 = *(dev_lor_data_array + 6 * dev_indexymax[(i + noffset)] + 3);
		moduleID2 = *(dev_lor_data_array + 6 * dev_indexymax[(i + noffset)] + 4);
		crystalID2 = *(dev_lor_data_array + 6 * dev_indexymax[(i + noffset)] + 5);

		Lorposition lor = CalcLorPositionFull(rsectorID1, rsectorID2, moduleID1, moduleID2, crystalID1, crystalID2);
		CUDAlor* this_line = (CUDAlor*)lines + i;

		

		this_line->x0 = lor.x0 / pixel_size + (Nx - 1) / 2.0f;
		this_line->x1 = lor.x1 / pixel_size + (Nx - 1) / 2.0f;
		this_line->dx = (lor.x1 - lor.x0) / sqrt((lor.x1 - lor.x0) * (lor.x1 - lor.x0) + (lor.y1 - lor.y0) * (lor.y1 - lor.y0) + (lor.z1 - lor.z0) * (lor.z1 - lor.z0));
		this_line->y0 = lor.y0 / pixel_size + (Ny - 1) / 2.0f;
		this_line->y1 = lor.y1 / pixel_size + (Ny - 1) / 2.0f;
		this_line->dy = (lor.y1 - lor.y0) / sqrt((lor.x1 - lor.x0) * (lor.x1 - lor.x0) + (lor.y1 - lor.y0) * (lor.y1 - lor.y0) + (lor.z1 - lor.z0) * (lor.z1 - lor.z0));
		this_line->z0 = lor.z0 / pixel_size + (Nz - 1) / 2.0f;
		this_line->z1 = lor.z1 / pixel_size + (Nz - 1) / 2.0f;
		this_line->dz = (lor.z1 - lor.z0) / sqrt((lor.x1 - lor.x0) * (lor.x1 - lor.x0) + (lor.y1 - lor.y0) * (lor.y1 - lor.y0) + (lor.z1 - lor.z0) * (lor.z1 - lor.z0));
		this_line->rx0 = lor.x0;
		this_line->rx1 = lor.x1;
		this_line->ry0 = lor.y0;
		this_line->ry1 = lor.y1;
		this_line->rz0 = lor.z0;
		this_line->rz1 = lor.z1;
		this_line->value = 0.0;
		this_line->attcorrvalue = 1.0;

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

__global__ void convertolorxz_ac(short* dev_lor_data_array, int* dev_indexymax, CUDAlor* lines, float* linesxz_attvalue_list, int nlines, int noffset)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nlines; i += blockDim.x * gridDim.x) {



		short rsectorID1, rsectorID2, moduleID1, moduleID2, crystalID1, crystalID2;
		rsectorID1 = *(dev_lor_data_array + 6 * dev_indexymax[(i + noffset)] + 0);
		moduleID1 = *(dev_lor_data_array + 6 * dev_indexymax[(i + noffset)] + 1);
		crystalID1 = *(dev_lor_data_array + 6 * dev_indexymax[(i + noffset)] + 2);
		rsectorID2 = *(dev_lor_data_array + 6 * dev_indexymax[(i + noffset)] + 3);
		moduleID2 = *(dev_lor_data_array + 6 * dev_indexymax[(i + noffset)] + 4);
		crystalID2 = *(dev_lor_data_array + 6 * dev_indexymax[(i + noffset)] + 5);

		Lorposition lor = CalcLorPositionFull(rsectorID1, rsectorID2, moduleID1, moduleID2, crystalID1, crystalID2);
		CUDAlor* this_line = (CUDAlor*)lines + i;

		float att_value_for_this_line = linesxz_attvalue_list[noffset + i];

		this_line->x0 = lor.x0 / pixel_size + (Nx - 1) / 2.0f;
		this_line->x1 = lor.x1 / pixel_size + (Nx - 1) / 2.0f;
		this_line->dx = (lor.x1 - lor.x0) / sqrt((lor.x1 - lor.x0) * (lor.x1 - lor.x0) + (lor.y1 - lor.y0) * (lor.y1 - lor.y0) + (lor.z1 - lor.z0) * (lor.z1 - lor.z0));
		this_line->y0 = lor.y0 / pixel_size + (Ny - 1) / 2.0f;
		this_line->y1 = lor.y1 / pixel_size + (Ny - 1) / 2.0f;
		this_line->dy = (lor.y1 - lor.y0) / sqrt((lor.x1 - lor.x0) * (lor.x1 - lor.x0) + (lor.y1 - lor.y0) * (lor.y1 - lor.y0) + (lor.z1 - lor.z0) * (lor.z1 - lor.z0));
		this_line->z0 = lor.z0 / pixel_size + (Nz - 1) / 2.0f;
		this_line->z1 = lor.z1 / pixel_size + (Nz - 1) / 2.0f;
		this_line->dz = (lor.z1 - lor.z0) / sqrt((lor.x1 - lor.x0) * (lor.x1 - lor.x0) + (lor.y1 - lor.y0) * (lor.y1 - lor.y0) + (lor.z1 - lor.z0) * (lor.z1 - lor.z0));
		this_line->rx0 = lor.x0;
		this_line->rx1 = lor.x1;
		this_line->ry0 = lor.y0;
		this_line->ry1 = lor.y1;
		this_line->rz0 = lor.z0;
		this_line->rz1 = lor.z1;
		this_line->value = 0.0;
		this_line->attcorrvalue = att_value_for_this_line;

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