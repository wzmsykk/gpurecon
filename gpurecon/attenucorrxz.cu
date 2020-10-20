#include"headerfiles.h"
__global__ void attenucorrxz(float* lines, int linesN, float* attenuation_matrix) {

	int img_slice = Nx * Nz;

	for (int line = threadIdx.x + blockIdx.x * blockDim.x; line < linesN; line += blockDim.x * gridDim.x) {

		// convert pointer type to struct
		CUDAlor* the_line = (CUDAlor*)lines + line;		

		//计算衰减参数
		float y0 = the_line->y0;
		float y1 = the_line->y1;
		int y0r = floor(y0+0.5);
		int y1r = floor(y1+0.5);
		int swap = 0;
		if (y0r > y1r) {
			swap = y0r;
			y0r = y1r;
			y1r = swap;
		}
		//由于dy>dx且dy>dz
		//y为最大步进方向,y改变1,x,z改变均小于1
		// line direction
		float l1 = the_line->dx, l2 = the_line->dz;
		float atf = 0;
		for (int ysli = y0r; ysli < y1r; ++ysli) {
			float t = float(ysli - y0) / the_line->dy;
			// point (yi,zi,xi)
			float x = the_line->x0 + t * l1, z = the_line->z0 + t * l2;
			int xx = floor(x + 0.5), zz = floor(z + 0.5);
			atf += 1.0 /abs( the_line->dy) * attenuation_matrix[img_slice * ysli + Nx * zz + xx];

		}
		if (atf > 0) {
			float cv = exp(-atf);
			the_line->attcorrvalue = cv;
		}
		else {
			the_line->attcorrvalue = 1.0;
		}
		
	}
	__syncthreads();
}