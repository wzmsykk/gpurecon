#include"headerfiles.h"
__global__ void attenucorryz(float* lines, int linesN, float* attenuation_matrix) {

	int img_slice = Ny * Nz;

	for (int line = threadIdx.x + blockIdx.x * blockDim.x; line < linesN; line += blockDim.x * gridDim.x) {

		// convert pointer type to struct
		CUDAlor* the_line = (CUDAlor*)lines + line;

		//计算衰减参数
		float x0 = the_line->y0;
		float x1 = the_line->y1;
		int x0r = floor(x0 + 0.5);
		int x1r = floor(x1 + 0.5);
		int swap = 0;
		if (x0r > x1r) {
			swap = x0r;
			x0r = x1r;
			x1r = swap;
		}
		//由于dx>dy且dx>dz
		//x为最大步进方向,x改变1,y,z改变均小于1
		float dl = sqrt(the_line->dx * the_line->dx + the_line->dy * the_line->dy + the_line->dz * the_line->dz);
		// line direction
		float l1 = the_line->dy, l2 = the_line->dz;
		float atf = 0;
		for (int xsli = x0r; xsli < x1r; ++xsli) {
			float t = float(xsli - x0) / the_line->dx;
			// point (xi,zi,yi)
			float y = the_line->y0 + t * l1, z = the_line->z0 + t * l2;
			int yy = floor(y + 0.5), zz = floor(z + 0.5);
			atf += 1.0 * dl / the_line->dx * attenuation_matrix[img_slice * xsli + Ny * zz + yy];
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