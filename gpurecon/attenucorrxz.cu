#include"headerfiles.h"
__global__ void attenucorrxz(float* lines, int linesN, CTdims* CTdim, float* attenuation_matrix) {

	int img_slice = Nx * Nz;

	for (int line = threadIdx.x + blockIdx.x * blockDim.x; line < linesN; line += blockDim.x * gridDim.x) {

		// convert pointer type to struct
		CUDAlor* the_line = (CUDAlor*)lines + line;		

		//����˥������
		float y0 = the_line->y0;//lor�߶˵�0
		float y1 = the_line->y1;//lor�߶˵�1
		int y0r = floor(y0+0.5);//�˵�0����yslice
		int y1r = floor(y1+0.5);//�˵�1����yslice
		//����dy>dx��dy>dz
		//yΪ��󲽽�����,y�ı�1,x,z�ı��С��1
		// line direction
		float l1 = the_line->dx, l2 = the_line->dz;
		float atf = 0;

		//��ά����ֱ�߱���
		for (int current_slice = 0; current_slice < Ny; ++current_slice) {
			float tin = float(current_slice - y0) / the_line->dy;//����ò�ʱt
			// point (yi,zi,xi)->point(yi+1,zi+1,xi+1)
			float tout = float(current_slice + 1 - y0) / the_line->dy;//�뿪�ò�ʱt
			float xin = the_line->x0 + tin * l1, zin = the_line->z0 + tin * l2;//����ò�ʱx,z
			float xout = the_line->x0 + tout * l1, zout = the_line->z0 + tout * l2;//�뿪�ò�ʱx,z
			if (the_line->dx > the_line->dz) {
				       
			}
			int centerx = floor(xin + 0.5), centerz = floor(zin + 0.5);
			atf += float(pixel_size) /abs( the_line->dy) * attenuation_matrix[img_slice * current_slice + Nx * centerz + centerx];

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