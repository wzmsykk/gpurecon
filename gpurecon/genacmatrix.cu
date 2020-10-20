#include"headerfiles.h"
__global__ void genacmatrix(float* attenuation_matrix, short * ct_matrix) {
	//do nothing
	////miu=1e-8
	//AM[X][Y][Z]
	//CT[X][Y][Z]
	//����SIMPLEITK�ȹ��߽�CT����+ƽ�Ƶ���ʵ�ռ�λ��,����AMͬ��SPACE��DIM
	for (int mx=0;mx<Nx;mx++){
		for (int my = 0; my < Ny; my++) {
			for (int mz = 0; mz < Nz; mz++) {

				//short value = ct_matrix[mx*Ny*Nz+my*Nz+mz];
				double at = 0;
				//CT HU TO ATTENUATION LIST
				//H20 9.598E-02
				//if (value<A && ){
				//atv=CONST01
				//}else if ..
				//TO DO 
				//����CT ��HUֵ����˥��
				//����ˮģ��2cm
				//�뾶5cm
				float rx = (mx - (Nx - 1) / 2.0f) * (float)pixel_size;
				float ry = (my - (Ny - 1) / 2.0f) * (float)pixel_size;
				float rz = (mz - (Nz - 1) / 2.0f) * (float)pixel_size;
				if ((rx * rx + ry * ry) < 2500.0f && rz <10.0f && rz>-10.0f) {
					attenuation_matrix[my*Nz*Nx+mz*Nx+mx] = 9.598E-02 * (float)pixel_size*0;
				}
				else {
					attenuation_matrix[my * Nz * Nx + mz * Nx + mx] = 0;
				}

			}
		}

		
	}__syncthreads();

}