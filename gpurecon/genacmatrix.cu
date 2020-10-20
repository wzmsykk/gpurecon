#include"headerfiles.h"
__global__ void genacmatrix(float* attenuation_matrix, short * ct_matrix) {
	//do nothing
	////miu=1e-8
	//AM[X][Y][Z]
	//CT[X][Y][Z]
	//请用SIMPLEITK等工具将CT缩放+平移到真实空间位置,且与AM同样SPACE与DIM
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
				//根据CT 的HU值查找衰减
				//现在水模高2cm
				//半径5cm
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