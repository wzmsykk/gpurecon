#include"headerfiles.h"
__global__ void genacmatrix(float* attenuation_matrix, CTdims* CTdim, short* ct_matrix) {
	//do nothing
	////miu=1e-8
	//AM[X][Y][Z]
	//CT[X][Y][Z]
	//请用SIMPLEITK等工具将CT缩放+平移与PET同XYZ轴
	CTdim->x0 = -Nx * pixel_size/2.0f;
	CTdim->y0 = -Ny * pixel_size/2.0f;
	CTdim->z0 = -Nz * pixel_size / 2.0f;
	CTdim->xspacing = pixel_size;
	CTdim->yspacing = pixel_size;
	CTdim->zspacing = pixel_size;
	CTdim->xdim =Nx;
	CTdim->ydim =Ny;
	CTdim->zdim =Nz;
	for (int mx=0;mx< CTdim->xdim;mx++){
		for (int my = 0; my < CTdim->ydim; my++) {
			for (int mz = 0; mz < CTdim->zdim; mz++) {

				//short value = ct_matrix[mx*Ny*Nz+my*Nz+mz];
				double at = 0;
				//CT HU TO ATTENUATION LIST
				//H20 9.598E-02 cm-1=9.598E-3 mm-1
				//if (value<A && ){
				//atv=CONST01
				//}else if ..
				//TO DO 
				//根据CT 的HU值查找衰减
				//现在水模高2cm
				//半径5cm
				float rx = (mx - (CTdim->xdim - 1) / 2.0f) * CTdim->xspacing;
				float ry = (my - (CTdim->ydim - 1) / 2.0f) * CTdim->yspacing;
				float rz = (mz - (CTdim->zdim - 1) / 2.0f) * CTdim->zspacing;
				if ((rx * rx + ry * ry) < 2500.0f && rz <10.0f && rz>-10.0f) {
					attenuation_matrix[my* CTdim->zdim * CTdim->xdim +mz* CTdim->xdim +mx] = 9.598E-03 ;
				}
				else {
					attenuation_matrix[my * CTdim->zdim * CTdim->xdim + mz * CTdim->xdim + mx] = 0;
				}

			}
		}

		
	}__syncthreads();

}