#include"headerfiles.h"
__global__ void genacmatrix(float* attenuation_matrix, CTdims* ctdim, short* ct_matrix) {
	//do nothing
	////miu=1e-8
	//AM[X][Y][Z]
	//CT[X][Y][Z]
	//请用SIMPLEITK等工具将CT缩放+平移与PET图像同XYZ轴
	ctdim->x0 = -Nx * pixel_size/2.0f;
	ctdim->y0 = -Ny * pixel_size/2.0f;
	ctdim->z0 = -Nz * pixel_size / 2.0f;
	ctdim->xspacing = pixel_size;
	ctdim->yspacing = pixel_size;
	ctdim->zspacing = pixel_size;
	ctdim->xdim =Nx;
	ctdim->ydim =Ny;
	ctdim->zdim =Nz;
	for (int mx=0;mx< ctdim->xdim;mx++){
		for (int my = 0; my < ctdim->ydim; my++) {
			for (int mz = 0; mz < ctdim->zdim; mz++) {

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
				float rx = (mx - (ctdim->xdim - 1) / 2.0f) * ctdim->xspacing;
				float ry = (my - (ctdim->ydim - 1) / 2.0f) * ctdim->yspacing;
				float rz = (mz - (ctdim->zdim - 1) / 2.0f) * ctdim->zspacing;
				if ((rx * rx + ry * ry) < 2500.0f && rz <10.0f && rz>-10.0f) {
					attenuation_matrix[mz* ctdim->ydim * ctdim->xdim +my* ctdim->xdim +mx] = 9.598E-03*pixel_size ;
				}
				else {
					attenuation_matrix[mz * ctdim->ydim * ctdim->xdim + my * ctdim->xdim + mx] = 0;
				}//YZX

			}
		}

		
	}__syncthreads();

}