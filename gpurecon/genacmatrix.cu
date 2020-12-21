#include"headerfiles.h"
__global__ void genctdim(CTdims* dev_ctdim);
__global__ void genctdim(CTdims* dev_ctdim) {
	dev_ctdim->x0 = -Nx * pixel_size / 2.0f;
	dev_ctdim->y0 = -Ny * pixel_size / 2.0f;
	dev_ctdim->z0 = -Nz * pixel_size / 2.0f;
	dev_ctdim->xspacing = pixel_size;
	dev_ctdim->yspacing = pixel_size;
	dev_ctdim->zspacing = pixel_size;
	dev_ctdim->xdim = Nx;
	dev_ctdim->ydim = Ny;
	dev_ctdim->zdim = Nz;
}
int genacmatrix(float* dev_attenuation_matrix, CTdims* dev_ctdim, short* dev_ct_matrix) {
	genctdim<<<1,1>>>(dev_ctdim);
	cudaDeviceSynchronize();

	genacvalue << <256, 512 >> > (dev_attenuation_matrix, dev_ctdim, nullptr);
	cudaDeviceSynchronize();
	return 0;
}
__global__ void genacvalue(float* attenuation_matrix, CTdims* dev_ctdim, short* dev_ct_matrix) {
	//do nothing
	////miu=1e-8
	//AM[X][Y][Z]
	//CT[X][Y][Z]
	//请用SIMPLEITK等工具将CT缩放+平移与PET图像同XYZ轴
	int xdim = dev_ctdim->xdim;
	int ydim = dev_ctdim->ydim;
	int zdim = dev_ctdim->zdim;
	int totalvoxels = xdim * ydim * zdim;
	for (int line_index = threadIdx.x + blockIdx.x * blockDim.x; line_index < totalvoxels; line_index += blockDim.x * gridDim.x) {
		int mz = line_index / (xdim * ydim);
		int temp = line_index - mz * (xdim * ydim);
		int my = temp / xdim;
		int mx = temp - my * xdim;

		//short value = dev_ct_matrix[mx*Ny*Nz+my*Nz+mz];
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
		float rx = ((float)mx + 0.5) * dev_ctdim->xspacing + dev_ctdim->x0;
		float ry = ((float)my + 0.5) * dev_ctdim->yspacing + dev_ctdim->y0;
		float rz = ((float)mz + 0.5) * dev_ctdim->zspacing + dev_ctdim->z0;
		if ((rx * rx + ry * ry) < 2500.0f && rz <10.0f && rz>-10.0f) {
			attenuation_matrix[mz* dev_ctdim->ydim * dev_ctdim->xdim +my* dev_ctdim->xdim +mx] = 9.598E-03;
		}
		else {
			attenuation_matrix[mz * dev_ctdim->ydim * dev_ctdim->xdim + my * dev_ctdim->xdim + mx] = 0;
		}//YZX		

	}__syncthreads();
}