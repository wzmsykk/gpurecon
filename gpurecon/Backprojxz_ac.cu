#include"headerfiles.h"
__global__ void Backprojxz_ac(float* dev_image, float* back_image, CUDAlor* lines, int linesN, int backProjOnly)
{
	//current slice YZX
	__shared__ float slice[Nx * Nz];
	int img_slice = Nx * Nz;
	float max_distance = TOR_WIDTH * TOR_WIDTH / 2.0f;

	// loop over slices
	for (int current_slice = 0; current_slice < Ny; ++current_slice) {

		int offset = current_slice * img_slice;
		for (int vi = threadIdx.x; vi < img_slice; vi += blockDim.x) {

			slice[vi] = 0; // back_image[vi+offset] = 0; 
		}
		__syncthreads();
		for (int line = threadIdx.x + blockIdx.x * blockDim.x; line < linesN; line += blockDim.x * gridDim.x) {

			// convert pointer type to struct
			CUDAlor* the_line = (CUDAlor*)lines + line;

			// line direction
			float l1 = the_line->dx, l2 = the_line->dz;
			float t = (current_slice - the_line->y0) / the_line->dy;
			// point O
			float x = the_line->x0 + t * l1, z = the_line->z0 + t * l2;
			int centerX = floor(x + 0.5), centerZ = floor(z + 0.5);
			// search pixels within the tube (eclipse)
			for (int xx = centerX - TOR_WIDTH; xx <= centerX + TOR_WIDTH; ++xx) {
				for (int zz = centerZ - TOR_WIDTH; zz <= centerZ + TOR_WIDTH; ++zz) {
					if (xx >= 0 && xx < Nx && zz >= 0 && zz < Nz) {
						float dx = xx - centerX, dz = zz - centerZ;
						float inner = dx * l1 + dz * l2;

						// Distance to the line, squared 
						float d2 = dx * dx + dz * dz - inner * inner;
						float kern = (d2 < max_distance) ? exp(-d2 * ISIGMA) : 0;
						float divvalue = backProjOnly > 0 ? 1 : the_line->value;
						float thevalue = ((the_line->value) > 0) ? (kern / divvalue) : 0;
						float acvalue = the_line->attcorrvalue;
						float finvalue = thevalue/acvalue;
						atomicAdd(&slice[xx + Nx * zz], finvalue);
					} //endif
				} // endfor Write the value back to global memory 
			} // endfor

		}__syncthreads();	// endfor line
		for (int vi = threadIdx.x; vi < img_slice; vi += blockDim.x) {
			back_image[vi + offset] += slice[vi];
		}  __syncthreads();
	} // endfor slice
} //end function
